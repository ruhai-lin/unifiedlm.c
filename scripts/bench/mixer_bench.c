// Per-token latency of the block's memory mixer at batch 1, single thread, fp32, for the layouts the
// unification experiment compares. Not run.c: only the parts that differ between layouts are timed,
// i.e. the address projection(s), the static read, the fast state update and read. The value projection,
// conv, norm and output head are the same for every layout and are left out.
//
//   S-merged   run.c's form: U = lam U + (1-lam) mu w + v k^T, y = U q          (one address)
//   S-split    y = mu (q w) + S q, S = lam S + v k^T                          (one address)
//   S-lazy     S = a H with a scalar a, H += v k^T / a, rescaled when a gets small (one address)
//   S-fused    S-split with the static read and the fast update and read in one pass over mu and S,
//              the same memory traffic as S-merged; S-fused-lazy is the same with the lazy decay
//   P          y = mu r_s + S_f q_f with its own address for each path (eager or lazy fast state);
//              P-free-fused reads mu and S_f in one pass like S-fused, so S-fused vs P-free-fused differ
//              only in the second address
//   A          alternating layers: fast-only, static-only
//
// Each scale then times the whole model for one token: every block with its RMSNorm, conv, value
// projection and residual, plus the final RMSNorm and the output head.
//
// $ cc -O3 -ffast-math -mcpu=native -o mixer_bench scripts/bench/mixer_bench.c && ./mixer_bench
// -ffast-math lets the compiler vectorize the float reductions; without it every layout is scalar-bound.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float *buf(size_t n) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = (float)rand() / (float)RAND_MAX * 0.02f - 0.01f;
    return p;
}

static void matvec(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float s = 0;
        const float *w = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) s += w[c] * x[c];
        y[r] = s;
    }
}

static void code(float *z, int m) {  // normalized relu^2, the address
    float s = 1.0f;
    for (int a = 0; a < m; a++) { float v = z[a] > 0 ? z[a] * z[a] : 0; z[a] = v; s += v; }
    for (int a = 0; a < m; a++) z[a] /= s;
}

typedef struct {
    int m_s, m_f, shared, merged, lazy, fused;
    float *Ks, *Kf, *mu, *S, *w, *c, *rs, *rf, *rprev, *q, scale;
} Layer;

static Layer make(int d, int m_s, int m_f, int shared, int merged, int lazy, int fused) {
    Layer L = {m_s, m_f, shared, merged, lazy, fused};
    int m = m_s > m_f ? m_s : m_f;
    L.Ks = m_s ? buf((size_t)m_s * d) : NULL;
    L.Kf = (m_f && !shared) ? buf((size_t)m_f * d) : NULL;
    L.mu = m_s ? buf((size_t)d * m_s) : NULL;
    L.S = m_f ? buf((size_t)d * m_f) : NULL;
    L.w = buf(m); L.c = buf(m); L.rs = buf(m); L.rf = buf(m); L.rprev = buf(m); L.q = buf(m);
    for (int a = 0; a < m; a++) { L.w[a] = 4.0f; L.c[a] = 4.0f; }
    L.scale = 1.0f;
    return L;
}

static void step(Layer *L, const float *u, const float *v, float *y, int d, float lam) {
    int m_s = L->m_s, m_f = L->m_f;
    if (m_s) { matvec(L->Ks, u, L->rs, m_s, d); code(L->rs, m_s); }
    float *rf = L->shared ? L->rs : L->rf;
    if (m_f && !L->shared) { matvec(L->Kf, u, L->rf, m_f, d); code(L->rf, m_f); }
    if (m_f) {  // evidence c and posterior q, O(m)
        for (int a = 0; a < m_f; a++) { L->c[a] = lam * (L->c[a] - L->w[a]) + L->w[a] + L->rprev[a]; L->q[a] = rf[a] / L->c[a]; }
    }
    if (L->merged) {  // run.c:244-255, U stored in L->S, mu read for the prior restore
        float *qw = L->q;
        for (int i = 0; i < d; i++) {
            float *Ui = L->S + (size_t)i * m_f; const float *mui = L->mu + (size_t)i * m_f;
            float vi = v[i], yi = 0;
            for (int a = 0; a < m_f; a++) {
                Ui[a] = lam * Ui[a] + (1.0f - lam) * mui[a] * L->w[a] + vi * L->rprev[a];
                yi += Ui[a] * qw[a];
            }
            y[i] = yi;
        }
    } else if (L->fused) {  // one pass over mu and S per row; needs m_s == m_f. Static read mu (q w) for S, mu r_s for P
        float qw[4096];
        for (int a = 0; a < m_f; a++) qw[a] = L->shared ? L->q[a] * L->w[a] : L->rs[a];
        float sc = 1.0f, inv = 1.0f;
        if (L->lazy) { L->scale *= lam; sc = L->scale; inv = 1.0f / sc; }
        for (int i = 0; i < d; i++) {
            float *Si = L->S + (size_t)i * m_f; const float *mui = L->mu + (size_t)i * m_f;
            float vi = v[i] * inv, ss = 0, sf = 0;
            if (L->lazy) {
                for (int a = 0; a < m_f; a++) { Si[a] += vi * L->rprev[a]; sf += Si[a] * L->q[a]; ss += mui[a] * qw[a]; }
            } else {
                for (int a = 0; a < m_f; a++) { Si[a] = lam * Si[a] + vi * L->rprev[a]; sf += Si[a] * L->q[a]; ss += mui[a] * qw[a]; }
            }
            y[i] = ss + sc * sf;
        }
        if (L->lazy && L->scale < 1e-6f) {
            for (size_t k = 0; k < (size_t)d * m_f; k++) L->S[k] *= L->scale;
            L->scale = 1.0f;
        }
    } else {
        for (int i = 0; i < d; i++) y[i] = 0;
        if (m_s) {  // static read: mu (q w) when shared (S), mu r_s otherwise (P, A)
            const float *x = L->rs;
            float tmp[4096];
            if (L->shared) { for (int a = 0; a < m_s; a++) tmp[a] = L->q[a] * L->w[a]; x = tmp; }
            for (int i = 0; i < d; i++) {
                const float *mui = L->mu + (size_t)i * m_s; float s = 0;
                for (int a = 0; a < m_s; a++) s += mui[a] * x[a];
                y[i] += s;
            }
        }
        if (m_f && !L->lazy) {
            for (int i = 0; i < d; i++) {
                float *Si = L->S + (size_t)i * m_f; float vi = v[i], s = 0;
                for (int a = 0; a < m_f; a++) { Si[a] = lam * Si[a] + vi * L->rprev[a]; s += Si[a] * L->q[a]; }
                y[i] += s;
            }
        } else if (m_f) {
            L->scale *= lam;
            float inv = 1.0f / L->scale;
            for (int i = 0; i < d; i++) {
                float *Hi = L->S + (size_t)i * m_f; float vi = v[i] * inv, s = 0;
                for (int a = 0; a < m_f; a++) { Hi[a] += vi * L->rprev[a]; s += Hi[a] * L->q[a]; }
                y[i] += L->scale * s;
            }
            if (L->scale < 1e-6f) {  // rescale, amortized
                for (size_t k = 0; k < (size_t)d * m_f; k++) L->S[k] *= L->scale;
                L->scale = 1.0f;
            }
        }
    }
    if (m_f) memcpy(L->rprev, rf, m_f * sizeof(float));
}

typedef struct { const char *name; int m_s, m_f, shared, merged, lazy, alternate, fused; } Cfg;

typedef struct { float *nw, *conv, *cbuf, *Wv; } Shell;  // the parts every layout has: norm, conv, value

static void rmsnorm(float *o, const float *x, const float *w, int d) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / d + 1e-5f);
    for (int i = 0; i < d; i++) o[i] = x[i] * ss * w[i];
}

// One whole block as run.c computes it: RMSNorm, causal depthwise conv (width 4), value projection
// (only for layers with a fast path), the mixer, residual add.
static void block(Layer *L, Shell *B, float *x, float *h, float *u, float *v, float *y, int d, float lam) {
    rmsnorm(h, x, B->nw, d);
    for (int i = 0; i < d; i++) {
        float *st = B->cbuf + i * 3, s = B->conv[i * 4 + 3] * h[i];
        for (int j = 0; j < 3; j++) s += B->conv[i * 4 + j] * st[j];
        st[0] = st[1]; st[1] = st[2]; st[2] = h[i];
        u[i] = s;
    }
    if (L->m_f) matvec(B->Wv, h, v, d, d);
    step(L, u, v, y, d, lam);
    for (int i = 0; i < d; i++) x[i] += y[i];
}

// vocab = 0 times the mixer only; vocab > 0 times the whole model for one token: every block plus the
// final RMSNorm and the output head (the embedding lookup is a copy and is left out).
static double run(int d, int n_layers, Cfg c, int vocab) {
    Layer *Ls = malloc(n_layers * sizeof(Layer));
    Shell *Bs = malloc(n_layers * sizeof(Shell));
    for (int l = 0; l < n_layers; l++) {
        if (c.alternate) Ls[l] = l % 2 == 0 ? make(d, 0, c.m_f, 0, 0, c.lazy, 0) : make(d, c.m_s, 0, 0, 0, 0, 0);
        else Ls[l] = make(d, c.m_s, c.m_f, c.shared, c.merged, c.lazy, c.fused);
        Bs[l] = (Shell){buf(d), buf((size_t)d * 4), buf((size_t)d * 3), Ls[l].m_f ? buf((size_t)d * d) : NULL};
        for (int i = 0; i < d; i++) Bs[l].nw[i] = 1.0f;
    }
    float *u = buf(d), *v = buf(d), *y = buf(d), *x = buf(d), *h = buf(d), *emb = buf(d), *fw = buf(d);
    float *head = vocab ? buf((size_t)vocab * d) : NULL, *logits = vocab ? buf(vocab) : NULL;
    float lam = 0.999f;
    int tokens = 0;
    struct timespec t0, t1;
    double el = 0;
    for (int pass = 0; pass < 2; pass++) {  // pass 0 warms up, pass 1 is timed
        tokens = 0; el = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        while (el < (pass ? 1.5 : 0.1)) {
            for (int k = 0; k < 20; k++, tokens++) {
                if (vocab) {
                    memcpy(x, emb, d * sizeof(float));
                    for (int l = 0; l < n_layers; l++) block(&Ls[l], &Bs[l], x, h, u, v, y, d, lam);
                    rmsnorm(h, x, fw, d);
                    matvec(head, h, logits, vocab, d);
                    emb[tokens % d] += logits[0] * 1e-6f;
                } else {
                    for (int l = 0; l < n_layers; l++) { step(&Ls[l], u, v, y, d, lam); u[l % d] += y[0] * 1e-6f; }
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        }
    }
    if (y[0] == 12345.f || (vocab && logits[0] == 12345.f)) printf("!");
    for (int l = 0; l < n_layers; l++) {  // the larger scales would otherwise hold gigabytes across runs
        Layer *L = &Ls[l];
        float *ps[] = {L->Ks, L->Kf, L->mu, L->S, L->w, L->c, L->rs, L->rf, L->rprev, L->q,
                       Bs[l].nw, Bs[l].conv, Bs[l].cbuf, Bs[l].Wv};
        for (int k = 0; k < 14; k++) free(ps[k]);
    }
    free(Ls); free(Bs); free(u); free(v); free(y); free(x); free(h); free(emb); free(fw); free(head); free(logits);
    return el / tokens * 1e6;  // microseconds per token, all layers
}

int main(void) {
    struct { const char *name; int d, n_layers, m, vocab; Cfg extra[6]; } scales[] = {
        {"260K", 64, 10, 140, 512, {{"P-free", 140, 140}, {"P(69,140)", 69, 140}, {"P(93,93)", 93, 93}, {"P(105,70)", 105, 70}, {"A(207,207)", 207, 207, 0, 0, 0, 1}}},
        {"15M", 288, 12, 712, 32000, {{"P-free", 712, 712}, {"P(355,712)", 355, 712}, {"P(474,474)", 474, 474}, {"P(534,356)", 534, 356}, {"A(1045,1045)", 1045, 1045, 0, 0, 0, 1}}},
        {"42M", 512, 16, 1280, 32000, {{"P-free", 1280, 1280}}},
        {"110M", 768, 24, 1912, 32000, {{"P-free", 1912, 1912}}},
    };
    for (int s = 0; s < 4; s++) {
        int d = scales[s].d, nl = scales[s].n_layers, m = scales[s].m, V = scales[s].vocab;
        double base = run(d, nl, (Cfg){"S-merged", m, m, 1, 1, 0, 0}, 0);
        printf("%-5s d=%d layers=%d m=%d vocab=%d, state %.1f MB\n", scales[s].name, d, nl, m, V, (double)d * m * nl * 4 / 1e6);
        printf("  %-14s %9.1f us/token  1.00x\n", "S-merged", base);
        Cfg base_cfgs[] = {{"S-split", m, m, 1, 0, 0}, {"S-lazy", m, m, 1, 0, 1}, {"S-fused", m, m, 1, 0, 0, 0, 1},
                           {"S-fused-lazy", m, m, 1, 0, 1, 0, 1}, {"P-free-fused", m, m, 0, 0, 0, 0, 1}};
        for (int k = 0; k < 5; k++) { double t = run(d, nl, base_cfgs[k], 0); printf("  %-14s %9.1f us/token  %.2fx\n", base_cfgs[k].name, t, t / base); }
        for (int k = 0; k < 6 && scales[s].extra[k].name; k++) {
            Cfg c = scales[s].extra[k];
            double t = run(d, nl, c, 0);
            printf("  %-14s %9.1f us/token  %.2fx", c.name, t, t / base);
            c.lazy = 1; t = run(d, nl, c, 0);
            printf("   lazy %9.1f us/token  %.2fx\n", t, t / base);
        }
        // whole model, eager fast state: norm, conv, value projection, residual, final norm and head included
        double wbase = run(d, nl, (Cfg){"S-merged", m, m, 1, 1, 0, 0}, V);
        printf("  whole model, one token\n");
        printf("  %-14s %9.1f us/token  1.00x  whole\n", "S-merged", wbase);
        double t = run(d, nl, (Cfg){"S-fused", m, m, 1, 0, 0, 0, 1}, V);
        printf("  %-14s %9.1f us/token  %.2fx  whole\n", "S-fused", t, t / wbase);
        t = run(d, nl, (Cfg){"P-free-fused", m, m, 0, 0, 0, 0, 1}, V);
        printf("  %-14s %9.1f us/token  %.2fx  whole\n", "P-free-fused", t, t / wbase);
        for (int k = 0; k < 6 && scales[s].extra[k].name; k++) {
            t = run(d, nl, scales[s].extra[k], V);
            printf("  %-14s %9.1f us/token  %.2fx  whole\n", scales[s].extra[k].name, t, t / wbase);
        }
        fflush(stdout);
    }
    return 0;
}
