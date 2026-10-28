/* Inference for the UnifiedLM language model in pure C. */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int dim;               // residual width
    int n_layers;          // number of UnifiedBlocks
    int hidden_dim;        // memory slots per block
    int vocab_size;
    int seq_len;
    int conv_size;         // causal depthwise conv width on the address path
    int shared_classifier;
} Config;

typedef struct {
    float *token_embedding_table;
    float *layers;
    float *rms_final_weight;
    float *wcls;
    size_t layer_stride;
    float *data;
} Weights;

typedef struct {
    float *x, *h, *u, *v, *y;
    float *r;              // read address at t (hidden_dim)
    float *q;              // r / c (hidden_dim)
    float *logits;
    // per-block recurrent state, constant in context length
    float *conv_state;     // (n_layers, dim, conv_size - 1), oldest -> newest
    float *U;              // (n_layers, dim, hidden_dim) slot contents
    float *c;              // (n_layers, hidden_dim) slot evidence mass
    float *r_prev;         // (n_layers, hidden_dim) write address = read address at t-1
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
} UnifiedLM;

static void *xcalloc(size_t n, size_t size) {
    void *ptr = calloc(n, size);
    if (!ptr) {
        fprintf(stderr, "calloc failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void rmsnorm(float *o, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) o[i] = weight[i] * (ss * x[i]);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul(float *out, const float *x, const float *w, int n, int d) {
    // W (d,n) @ x (n,) -> out (d,)
    #pragma omp parallel for if (d > 256)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        out[i] = val;
    }
}

static size_t layer_stride(const Config *p) {
    return (size_t)p->dim                          // norm
        + (size_t)p->dim * p->conv_size            // conv
        + (size_t)p->hidden_dim * p->dim           // key weight
        + (size_t)p->hidden_dim                    // key bias
        + (size_t)p->dim * p->hidden_dim           // mu
        + (size_t)p->dim * p->dim                  // val
        + (size_t)p->hidden_dim                    // w
        + 1                                        // lam
        + 1;                                       // eps
}

static void malloc_run_state(RunState *s, const Config *p) {
    int kw = p->conv_size - 1;
    s->x = xcalloc(p->dim, sizeof(float));
    s->h = xcalloc(p->dim, sizeof(float));
    s->u = xcalloc(p->dim, sizeof(float));
    s->v = xcalloc(p->dim, sizeof(float));
    s->y = xcalloc(p->dim, sizeof(float));
    s->r = xcalloc(p->hidden_dim, sizeof(float));
    s->q = xcalloc(p->hidden_dim, sizeof(float));
    s->logits = xcalloc(p->vocab_size, sizeof(float));
    s->conv_state = xcalloc((size_t)p->n_layers * p->dim * (kw > 0 ? kw : 1), sizeof(float));
    s->U = xcalloc((size_t)p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    s->c = xcalloc((size_t)p->n_layers * p->hidden_dim, sizeof(float));
    s->r_prev = xcalloc((size_t)p->n_layers * p->hidden_dim, sizeof(float));
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->h); free(s->u); free(s->v); free(s->y);
    free(s->r); free(s->q); free(s->logits);
    free(s->conv_state); free(s->U); free(s->c); free(s->r_prev);
}

static void reset_state(UnifiedLM *model) {
    // the fast store starts at the prior: U = mu * w, c = w, no previous address, empty conv buffer
    Config *p = &model->config;
    Weights *w = &model->weights;
    RunState *s = &model->state;
    int dim = p->dim, m = p->hidden_dim, kw = p->conv_size - 1;
    for (int l = 0; l < p->n_layers; l++) {
        float *base = w->layers + (size_t)l * w->layer_stride;
        float *mu = base + dim + (size_t)dim * p->conv_size + (size_t)m * dim + m;
        float *wa = mu + (size_t)dim * m + (size_t)dim * dim;
        float *U = s->U + (size_t)l * dim * m;
        for (int i = 0; i < dim; i++)
            for (int a = 0; a < m; a++) U[i * m + a] = mu[i * m + a] * wa[a];
        memcpy(s->c + (size_t)l * m, wa, m * sizeof(float));
    }
    memset(s->r_prev, 0, (size_t)p->n_layers * m * sizeof(float));
    if (kw > 0) memset(s->conv_state, 0, (size_t)p->n_layers * dim * kw * sizeof(float));
}

static void read_checkpoint(const char *path, UnifiedLM *model) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "could not open checkpoint %s\n", path);
        exit(EXIT_FAILURE);
    }
    uint32_t magic;
    int version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || fread(&version, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "failed to read checkpoint header\n");
        exit(EXIT_FAILURE);
    }
    if (magic != 0x554c4d65) {
        fprintf(stderr, "bad checkpoint magic for UnifiedLM: 0x%08x\n", magic);
        exit(EXIT_FAILURE);
    }
    if (version != 1) {
        fprintf(stderr, "run requires an fp32 checkpoint (version 1), got %d\n", version);
        exit(EXIT_FAILURE);
    }
    if (fread(&model->config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "failed to read checkpoint config\n");
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 256, SEEK_SET);
    size_t n_floats = (size_t)(file_size - 256) / sizeof(float);
    model->weights.data = xcalloc(n_floats, sizeof(float));
    if (fread(model->weights.data, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "failed to read checkpoint weights\n");
        exit(EXIT_FAILURE);
    }
    fclose(f);

    Config *p = &model->config;
    Weights *w = &model->weights;
    w->layer_stride = layer_stride(p);
    float *ptr = w->data;
    w->token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * p->dim;
    w->layers = ptr;
    ptr += (size_t)p->n_layers * w->layer_stride;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->wcls = p->shared_classifier ? w->token_embedding_table : ptr;
}

static void free_model(UnifiedLM *model) {
    free(model->weights.data);
    free_run_state(&model->state);
}

static float *forward(UnifiedLM *model, int token) {
    Config *p = &model->config;
    Weights *w = &model->weights;
    RunState *s = &model->state;
    int dim = p->dim, m = p->hidden_dim, K = p->conv_size, kw = K - 1;
    const float eps = 1e-5f;

    memcpy(s->x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        float *base = w->layers + (size_t)l * w->layer_stride;
        float *norm = base; base += dim;
        float *conv = base; base += (size_t)dim * K;
        float *key_w = base; base += (size_t)m * dim;
        float *key_b = base; base += m;
        float *mu = base; base += (size_t)dim * m;
        float *val = base; base += (size_t)dim * dim;
        float *wa = base; base += m;
        float lam = base[0];
        float null_mass = base[1];
        float *buf = s->conv_state + (size_t)l * dim * kw;
        float *U = s->U + (size_t)l * dim * m;
        float *c = s->c + (size_t)l * m;
        float *r_prev = s->r_prev + (size_t)l * m;

        // address: causal depthwise conv over RMSNorm(x), then normalized relu^2 code
        rmsnorm(s->h, s->x, norm, dim, eps);
        for (int i = 0; i < dim; i++) {
            float *st = buf + (size_t)i * kw;
            float val_i = conv[i * K + kw] * s->h[i];
            for (int j = 0; j < kw; j++) val_i += conv[i * K + j] * st[j];
            for (int j = 0; j + 1 < kw; j++) st[j] = st[j + 1];
            if (kw > 0) st[kw - 1] = s->h[i];
            s->u[i] = val_i;
        }
        matmul(s->r, s->u, key_w, dim, m);
        float zsum = 0.0f;
        for (int a = 0; a < m; a++) {
            float z = s->r[a] + key_b[a];
            z = z > 0.0f ? z * z : 0.0f;
            s->r[a] = z;
            zsum += z;
        }
        for (int a = 0; a < m; a++) s->r[a] /= (null_mass + zsum);

        // write: value v_t under the previous address, decaying toward the prior
        matmul(s->v, s->h, val, dim, dim);
        for (int a = 0; a < m; a++) {
            c[a] = lam * c[a] + (1.0f - lam) * wa[a] + r_prev[a];
            s->q[a] = s->r[a] / c[a];
        }
        #pragma omp parallel for if (dim > 256)
        for (int i = 0; i < dim; i++) {
            float *Ui = U + (size_t)i * m;
            const float *mui = mu + (size_t)i * m;
            float vi = s->v[i];
            float yi = 0.0f;
            for (int a = 0; a < m; a++) {
                Ui[a] = lam * Ui[a] + (1.0f - lam) * mui[a] * wa[a] + vi * r_prev[a];
                yi += Ui[a] * s->q[a];
            }
            s->y[i] = yi;
        }
        memcpy(r_prev, s->r, m * sizeof(float));

        // read: residual
        for (int i = 0; i < dim; i++) s->x[i] += s->y[i];
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim, eps);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = xcalloc(vocab_size, sizeof(char *));
    t->vocab_scores = xcalloc(vocab_size, sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "could not load tokenizer %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        t->vocab[i] = xcalloc((size_t)len + 1, sizeof(char));
        if (len > 0 && fread(t->vocab[i], len, 1, file) != 1) exit(EXIT_FAILURE);
    }
    fclose(file);
}

static void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printf(char *piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

static void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = xcalloc(t->vocab_size, sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = xcalloc((size_t)t->max_token_length * 2 + 1 + 2, sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_prob(const void *a, const void *b) {
    const ProbIndex *pa = (const ProbIndex *)a;
    const ProbIndex *pb = (const ProbIndex *)b;
    return (pa->prob < pb->prob) - (pa->prob > pb->prob);
}

static int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
    for (int q = 0; q < sampler->vocab_size; q++) logits[q] /= sampler->temperature;
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) return sample_mult(logits, sampler->vocab_size, coin);
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

static long time_in_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static void generate(UnifiedLM *model, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    int num_prompt_tokens = 0;
    int *prompt_tokens = xcalloc(strlen(prompt) + 3, sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "empty prompt after encoding\n");
        exit(EXIT_FAILURE);
    }
    int token = prompt_tokens[0];
    int next = 0;
    long start = 0;
    int pos = 0;
    for (; pos < steps; pos++) {
        float *logits = forward(model, token);
        next = (pos < num_prompt_tokens - 1) ? prompt_tokens[pos + 1] : sample(sampler, logits);
        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) break;
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (start != 0 && pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: run <checkpoint> [-z tokenizer.bin] [-t temperature] [-p topp] [-s seed] [-n steps] [-i prompt]\n");
        return 1;
    }
    const char *checkpoint = argv[1];
    const char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    const char *prompt = "";
    unsigned long long rng_seed = (unsigned long long)time(NULL);
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') {
            fprintf(stderr, "Usage: run <checkpoint> [-z tokenizer.bin] [-t temperature] [-p topp] [-s seed] [-n steps] [-i prompt]\n");
            return 1;
        }
        if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
        else if (argv[i][1] == 't') temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's') rng_seed = strtoull(argv[i + 1], NULL, 10);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
    }

    UnifiedLM model = {0};
    read_checkpoint(checkpoint, &model);
    malloc_run_state(&model.state, &model.config);
    reset_state(&model);
    if (steps <= 0 || steps > model.config.seq_len) steps = model.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model.config.vocab_size);

    Sampler sampler = {
        .vocab_size = model.config.vocab_size,
        .probindex = xcalloc(model.config.vocab_size, sizeof(ProbIndex)),
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
    };

    generate(&model, &tokenizer, &sampler, prompt, steps);

    free(sampler.probindex);
    free_tokenizer(&tokenizer);
    free_model(&model);
    return 0;
}
