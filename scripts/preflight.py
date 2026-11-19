"""
End-to-end smoke test of the run path on real data, before spending the training budget. Needs CUDA and
data/tok512. Run from the repository root:

$ python scripts/preflight.py

Checks that --seed changes the initial weights and reproduces them, that S and D share an init at the same
seed, that every variant trains and evaluates through the wrapper with the expected parameter count, that
eval_nll really builds D rather than S, and that the unsafe paths are refused.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import torch

FLAGS = ("--vocab_source=custom --vocab_size=512 --dim=64 --n_layers=10 --hidden_dim=140 --max_seq_len=512 "
         "--batch_size=4 --gradient_accumulation_steps=1 --compile=False --eval_iters=1 --max_iters=1").split()


def sh(*cmd, ok=True):
    done = subprocess.run([sys.executable, "-u", *cmd], capture_output=True, text=True)
    if ok and done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return done


def train(out, *flags):
    sh("scripts/train_variant.py", f"--out_dir={out}", *FLAGS, *flags)
    # get_lr(0) is 0, so after this single update the checkpoint still holds the initial weights
    state = torch.load(os.path.join(out, "ckpt.pt"), map_location="cpu")["model"]
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].numpy().tobytes())
    return digest.hexdigest()[:16]


def evaluate(out):
    sh("scripts/eval_nll.py", f"--out_dir={out}", "--eval_iters=1")
    with open(os.path.join(out, "eval.json")) as f:
        return json.load(f)


with tempfile.TemporaryDirectory() as tmp:
    a = train(f"{tmp}/s1", "--seed=1")
    assert a == train(f"{tmp}/s1b", "--seed=1"), "the same seed must reproduce the initial weights"
    assert a != train(f"{tmp}/s2", "--seed=2"), "different seeds must change the initial weights"
    print("seed: same seed reproduces the init, different seed changes it")

    assert a == train(f"{tmp}/d1", "--seed=1", "--variant=D"), "S and D must share an init at the same seed"
    s, d = evaluate(f"{tmp}/s1"), evaluate(f"{tmp}/d1")
    assert s["params"] == d["params"] == 259012
    assert s["nll"] != d["nll"], "identical weights must score differently, otherwise eval_nll built S for both"
    print(f"S and D: same init, NLL {s['nll']:.5f} vs {d['nll']:.5f} on identical weights")

    for name, flags, params in [("p_free", ("--variant=P", "--m_s=140", "--m_f=140"), 350022),
                                ("p_a", ("--variant=P", "--m_s=69", "--m_f=140"), 258432),
                                ("p_b", ("--variant=P", "--m_s=93", "--m_f=93"), 258372),
                                ("p_c", ("--variant=P", "--m_s=105", "--m_f=70"), 258672),
                                ("a", ("--variant=A", "--m_s=207", "--m_f=207"), 258352)]:
        train(f"{tmp}/{name}", "--seed=1", *flags)
        e = evaluate(f"{tmp}/{name}")
        assert e["params"] == params, (name, e["params"])
        print(f"{name}: {params:,} params, NLL {e['nll']:.5f}")

    os.remove(f"{tmp}/d1/variant.json")
    assert sh("scripts/eval_nll.py", f"--out_dir={tmp}/d1", "--eval_iters=1", ok=False).returncode != 0, "a missing variant.json must fail"
    assert sh("scripts/train_variant.py", f"--out_dir={tmp}/r", "--init_from=resume", *FLAGS, ok=False).returncode != 0, "resume must be refused"
    print("refused: evaluation without variant.json, and resume")

print("preflight passed")
