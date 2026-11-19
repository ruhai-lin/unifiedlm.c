"""
Train one ablation variant with the unmodified training loop. Run from the repository root:

$ python scripts/train_variant.py --variant=P --m_s=105 --m_f=70 --out_dir=runs/x [train.py flags]

Writes out_dir/variant.json, which scripts/eval_nll.py requires: D has the same state dict as S, so the
checkpoint alone cannot tell them apart.

Resuming (--init_from=resume, out_dir/ckpt.pt from an interrupted run) is FineWeb only. train.py restarts the data
stream on resume, so the training stream here skips the iter_num * gradient_accumulation_steps * batch_size windows
the run already used, and the resumed run sees the same batches as an uninterrupted one. Other datasets refuse to
resume. With CKPT_MIRROR=<dir>, every ckpt.pt that train.py saves is copied there with train.log, for another pod
to resume from.
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())

own = {"variant": "S", "m_s": 0, "m_f": 0, "dataset": "tinystories", "head_dim": 48}
argv = []
for arg in sys.argv[1:]:
    key, _, val = arg[2:].partition("=")
    if key in own:
        own[key] = type(own[key])(val)
    else:
        argv.append(arg)
out_dir = next((a.split("=", 1)[1] for a in argv if a.startswith("--out_dir=")), None)
if out_dir is None:
    raise SystemExit("--out_dir is required")
resume = "--init_from=resume" in argv
if resume and own["dataset"] not in ("fineweb", "fineweb4"):
    raise SystemExit("resuming is FineWeb only: rerun from scratch")
sys.argv = [sys.argv[0]] + argv

import export    # noqa: E402
import variants  # noqa: E402

variants.HEAD_DIM = own["head_dim"]  # T only; llama2.c uses 64 at dim 512 and 768
variants.install(own["variant"], own["m_s"], own["m_f"])
if own["dataset"] in ("fineweb", "fineweb4"):  # 2-of-14 or 4-of-14 subset; train.py imports Task from tinystories
    os.environ["FW_SUBSET"] = "4of14" if own["dataset"] == "fineweb4" else "2of14"
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
    import fineweb_subset  # noqa: E402
    sys.modules["tinystories"] = fineweb_subset
    seed = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--seed=")), 1337)
    fineweb_subset.DATA_SEED = seed - 1337  # as for BabyLM, a new seed changes the data order too
    if resume:  # the first training stream train.py opens skips what the run already trained on
        import torch  # noqa: E402
        ck = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu")
        skip = ck["iter_num"] * ck["config"]["gradient_accumulation_steps"] * ck["config"]["batch_size"]
        print(f"resuming at iter {ck['iter_num']}: skipping the first {skip:,} training windows")
        del ck
        stream = fineweb_subset.fineweb.PretokDataset.__iter__

        def skipped(self):
            g = stream(self)
            if self.split == "train" and skipped.left:
                n, skipped.left = skipped.left, 0
                for _ in range(n):
                    next(g)
            yield from g
        skipped.left = skip
        fineweb_subset.fineweb.PretokDataset.__iter__ = skipped
elif own["dataset"] == "babylm":  # train.py imports its Task from tinystories
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
    import babylm  # noqa: E402
    sys.modules["tinystories"] = babylm
    seed = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--seed=")), 1337)
    babylm.DATA_SEED += seed - 1337  # a new seed changes the data order as well as the initialization
elif own["dataset"] != "tinystories":
    raise SystemExit(f"unknown dataset {own['dataset']}")
export.model_export = lambda *a, **k: None       # the run.c format belongs to the standard model
mirror = os.environ.get("CKPT_MIRROR")
if mirror:
    import shutil  # noqa: E402
    import torch   # noqa: E402
    save = torch.save

    def save_and_mirror(obj, f, *a, **k):  # after train.py writes ckpt.pt, copy it and train.log, ckpt.pt last
        save(obj, f, *a, **k)
        if isinstance(f, str) and os.path.basename(f) == "ckpt.pt":
            os.makedirs(mirror, exist_ok=True)
            for name in ("train.log", "ckpt.pt"):
                if os.path.exists(os.path.join(out_dir, name)):
                    shutil.copyfile(os.path.join(out_dir, name), os.path.join(mirror, name + ".tmp"))
                    os.replace(os.path.join(mirror, name + ".tmp"), os.path.join(mirror, name))
    torch.save = save_and_mirror

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "variant.json"), "w") as f:
    json.dump({**own, "argv": argv}, f, indent=2)

runpy.run_path("train.py", run_name="__main__")
