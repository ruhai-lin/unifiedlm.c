"""scripts/check_kernel.py restricted to the S configurations, in fp32 and fp16 (eager and compiled), for cards
without bf16 such as the T4. Same tolerances and rules; rule 3 compares S with itself, so it always passes.

$ python scripts/colab/check_kernel_fp16.py
"""
import os
import re
import sys

src = open(os.path.join(os.getcwd(), "scripts", "check_kernel.py")).read()
src = re.sub(r"\nMODES = .*\n", "\nMODES = [(torch.float32, False), (torch.float16, False), (torch.float16, True)]\n", src)
src = src.replace("\ndef set_memory", "\nCONFIGS = [c for c in CONFIGS if c[0] == 'S']\n\n\ndef set_memory", 1)
assert "torch.float16, True" in src and "c[0] == 'S'" in src
sys.argv[0] = os.path.join(os.getcwd(), "scripts", "check_kernel.py")
sys.path.insert(0, os.path.dirname(sys.argv[0]))  # as when check_kernel.py is run directly
exec(compile(src, sys.argv[0], "exec"), {"__name__": "__main__", "__file__": sys.argv[0]})
