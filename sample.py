"""
Sample from a trained UnifiedLM checkpoint.
$ python sample.py --checkpoint=outputs/15M/best.pt --start="Once upon a time"
"""
from contextlib import nullcontext

import torch

from model import UnifiedLM, ModelArgs
from tokenizer import Tokenizer

# -----------------------------------------------------------------------------
checkpoint = "outputs/15M/best.pt"
start = "Once upon a time"  # or "" for an unconditional sample
num_samples = 1
max_new_tokens = 256
temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random; 0.0 = greedy
top_k = 300  # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = ""  # override the tokenizer model path
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "float32"
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
model = UnifiedLM(ModelArgs(**checkpoint_dict["model_args"]))
state_dict = checkpoint_dict["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# load the tokenizer: the Llama 2 one unless the checkpoint was trained on a custom vocab
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = model.params.vocab_size
if not tokenizer and vocab_source == "custom":
    tokenizer = f"tok{vocab_size}.model"
enc = Tokenizer(tokenizer_model=tokenizer or None)

# encode the beginning of the prompt
start_ids = enc.encode(start, bos=True, eos=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print("---------------")
