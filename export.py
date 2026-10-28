"""
Export a trained UnifiedLM checkpoint to a .bin file to be read from and inferenced in C (run.c).

Format (version 1, fp32):
- 256-byte header: magic "ULMe", version, then dim, n_layers, hidden_dim, vocab_size,
  max_seq_len, conv_size, shared_classifier as int32
- token embedding table
- per block: norm, conv, key weight, key bias, mu, val, w, lam, eps
- final norm, and the classifier if it is not shared with the embedding

The learned scalars are exported in the form run.c uses them:
w = exp(clamp(log_w, -7, 12)), lam = exp(-exp(log_rate)), eps = exp(clamp(log_eps, -10, 10)).
"""
import argparse
import os
import struct

import torch

from model import UnifiedLM, ModelArgs


ULME_MAGIC = 0x554C4D65  # "ULMe"
HEADER_SIZE = 256

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy()
    file.write(d.tobytes())

# -----------------------------------------------------------------------------
# version 1: fp32 export (consumed by run.c)

def version1_export(model, filepath):
    """Export UnifiedLM weights to a C-readable FP32 binary file for run.c."""
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ULMe" in ASCII
    out_file.write(struct.pack("<I", ULME_MAGIC))
    # 2) write the format version
    out_file.write(struct.pack("<i", 1))
    # 3) write the params
    out_file.write(
        struct.pack(
            "<7i",
            p.dim,
            p.n_layers,
            p.hidden_dim,
            p.vocab_size,
            p.max_seq_len,
            p.conv_size,
            int(shared_classifier),
        )
    )
    pad = HEADER_SIZE - out_file.tell()
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # now write the model: embedding, then each block, final norm
    serialize_fp32(out_file, model.tok_embeddings.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.norm.weight)
        serialize_fp32(out_file, layer.conv.weight[:, 0, :])     # (dim, conv_size), oldest -> newest
        serialize_fp32(out_file, layer.key.weight)               # (hidden_dim, dim)
        serialize_fp32(out_file, layer.key.bias)
        serialize_fp32(out_file, layer.mu.weight)                # (dim, hidden_dim)
        serialize_fp32(out_file, layer.val.weight)               # (dim, dim)
        serialize_fp32(out_file, layer._w())
        serialize_fp32(out_file, torch.exp(-layer.log_rate.exp()))
        serialize_fp32(out_file, layer.log_eps.clamp(-10, 10).exp())
    serialize_fp32(out_file, model.norm.weight)
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, filepath, version=1):
    """
    v1: float32 export for run.c
    """
    if version == 1:
        version1_export(model, filepath)
    else:
        raise ValueError(f"unknown version {version}")

# -----------------------------------------------------------------------------
# checkpoint loading

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise ValueError(f"{path} does not contain model_args")
    model = UnifiedLM(ModelArgs(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("outputs", "15M", "best.pt"),
                        help="model checkpoint, .pt file")
    parser.add_argument("--output", default=os.path.join("outputs", "15M", "model.bin"),
                        help="the output filepath")
    parser.add_argument("--version", type=int, default=1, choices=(1,),
                        help="v1 fp32 for run.c")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model_export(model, args.output, args.version)
