"""VARIANCE-LIES: variance captured is not behavior preserved.

A rank-k replay can capture ~99% of a model's activation variance and still
behave completely differently. This probe runs a rank sweep and prints two
curves side by side:

- variance_captured: how much of the delta energy the basis keeps
- behavior: how many generated tokens still match the original model

If the curves disagree, the compression metric is lying.

GPU-first, CPU-supported. No training. Bounded evaluation.
"""

__version__ = "0.1.0"

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("unsupported architecture")


def load_tok(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")


@torch.no_grad()
def collect_deltas(model, tok, prompts, max_len=64):
    layers = get_layers(model)
    store = {i: [] for i in range(len(layers))}
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out[0] if isinstance(out, tuple) else out
            d = y - x
            store[idx].append(d.reshape(-1, d.shape[-1]).float().cpu())
        return hook

    for i, lyr in enumerate(layers):
        handles.append(lyr.register_forward_hook(make_hook(i)))
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids[:, :max_len]
        model(ids)
    for h in handles:
        h.remove()
    return store


def spectrum(store, rank):
    """Per-layer: basis + fraction of delta variance captured at this rank."""
    bases = {}
    fracs = []
    for i, chunks in store.items():
        D = torch.cat(chunks, dim=0)
        mu = D.mean(dim=0)
        A = D - mu
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        total = float((S ** 2).sum()) + 1e-12
        fracs.append(float((S[:rank] ** 2).sum()) / total)
        bases[i] = {"mu": mu, "V": V[:rank].T.contiguous()}
    return bases, sum(fracs) / len(fracs)


class Replay:
    def __init__(self, model, bases):
        self.on = False
        self.handles = []

        def make_hook(b):
            def hook(mod, inp, out):
                if not self.on:
                    return out
                x = inp[0] if isinstance(inp, tuple) else inp
                y = out[0] if isinstance(out, tuple) else out
                d = y - x
                flat = d.reshape(-1, d.shape[-1]).float()
                rec = (flat - b["mu"]) @ b["V"] @ b["V"].T + b["mu"]
                new = x + rec.reshape(x.shape).to(x.dtype)
                if isinstance(out, tuple):
                    return (new,) + out[1:]
                return new
            return hook

        for i, lyr in enumerate(get_layers(model)):
            self.handles.append(lyr.register_forward_hook(make_hook(bases[i])))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def greedy(model, tok, prompt, steps):
    ids = tok(prompt, return_tensors="pt").input_ids
    gen = []
    for _ in range(steps):
        nxt = int(model(ids).logits[:, -1, :].argmax(dim=-1).item())
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return gen


def probe(model, tok, store, prompts, rank, steps):
    bases, var_frac = spectrum(store, rank)
    replay = Replay(model, bases)
    agree = 0
    total = 0
    for p in prompts:
        replay.on = False
        ref = greedy(model, tok, p, steps)
        replay.on = True
        rep = greedy(model, tok, p, steps)
        replay.on = False
        agree += sum(1 for a, b in zip(ref, rep) if a == b)
        total += steps
    replay.remove()
    return {"rank": rank, "variance_captured": round(var_frac, 4), "behavior": round(agree / max(total, 1), 4)}


def main():
    ap = argparse.ArgumentParser(prog="variance-lies", description="Variance captured vs behavior preserved across replay ranks.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--output", default="variance_vs_behavior.json")
    a = ap.parse_args()

    prompts = a.prompt or ["Tokyo is the capital of", "The sun rises in the", "Water boils at"]
    ranks = [int(x) for x in a.ranks.split(",")]

    tok = load_tok(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    store = collect_deltas(model, tok, prompts)

    curve = [probe(model, tok, store, prompts, r, a.steps) for r in ranks]
    print("rank | variance_captured | behavior")
    for row in curve:
        print(str(row["rank"]).rjust(4), "|", str(row["variance_captured"]).ljust(17), "|", row["behavior"])

    with open(a.output, "w") as f:
        json.dump({"model": a.model, "steps": a.steps, "curve": curve}, f, indent=2)
    print("saved: " + a.output)


if __name__ == "__main__":
    main()
