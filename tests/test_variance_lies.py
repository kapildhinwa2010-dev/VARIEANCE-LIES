import sys

sys.path.insert(0, ".")

import pytest
from transformers import AutoModelForCausalLM

from variance_lies import collect_deltas, get_layers, load_tok, probe

MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def setup():
    tok = load_tok(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    prompts = ["Hello world", "The cat sat on the"]
    store = collect_deltas(model, tok, prompts, max_len=16)
    return model, tok, store, prompts


def test_arch_detected(setup):
    model, *_ = setup
    assert len(get_layers(model)) >= 1


def test_curve_sanity(setup):
    model, tok, store, prompts = setup
    lo = probe(model, tok, store, prompts, rank=2, steps=3)
    hi = probe(model, tok, store, prompts, rank=8, steps=3)
    for row in (lo, hi):
        for k in ["rank", "variance_captured", "behavior"]:
            assert k in row
        assert 0.0 <= row["variance_captured"] <= 1.0
        assert 0.0 <= row["behavior"] <= 1.0
    assert hi["variance_captured"] >= lo["variance_captured"]


def test_variance_not_equal_behavior(setup):
    """The whole point: these are two different numbers, not one."""
    model, tok, store, prompts = setup
    row = probe(model, tok, store, prompts, rank=4, steps=3)
    assert abs(row["variance_captured"] - row["behavior"]) > 1e-6
