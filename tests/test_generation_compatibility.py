"""Worker generation matches training while preserving legacy inference."""

import torch

from ruh_model.config import RuhConfig
from ruh_model.model import RuhModel
from ruh_model.tokenizer.bayan import BayanTokenizer


def test_v2_generation_uses_the_training_pattern_and_surface_vocabulary():
    tokenizer = BayanTokenizer(version=2)
    config = RuhConfig(
        d_model=32,
        d_root=8,
        d_pattern=8,
        n_heads=4,
        n_layers=1,
        n_roots=tokenizer._vocab.n_roots,
        max_seq_len=64,
        tokenizer_version=2,
        moe_interval=0,
        dropout=0,
    )
    model = RuhModel(config)
    model.tokenizer = tokenizer
    seen = []
    original = model._sample_next_token

    def sample(roots, patterns, temperature, valid_n_roots):
        seen.append(patterns.clone())
        return original(roots, patterns, temperature, valid_n_roots)

    model._sample_next_token = sample
    generated = model.generate(
        torch.tensor([[1, tokenizer._byte_start + 97]]),
        torch.tensor([[0, 0]]),
        max_new_tokens=3,
        temperature=0,
    )
    assert all(
        token == 2 or tokenizer._byte_start <= token < tokenizer._byte_start + 256
        for token in generated[0, 2:].tolist()
    )
    for patterns in seen:
        assert not patterns.any()


def test_v2_masks_dominant_legacy_unknown_and_root_logits():
    tokenizer = BayanTokenizer(version=2)
    config = RuhConfig(
        d_model=32,
        d_root=8,
        d_pattern=8,
        n_heads=4,
        n_layers=1,
        n_roots=tokenizer._vocab.n_roots,
        max_seq_len=64,
        tokenizer_version=2,
        moe_interval=0,
    )
    model = RuhModel(config)
    model.tokenizer = tokenizer

    def forward(roots, patterns):
        logits = torch.zeros((len(roots), roots.shape[1], config.n_roots))
        logits[..., 3] = 1e6  # Legacy UNK would win without the new mask.
        logits[..., 4] = 1e5  # A legacy root cannot reconstruct V2 surface text.
        logits[..., tokenizer._byte_start + ord("a")] = 1
        return {"logits": logits}

    model.forward = forward
    generated = model.generate(
        torch.tensor([[1]]), torch.tensor([[0]]), max_new_tokens=3, temperature=0
    )
    assert tokenizer.decode([(token, 0) for token in generated[0].tolist()]) == "aaa"


def test_v1_generation_keeps_legacy_pattern_one():
    config = RuhConfig(
        d_model=32,
        d_root=8,
        d_pattern=8,
        n_heads=4,
        n_layers=1,
        n_roots=100,
        max_seq_len=64,
        tokenizer_version=1,
        moe_interval=0,
    )
    model = RuhModel(config)
    seen = []

    def next_token(roots, patterns, temperature, valid_n_roots):
        seen.append(patterns.clone())
        return torch.full((len(roots), 1), 4)

    model._sample_next_token = next_token
    model.generate(
        torch.tensor([[1, 4]]), torch.tensor([[0, 1]]), max_new_tokens=3, temperature=0
    )
    assert seen[1][0, -1].item() == 1
    assert seen[2][0, -2:].tolist() == [1, 1]
