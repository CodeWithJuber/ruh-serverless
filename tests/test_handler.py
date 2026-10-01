"""Worker contract tests: no model download, external job, or GPU needed."""
from types import SimpleNamespace

import pytest
import torch

import handler


def test_prompt_context_zero_temperature_and_continuation(monkeypatch):
    observed = {}
    class Tokenizer:
        _vocab = SimpleNamespace(n_roots=62)
        def encode(self, prompt, *, add_eos):
            observed.update(prompt=prompt, add_eos=add_eos)
            return [(1, 0), (4, 1), (5, 1)]
        def decode(self, tokens):
            observed["decoded"] = tokens
            return "generated"
    class Model:
        config = SimpleNamespace(max_seq_len=2)
        def generate(self, roots, patterns, **kwargs):
            observed.update(kwargs)
            observed["roots"] = roots.tolist()
            assert roots.device.type == patterns.device.type == "cpu"
            return torch.cat([roots, torch.tensor([[10, 2]])], dim=1)
    monkeypatch.setattr(handler, "DEVICE", "cpu")
    monkeypatch.setattr(handler, "_model", Model())
    monkeypatch.setattr(handler, "_tokenizer", Tokenizer())
    messages = [{"role": "system", "content": "system rule"}, {"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}, {"role": "user", "content": "third"}]
    result = handler.handler({"input": {"messages": messages, "temperature": 0, "max_tokens": 8}})
    assert result["choices"][0]["message"]["content"] == "generated"
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert observed["add_eos"] is False and observed["temperature"] == 0
    assert observed["roots"] == [[4, 5]] and observed["decoded"] == [(10, 1), (2, 1)]
    assert all(text in observed["prompt"] for text in ("system rule", "first", "second", "third"))


@pytest.mark.parametrize("input", [{"messages": []}, {"messages": ["bad"]}, {"messages": [{}], "max_tokens": 0}, {"messages": [{}], "temperature": float("nan")}])
def test_malformed_job_rejected_before_model_loading(input, monkeypatch):
    monkeypatch.setattr(handler, "load_model", lambda: pytest.fail("Invalid input must not load weights"))
    assert "error" in handler.handler({"input": input})
