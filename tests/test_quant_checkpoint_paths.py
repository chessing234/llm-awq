import importlib
import json
import sys
import types

import pytest
import torch


@pytest.fixture
def load_quant(monkeypatch):
    # Only CUDA kernel imports and quantization initialization are substituted.
    # Checkpoint discovery, deserialization, and tensor loading use Accelerate.
    module_names = (
        "tinychat.utils.load_quant",
        "awq.quantize.quantizer",
        "awq.quantize.qmodule",
    )
    previously_loaded = set(sys.modules)
    if "awq_inference_engine" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "awq_inference_engine",
            types.ModuleType("awq_inference_engine"),
        )
    module = importlib.import_module("tinychat.utils.load_quant")
    monkeypatch.setattr(module, "real_quantize_model_weight", lambda *a, **kw: None)
    monkeypatch.setattr(module.tinychat.utils.constants, "mem_efficient_load", False)
    yield module
    for name in module_names:
        if name not in previously_loaded:
            sys.modules.pop(name, None)


def make_model(bias=True):
    model = torch.nn.Linear(2, 2, bias=bias)
    model.config = types.SimpleNamespace()
    return model


def checkpoint_state():
    return {"weight": torch.arange(4).reshape(2, 2).float(), "bias": torch.ones(2)}


def save_checkpoint(path):
    state = checkpoint_state()
    torch.save(state, path)
    return state


def assert_loaded(model, state):
    for name, tensor in state.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


@pytest.mark.parametrize("stale_index", [False, True])
def test_directory_loads_unique_awq_checkpoint(load_quant, tmp_path, stale_index):
    checkpoint = tmp_path / "vila-1.5-w4-g128-awq-v2.pt"
    (tmp_path / "not-a-checkpoint.pt").mkdir()
    state = save_checkpoint(checkpoint)
    if stale_index:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model-00001-of-00006.safetensors"}}),
            encoding="utf-8",
        )
    model = load_quant.load_awq_model(make_model(), str(tmp_path), 4, 128, "cpu")
    assert_loaded(model, state)


def test_explicit_checkpoint_is_not_replaced(load_quant, tmp_path):
    checkpoint = tmp_path / "selected-v2.pt"
    state = save_checkpoint(checkpoint)
    torch.save(
        {name: tensor + 10 for name, tensor in state.items()}, tmp_path / "other-v2.pt"
    )
    model = load_quant.load_awq_model(make_model(), str(checkpoint), 4, 128, "cpu")
    assert_loaded(model, state)


def test_sharded_directory_still_uses_index(load_quant, tmp_path):
    state = checkpoint_state()
    torch.save({"weight": state["weight"]}, tmp_path / "first.pt")
    torch.save({"bias": state["bias"]}, tmp_path / "second.pt")
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"weight": "first.pt", "bias": "second.pt"}}),
        encoding="utf-8",
    )
    model = load_quant.load_awq_model(make_model(), str(tmp_path), 4, 128, "cpu")
    assert_loaded(model, state)


@pytest.mark.parametrize("names", [[], ["first.pt", "second.pt"], ["folder.pt"]])
def test_directory_without_unique_checkpoint_still_rejects(load_quant, tmp_path, names):
    for name in names:
        if name == "folder.pt":
            (tmp_path / name).mkdir()
        else:
            save_checkpoint(tmp_path / name)
    with pytest.raises(ValueError):
        load_quant.load_awq_model(make_model(), str(tmp_path), 4, 128, "cpu")


def test_missing_explicit_checkpoint_still_rejects(load_quant, tmp_path):
    with pytest.raises(ValueError):
        load_quant.load_awq_model(
            make_model(), str(tmp_path / "missing-v2.pt"), 4, 128, "cpu"
        )


def test_mem_efficient_directory_is_not_resolved(load_quant, tmp_path, monkeypatch):
    model = make_model(bias=False)
    state = {"weight": checkpoint_state()["weight"]}
    torch.save(state, tmp_path / "weight.pt")
    monkeypatch.setattr(load_quant.tinychat.utils.constants, "mem_efficient_load", True)
    assert load_quant.load_awq_model(model, str(tmp_path), 4, 128, "cpu") is model
    assert_loaded(model, state)
