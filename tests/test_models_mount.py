"""模块12 模型挂载约定测试：默认 no-op；enable 未实现的 backend 不崩、保持桩。"""
from __future__ import annotations

from api.models_status import models_status
from models import default_config, load_models, unload_models


def test_load_models_default_is_noop():
    res = load_models()
    assert res["loaded"] == []
    assert all("disabled" in reason for reason in res["skipped"].values())
    assert models_status()["all_stub"] is True


def test_enable_unimplemented_backend_stays_stub():
    res = load_models({"encoder_text": {"enabled": True}})
    assert "encoder_text" not in res["loaded"]
    assert "未实现" in res["skipped"]["encoder_text"]
    # 系统仍是桩，不崩。
    assert models_status()["points"]["encoders"]["text"] == "stub"
    unload_models()


def test_register_then_unload(stub_result):
    # 注册一个玩具文本编码器 → status 变 trained；unload 复位。
    from safemmsa import register_encoder
    from safemmsa.contracts import ModalityOutput

    def toy(modality, raw, quality):
        return ModalityOutput(modality="text", available=True, backend="trained_toy")

    register_encoder("text", toy)
    assert models_status()["points"]["encoders"]["text"] == "trained"
    assert models_status()["all_stub"] is False
    unload_models()
    assert models_status()["all_stub"] is True


def test_default_config_all_disabled():
    cfg = default_config()
    assert all(v["enabled"] is False for v in cfg.values())
