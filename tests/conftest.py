"""测试夹具：保证每个测试从干净的注册表状态开始，并提供一个端到端 stub 结果。"""
from __future__ import annotations

import pytest

import safemmsa
from safemmsa import (
    register_encoder,
    register_enhancement,
    register_fusion_backbone,
    register_reliability_head,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    """每个测试前后把所有 register_* 复位为默认桩/基线，避免测试间串扰。"""
    for m in ("video", "audio", "text"):
        register_encoder(m, None)
    register_fusion_backbone(None)
    register_enhancement(None)
    register_reliability_head(None)
    yield
    for m in ("video", "audio", "text"):
        register_encoder(m, None)
    register_fusion_backbone(None)
    register_enhancement(None)
    register_reliability_head(None)


@pytest.fixture()
def stub_result():
    return safemmsa.analyze(text="I am so happy and great, wonderful day")
