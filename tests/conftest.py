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
from safemmsa.persistence import analyze_history, eval_runs


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path_factory):
    """把持久化重定向到临时目录，避免测试在仓库里写 storage/。"""
    d = tmp_path_factory.mktemp("storage")
    analyze_history._path = str(d / "analyze_history.json")
    eval_runs._path = str(d / "eval_runs.json")
    analyze_history.clear()
    eval_runs.clear()
    yield


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
