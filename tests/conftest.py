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
from safemmsa.repository import reset_repository


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path_factory, monkeypatch):
    """把持久化（旧 JSON store + 新仓储）重定向到临时目录，避免污染仓库 storage/。"""
    d = tmp_path_factory.mktemp("storage")
    monkeypatch.setenv("SAFEMMSA_STORAGE_DIR", str(d))
    monkeypatch.delenv("SAFEMMSA_DB_BACKEND", raising=False)  # 测试默认 JSON 后端
    monkeypatch.delenv("SAFEMMSA_DB_PATH", raising=False)
    analyze_history._path = str(d / "analyze_history.json")
    eval_runs._path = str(d / "eval_runs.json")
    analyze_history.clear()
    eval_runs.clear()
    reset_repository()  # 单例按新目录重建（含 ensure_anonymous_user）
    yield
    reset_repository()


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
