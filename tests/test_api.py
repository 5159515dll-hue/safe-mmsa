"""模块8 API 测试：健康检查 / 模型状态 / 文件上传分析。"""
from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["language"] == "mn"


def test_models_status_all_stub_by_default():
    r = client.get("/api/models/status")
    assert r.status_code == 200
    body = r.json()
    assert body["all_stub"] is True
    assert body["mode"] == "stub_baseline"
    assert body["points"]["encoders"]["text"] == "stub"


def test_analyze_text_form():
    r = client.post("/api/analyze", data={"text": "I am so happy and great today"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "mn"
    assert len(body["modalities"]) == 3
    assert body["final_emotion"] == "happy"
    assert "score" in body["boundary"]


def test_analyze_empty_is_neutral():
    r = client.post("/api/analyze", data={})
    assert r.status_code == 200
    assert r.json()["final_emotion"] == "neutral"


def test_analyze_with_fake_uploads_degrades_gracefully():
    files = {
        "video": ("clip.mp4", io.BytesIO(b"not a real video"), "video/mp4"),
        "audio": ("clip.wav", io.BytesIO(b"not a real wav"), "audio/wav"),
    }
    r = client.post("/api/analyze", data={"text": "wonderful"}, files=files)
    assert r.status_code == 200
    body = r.json()
    by_mod = {m["modality"]: m for m in body["modalities"]}
    # 有上传 → available True，但无真实模型 → 无分布。
    assert by_mod["video"]["available"] is True
    assert by_mod["video"]["distribution"] is None
    assert by_mod["text"]["available"] is True
