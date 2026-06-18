"""模块9 评测台测试：metrics / runner / ablation / robustness / report / routes。"""
from __future__ import annotations

import os

import pytest

from eval import (
    accuracy,
    classification_report,
    confusion_matrix,
    load_manifest,
    macro_f1,
    run_manifest,
)
from eval.ablation import run_ablation
from eval.report import export, to_csv, to_json
from eval.robustness import drop_modality, run_robustness, text_char_noise
from eval.runner import EvalItem

_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "sample", "dev.jsonl")


# ── metrics ────────────────────────────────────────────────────────
def test_perfect_metrics():
    y = ["happy", "sad", "angry"]
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, y) == 1.0
    cm = confusion_matrix(y, y)
    assert int(cm.sum()) == 3 and int(cm.trace()) == 3


def test_metrics_partial():
    y_true = ["happy", "happy", "sad", "sad"]
    y_pred = ["happy", "sad", "sad", "sad"]
    assert accuracy(y_true, y_pred) == 0.75
    rep = classification_report(y_true, y_pred)
    assert rep.per_class["sad"].recall == 1.0
    assert rep.per_class["happy"].recall == 0.5
    assert rep.n == 4


def test_macro_f1_only_present_classes():
    # 只有 happy/sad 出现，macro 不应被 5 个 support=0 的类稀释。
    y = ["happy", "sad"]
    assert macro_f1(y, y) == 1.0


# ── runner ─────────────────────────────────────────────────────────
def test_load_manifest_and_run():
    items = load_manifest(_SAMPLE)
    assert len(items) == 10
    result = run_manifest(items, collect_per_item=True)
    assert result.n_items == 10
    # 桩关键词应让多数样本预测正确 → macro-F1 > 0.5。
    assert result.paired.initial.macro_f1 > 0.5
    # 默认 identity 增强 → final==initial。
    assert result.paired.delta_macro_f1 == 0.0
    assert result.paired.n_changed == 0
    assert result.per_item is not None and len(result.per_item) == 10


def test_run_manifest_neutral_when_no_signal():
    items = [EvalItem(id="x", label="neutral", text="the table is by the door")]
    result = run_manifest(items)
    assert result.paired.final.n == 1


# ── ablation ───────────────────────────────────────────────────────
def test_ablation_runs_variants_and_restores():
    from safemmsa import enhancement

    items = load_manifest(_SAMPLE)
    before = enhancement.get_enhancement()
    res = run_ablation(items, variants=("full", "no_enhancement"))
    assert set(res) == {"full", "no_enhancement"}
    # 跑完恢复原注册状态。
    assert enhancement.get_enhancement() is before


# ── robustness ─────────────────────────────────────────────────────
def test_drop_modality_transform():
    item = EvalItem(id="a", label="happy", text="great", video_path="/x.mp4")
    dropped = drop_modality("text")(item)
    assert dropped.text is None and dropped.video_path == "/x.mp4"


def test_text_noise_is_deterministic():
    item = EvalItem(id="a", label="happy", text="wonderful happy great day")
    t1 = text_char_noise(0.5, seed=7)(item).text
    t2 = text_char_noise(0.5, seed=7)(item).text
    assert t1 == t2  # 同 seed 可复现


def test_run_robustness_includes_clean():
    items = load_manifest(_SAMPLE)
    res = run_robustness(items, scenarios=("drop_text",))
    assert "clean" in res and "drop_text" in res
    # 丢掉文本（唯一有效模态）→ 全 neutral → macro-F1 退化。
    assert res["drop_text"].paired.final.macro_f1 <= res["clean"].paired.final.macro_f1


# ── report ─────────────────────────────────────────────────────────
def test_export_writes_json_and_csv(tmp_path):
    items = load_manifest(_SAMPLE)
    result = run_manifest(items)
    jp = to_json(result, str(tmp_path / "r.json"))
    cp = to_csv(result, str(tmp_path / "r.csv"))
    assert os.path.exists(jp) and os.path.exists(cp)
    paths = export(result, outdir=str(tmp_path / "out"))
    assert os.path.exists(paths["json"]) and os.path.exists(paths["csv"])


# ── routes ─────────────────────────────────────────────────────────
def test_eval_route_run_and_get(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    r = client.post("/api/eval/run", data={"manifest_path": _SAMPLE, "collect_per_item": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    rid = body["run_id"]
    assert body["n_items"] == 10
    # 取回。
    r2 = client.get(f"/api/eval/{rid}")
    assert r2.status_code == 200
    assert r2.json()["run_id"] == rid
