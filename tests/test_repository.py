"""模块11 — 仓储测试：JSON 与 SQLite 两后端共用同一接口、同一行为（参数化）。

同一组断言跑过两后端，即验证「JSON 与 RDBMS 共用契约、可平滑迁移」。
"""
from __future__ import annotations

import sqlite3

import pytest

from safemmsa import auth
from safemmsa.db.sqlite_repository import SqliteRepository
from safemmsa.records import (
    EvalItemRow,
    EvalRun,
    OperationLog,
    analyze_result_to_bundle,
)
from safemmsa.repository import ANONYMOUS_USER_ID, JsonRepository


@pytest.fixture(params=["json", "sqlite"])
def repo(request):
    r = JsonRepository() if request.param == "json" else SqliteRepository(":memory:")
    r.ensure_anonymous_user()
    return r


def test_anonymous_seeded(repo):
    anon = repo.get_user(ANONYMOUS_USER_ID)
    assert anon is not None and anon.username == "anonymous"


def test_user_create_and_lookup(repo):
    u = repo.create_user("alice", "secret123")
    assert repo.get_user(u.user_id).username == "alice"
    assert repo.get_user_by_username("alice").user_id == u.user_id
    assert auth.verify_password("secret123", u.password_hash)
    assert not auth.verify_password("wrong", u.password_hash)


def test_duplicate_username_rejected(repo):
    repo.create_user("bob", "secret123")
    with pytest.raises(ValueError):
        repo.create_user("bob", "another123")


def test_save_and_get_analysis_round_trip(repo, stub_result):
    bundle = analyze_result_to_bundle(stub_result, user_id=ANONYMOUS_USER_ID)
    rid = repo.save_analysis(bundle)
    got = repo.get_analysis(rid)
    assert got is not None
    assert got.model_dump() == bundle.model_dump()
    assert repo.get_analysis("nope") is None


def test_list_analyses_filtered_by_user(repo, stub_result):
    repo.create_user("carol", "secret123")
    carol = repo.get_user_by_username("carol")
    b1 = analyze_result_to_bundle(stub_result, user_id=carol.user_id)
    b2 = analyze_result_to_bundle(stub_result, user_id=ANONYMOUS_USER_ID)
    repo.save_analysis(b1)
    repo.save_analysis(b2)
    mine = repo.list_analyses(user_id=carol.user_id)
    assert len(mine) == 1 and mine[0].record_id == b1.record.record_id


def test_eval_run_round_trip(repo):
    run = EvalRun(user_id=ANONYMOUS_USER_ID, dataset="demo", n_items=2, macro_f1=0.5, accuracy=0.5)
    items = [
        EvalItemRow(run_id=run.run_id, gold_label="happy", text="a", pred_initial="happy", pred_final="happy"),
        EvalItemRow(run_id=run.run_id, gold_label="sad", text="b", pred_initial="neutral", pred_final="sad"),
    ]
    repo.save_eval_run(run, items)
    got = repo.get_eval_run(run.run_id)
    assert got is not None
    got_run, got_items = got
    assert got_run.dataset == "demo"
    assert len(got_items) == 2
    assert {i.gold_label for i in got_items} == {"happy", "sad"}
    assert [r.run_id for r in repo.list_eval_runs()][0] == run.run_id


def test_operation_log(repo):
    repo.add_log(OperationLog(user_id=ANONYMOUS_USER_ID, action="analyze", detail={"k": "v"}))
    logs = repo.list_logs()
    assert logs and logs[0].action == "analyze" and logs[0].detail == {"k": "v"}


def test_sqlite_enforces_one_to_one_state_diagnosis(stub_result):
    """SQLite 后端：state_diagnosis.record_id UNIQUE → 重复保存触发 IntegrityError。"""
    repo = SqliteRepository(":memory:")
    repo.ensure_anonymous_user()
    bundle = analyze_result_to_bundle(stub_result, user_id=ANONYMOUS_USER_ID)
    repo.save_analysis(bundle)
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_analysis(bundle)  # 同 record_id / sd record_id 再插入违反 PK/UNIQUE


def test_sqlite_foreign_key_enforced():
    """SQLite 后端：未知 user_id 的分析记录违反外键。"""
    repo = SqliteRepository(":memory:")
    repo.ensure_anonymous_user()
    from safemmsa.records import AnalysisBundle, AnalysisRecord, StateDiagnosis

    bundle = AnalysisBundle(
        record=AnalysisRecord(
            user_id="ghost", language="mn", final_emotion="happy", enhancement="identity_baseline",
            correction_magnitude=0.0, initial_top="happy", boundary_score=0.1,
        ),
        modality_outputs=[],
        state_diagnosis=StateDiagnosis(
            record_id="x", boundary_max_confidence=0.5, boundary_top2_margin=0.1,
            boundary_norm_entropy=0.5, boundary_score=0.1, comp_top1="happy", comp_top2="sad",
            comp_margin=0.1, comp_entropy=0.5,
        ),
    )
    # sd.record_id 与 record.record_id 不一致也会违反 FK；这里聚焦 user_id 外键
    bundle.state_diagnosis.record_id = bundle.record.record_id
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_analysis(bundle)
