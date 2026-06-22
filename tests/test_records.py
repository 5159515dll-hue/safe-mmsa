"""模块11 — 持久化契约与映射测试：AnalyzeResult → 规范化 bundle。"""
from __future__ import annotations

from safemmsa.records import AnalysisBundle, analyze_result_to_bundle


def test_bundle_mapping_shape(stub_result):
    bundle = analyze_result_to_bundle(stub_result, user_id="usr_x")
    rid = bundle.record.record_id

    # FK 一致：所有子行回指同一 record_id
    assert bundle.state_diagnosis.record_id == rid
    assert all(m.record_id == rid for m in bundle.modality_outputs)
    assert all(r.record_id == rid for r in bundle.reliabilities)

    # 内容对齐 compute 契约
    assert bundle.record.user_id == "usr_x"
    assert bundle.record.final_emotion == stub_result.final_emotion
    assert bundle.record.boundary_score == stub_result.boundary.score
    assert len(bundle.modality_outputs) == 3  # 始终 3 模态
    assert bundle.state_diagnosis.comp_top1 == stub_result.competition.top1


def test_bundle_round_trip(stub_result):
    bundle = analyze_result_to_bundle(stub_result, user_id="usr_x")
    again = AnalysisBundle.model_validate(bundle.model_dump())
    assert again.model_dump() == bundle.model_dump()


def test_record_id_is_stable_across_children(stub_result):
    bundle = analyze_result_to_bundle(stub_result, user_id="anon", record_id="rec_fixed")
    assert bundle.record.record_id == "rec_fixed"
    assert bundle.state_diagnosis.record_id == "rec_fixed"
    assert {m.record_id for m in bundle.modality_outputs} == {"rec_fixed"}
