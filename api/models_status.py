"""模型状态汇报：一眼看出每个插点当前是"桩/基线"还是"真模型"。

供 GET /api/models/status 使用，也供前端在页面上标注"当前为基线演示 / 已接入真实模型"。
"""
from __future__ import annotations

from safemmsa import backbone, enhancement, encoders
from safemmsa.states import get_reliability_head


def models_status() -> dict:
    """返回各 register_* 插点的桩/真模型状态。"""
    encoder_status = {
        m: ("stub" if encoders.is_stub(m) else "trained") for m in ("video", "audio", "text")
    }
    points = {
        "encoders": encoder_status,
        "fusion_backbone": "stub" if backbone.is_stub() else "trained",
        "enhancement": "stub" if enhancement.is_stub() else "trained",
        "reliability_head": "default" if get_reliability_head() is None else "trained",
    }
    any_real = (
        any(v == "trained" for v in encoder_status.values())
        or points["fusion_backbone"] == "trained"
        or points["enhancement"] == "trained"
        or points["reliability_head"] == "trained"
    )
    return {
        "mode": "real_models" if any_real else "stub_baseline",
        "all_stub": not any_real,
        "points": points,
        "note": (
            "当前全部为桩/基线，系统端到端可跑但不产生非零修正；"
            "训练完成后经 register_* 挂载真实模型即生效，其余代码零改动。"
            if not any_real
            else "已接入至少一个真实模型。"
        ),
    }
