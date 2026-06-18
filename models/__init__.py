"""模块12 — 模型挂载约定（插点 + 现在写约定，本期只给桩/no-op）。

``load_models(config)`` 按配置把训练产物经 ``register_*`` 挂到核心系统的插点上。
本期没有真实权重 → 默认 no-op（全用桩/基线），其余代码零改动。训练完成后：
1. 把 checkpoint 放进 ``models/weights/``；
2. 在 ``models/backends/<name>.py`` 实现 ``build(...) -> Backend``（签名见 models/README.md）；
3. 在 config 里把对应组件标 enabled + 指向权重；``load_models`` 一调用即生效。

**核心包不依赖 torch**；真实 backend 的 torch import 只发生在这里（挂载阶段）。
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Optional

from safemmsa import (
    register_encoder,
    register_enhancement,
    register_fusion_backbone,
    register_reliability_head,
)

# 组件 -> (注册函数, 期望的 backends 子模块名)
_COMPONENTS = {
    "encoder_text": ("encoder", "text"),
    "encoder_video": ("encoder", "video"),
    "encoder_audio": ("encoder", "audio"),
    "fusion_backbone": ("fusion_backbone", None),
    "enhancement": ("enhancement", None),
    "reliability_head": ("reliability_head", None),
}


def default_config() -> dict[str, Any]:
    """默认配置：全部 disabled（用桩/基线）。"""
    return {name: {"enabled": False} for name in _COMPONENTS}


def _load_config(config: Optional[dict | str]) -> dict[str, Any]:
    """配置来源：显式 dict / JSON 路径 / 环境变量 SAFEMMSA_MODELS_CONFIG / 默认。"""
    if isinstance(config, dict):
        merged = default_config()
        merged.update(config)
        return merged
    path = config or os.environ.get("SAFEMMSA_MODELS_CONFIG")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            merged = default_config()
            merged.update(json.load(f))
            return merged
    return default_config()


def _register(component: str, backend: Any) -> None:
    kind, modality = _COMPONENTS[component]
    if kind == "encoder":
        register_encoder(modality, backend)  # type: ignore[arg-type]
    elif kind == "fusion_backbone":
        register_fusion_backbone(backend)
    elif kind == "enhancement":
        register_enhancement(backend)
    elif kind == "reliability_head":
        register_reliability_head(backend)


def load_models(config: Optional[dict | str] = None) -> dict[str, Any]:
    """按配置挂载真实模型。无配置/未实现 → no-op（保持桩/基线）。

    返回 {"loaded": [...], "skipped": {...}}：loaded 为成功挂载的组件，
    skipped 记录原因（disabled / backend 未实现 / 加载失败），便于排查。
    """
    cfg = _load_config(config)
    loaded: list[str] = []
    skipped: dict[str, str] = {}

    for component, spec in cfg.items():
        if component not in _COMPONENTS:
            skipped[component] = "unknown component"
            continue
        if not spec.get("enabled"):
            skipped[component] = "disabled (使用桩/基线)"
            continue
        # 约定：models/backends/<component>.py 暴露 build(spec) -> Backend。
        module_name = f"models.backends.{component}"
        try:
            mod = importlib.import_module(module_name)
            backend = mod.build(spec)  # type: ignore[attr-defined]
            _register(component, backend)
            loaded.append(component)
        except ModuleNotFoundError:
            skipped[component] = f"backend 未实现：{module_name}"
        except Exception as e:  # 加载/权重错误 → 保持桩，不让系统崩
            skipped[component] = f"加载失败：{type(e).__name__}: {e}"

    return {"loaded": loaded, "skipped": skipped}


def unload_models() -> None:
    """全部恢复默认桩/基线（测试或回滚用）。"""
    for component in _COMPONENTS:
        _register(component, None)
