"""鲁棒性：模态缺失注入 + 文本噪声，记录退化与恢复。

模态缺失（drop）与文本字符噪声为确定性、可测；音/视频信号级加噪需要媒体解码，
留 ``register_av_noise`` 钩子（接入媒体库后生效），不在核心范围内实现。
"""
from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

from .runner import EvalItem, Transform, run_manifest
from .schemas import EvalRunResult

# 音/视频信号级加噪钩子（默认无）：transform(item) -> item，可对路径指向的媒体加噪后另存。
_av_noise: Optional[Transform] = None


def register_av_noise(transform: Optional[Transform]) -> None:
    """注册音/视频信号级加噪 transform（需媒体库；默认 None=不加噪）。"""
    global _av_noise
    _av_noise = transform


def drop_modality(modality: str) -> Transform:
    """返回一个把指定模态置空的 transform。"""

    def _t(item: EvalItem) -> EvalItem:
        return EvalItem(
            id=item.id,
            label=item.label,
            video_path=None if modality == "video" else item.video_path,
            audio_path=None if modality == "audio" else item.audio_path,
            text=None if modality == "text" else item.text,
        )

    return _t


def text_char_noise(prob: float = 0.2, seed: int = 0) -> Transform:
    """以概率 prob 随机替换文本中的字符（确定性 seed，便于复现）。"""
    rng = random.Random(seed)

    def _t(item: EvalItem) -> EvalItem:
        if not item.text:
            return item
        chars = list(item.text)
        for i, ch in enumerate(chars):
            if not ch.isspace() and rng.random() < prob:
                chars[i] = rng.choice("·×#@?")
        return EvalItem(
            id=item.id,
            label=item.label,
            video_path=item.video_path,
            audio_path=item.audio_path,
            text="".join(chars),
        )

    return _t


# 内置场景：name -> transform 工厂。
_SCENARIOS: dict[str, Callable[[], Transform]] = {
    "drop_video": lambda: drop_modality("video"),
    "drop_audio": lambda: drop_modality("audio"),
    "drop_text": lambda: drop_modality("text"),
    "text_noise_20": lambda: text_char_noise(0.2),
    "text_noise_50": lambda: text_char_noise(0.5),
}


def run_robustness(
    items: Sequence[EvalItem],
    scenarios: Sequence[str] = ("drop_video", "drop_audio", "text_noise_20"),
) -> dict[str, EvalRunResult]:
    """跑 clean 基线 + 各扰动场景，返回 {scenario: 结果}（含 clean）。

    退化 = clean.final.macro_f1 − scenario.final.macro_f1。
    """
    results: dict[str, EvalRunResult] = {
        "clean": run_manifest(items, run_id="robust_clean", config={"scenario": "clean"})
    }
    for name in scenarios:
        if name == "av_noise":
            transform = _av_noise
            if transform is None:
                continue  # 未注册媒体加噪 → 跳过
        else:
            factory = _SCENARIOS.get(name)
            if factory is None:
                raise ValueError(f"unknown robustness scenario: {name}")
            transform = factory()
        results[name] = run_manifest(
            items, transform=transform, run_id=f"robust_{name}", config={"scenario": name}
        )
    return results
