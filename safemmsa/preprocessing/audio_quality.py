"""音频信号质量：SNR / 有声帧比例 / 静音比例。

依赖 librosa（可选）。缺失或解码失败时全部降级为 None。基于能量门限做 VAD 近似，
不依赖深度模型。纯信号级。
"""
from __future__ import annotations

from typing import Optional

from ..contracts import ModalitySignalQuality


def _maybe_librosa():
    try:  # pragma: no cover - 依赖环境
        import librosa  # type: ignore
        import numpy as np  # noqa: F401

        return librosa
    except Exception:
        return None


def extract_audio_quality(path: Optional[str]) -> ModalitySignalQuality:
    """从音频文件算信号质量。无路径/无 librosa/解码失败 → 全 None。"""
    if not path:
        return ModalitySignalQuality()
    librosa = _maybe_librosa()
    if librosa is None:
        return ModalitySignalQuality()

    try:  # pragma: no cover - 依赖环境/IO
        import numpy as np

        y, sr = librosa.load(path, sr=16000, mono=True)
        if y.size == 0:
            return ModalitySignalQuality()

        # 短时能量 → 帧级 RMS。
        frame_length = 400  # 25ms @16k
        hop = 160          # 10ms
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        if rms.size == 0:
            return ModalitySignalQuality()

        # 能量门限：取最大能量的一个比例作为有声/静音分界。
        peak = float(rms.max())
        if peak <= 1e-9:
            return ModalitySignalQuality(snr_db=None, voiced_ratio=0.0, silence_ratio=1.0)
        threshold = 0.1 * peak
        voiced_mask = rms >= threshold
        voiced_ratio = float(voiced_mask.mean())
        silence_ratio = float(1.0 - voiced_ratio)

        # SNR 近似：有声帧能量 vs 静音帧能量（dB）。
        voiced_power = float((rms[voiced_mask] ** 2).mean()) if voiced_mask.any() else 0.0
        noise_power = float((rms[~voiced_mask] ** 2).mean()) if (~voiced_mask).any() else 0.0
        snr_db: Optional[float]
        if noise_power <= 1e-12:
            snr_db = 40.0 if voiced_power > 0 else None  # 几乎无噪 → 给高 SNR 上限
        elif voiced_power <= 1e-12:
            snr_db = 0.0
        else:
            snr_db = round(float(10.0 * np.log10(voiced_power / noise_power)), 2)

        return ModalitySignalQuality(
            snr_db=snr_db,
            voiced_ratio=round(voiced_ratio, 4),
            silence_ratio=round(silence_ratio, 4),
        )
    except Exception:
        return ModalitySignalQuality()
