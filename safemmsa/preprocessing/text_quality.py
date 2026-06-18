"""文本信号质量：长度 / 有效 token 比例 / 低频子词比例。

基础指标（长度、有效 token 比例）无外部依赖；低频子词比例可选用 CINO tokenizer
（transformers），缺失时降级为字符级启发式或 None。纯信号级，不做情感判断。
"""
from __future__ import annotations

from typing import Optional

from ..contracts import ModalitySignalQuality

# CINO（中文少数民族预训练，含蒙古语）tokenizer，仅用于子词统计，不做模型推理。
_CINO_NAME = "hfl/cino-base-v2"
_tokenizer = None
_tokenizer_tried = False


def _maybe_tokenizer():
    """惰性加载 CINO tokenizer；不可用返回 None（不抛错）。"""
    global _tokenizer, _tokenizer_tried
    if _tokenizer_tried:
        return _tokenizer
    _tokenizer_tried = True
    try:  # pragma: no cover - 依赖环境
        from transformers import AutoTokenizer  # type: ignore

        _tokenizer = AutoTokenizer.from_pretrained(_CINO_NAME)
    except Exception:
        _tokenizer = None
    return _tokenizer


def _basic_tokens(text: str) -> list[str]:
    """空格分词；无空格脚本（蒙古文/CJK）退化为字符级。"""
    tokens = text.split()
    if len(tokens) <= 1 and len(text) > 1:
        return [ch for ch in text if not ch.isspace()]
    return tokens


def extract_text_quality(text: Optional[str]) -> ModalitySignalQuality:
    """从文本算信号质量。空文本 → 全 None（由编码器标 unavailable）。"""
    if not text or not text.strip():
        return ModalitySignalQuality()
    text = text.strip()
    tokens = _basic_tokens(text)
    text_length = len(tokens)

    # 有效 token 比例：含字母/数字（非纯标点/空白）的占比。
    valid = sum(1 for t in tokens if any(ch.isalnum() for ch in t))
    valid_token_ratio = valid / text_length if text_length else None

    # 低频子词比例：用 CINO tokenizer 时统计 [UNK] / 单字符子词占比；否则字符级启发式。
    low_freq_subword_ratio: Optional[float] = None
    tok = _maybe_tokenizer()
    if tok is not None:  # pragma: no cover - 依赖环境
        try:
            pieces = tok.tokenize(text)
            if pieces:
                unk = getattr(tok, "unk_token", "[UNK]")
                rare = sum(1 for p in pieces if p == unk or len(p.replace("##", "")) <= 1)
                low_freq_subword_ratio = rare / len(pieces)
        except Exception:
            low_freq_subword_ratio = None

    return ModalitySignalQuality(
        text_length=text_length,
        valid_token_ratio=None if valid_token_ratio is None else round(valid_token_ratio, 4),
        low_freq_subword_ratio=(
            None if low_freq_subword_ratio is None else round(low_freq_subword_ratio, 4)
        ),
    )
