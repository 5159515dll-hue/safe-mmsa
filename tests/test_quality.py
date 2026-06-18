"""模块2 信号质量测试：基础指标无依赖可算，重依赖缺失时优雅降级为 None。"""
from __future__ import annotations

from safemmsa.preprocessing import (
    extract_audio_quality,
    extract_text_quality,
    extract_video_quality,
)


def test_text_quality_basic_counts():
    q = extract_text_quality("hello world foo")
    assert q.text_length == 3
    assert q.valid_token_ratio == 1.0


def test_text_quality_empty_is_all_none():
    q = extract_text_quality("")
    assert q.text_length is None and q.valid_token_ratio is None


def test_text_quality_punctuation_lowers_valid_ratio():
    q = extract_text_quality("good !!! ??? bad")
    assert q.text_length == 4
    assert q.valid_token_ratio is not None and q.valid_token_ratio < 1.0


def test_text_quality_no_space_script_char_level():
    # 无空格脚本退化为字符级计数（这里用 CJK 模拟）。
    q = extract_text_quality("难过压力")
    assert q.text_length == 4


def test_video_quality_degrades_without_input_or_file():
    assert extract_video_quality(None).blur is None
    # 不存在的文件：无 opencv → None；有 opencv → 打开失败也 None。
    q = extract_video_quality("/no/such/file.mp4")
    assert q.blur is None and q.brightness is None


def test_audio_quality_degrades_without_input_or_file():
    assert extract_audio_quality(None).snr_db is None
    q = extract_audio_quality("/no/such/file.wav")
    assert q.snr_db is None and q.voiced_ratio is None
