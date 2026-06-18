"""模块1 — 语种识别（脚本启发式，迁自 AIoT Copilot）。

系统定位为蒙古语：``AnalyzeResult.language`` 对外**固定为 "mn"**（见 SYSTEM_LANGUAGE）。
``detect_language`` 仍保留，用于文本质量/桩编码器对输入脚本做提示性判别（蒙文/西里尔→mn，
CJK→zh，否则 en），不改变系统对外的蒙古语定位。
"""
from __future__ import annotations

# 系统对外固定语种（docs/system-design.md §3：language 固定 "mn"）。
SYSTEM_LANGUAGE: str = "mn"


def detect_language(text: str) -> str:
    """脚本启发式语种识别：传统蒙文/西里尔→mn，CJK→zh，否则 en。

    传统蒙古文 Unicode 块 U+1800–U+18AF；西里尔块 U+0400–U+04FF（蒙古国现行书写）。
    仅作输入提示，系统对外语种恒为 SYSTEM_LANGUAGE。
    """
    for ch in text:
        if "᠀" <= ch <= "᢯" or "Ѐ" <= ch <= "ӿ":
            return "mn"
    for ch in text:
        if "一" <= ch <= "鿿":
            return "zh"
    return "en"
