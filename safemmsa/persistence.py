"""模块11 — 结果/历史持久化（JSON，沿用 AIoT JsonListStore 思路，无外部依赖）。

分析历史与评测 run 各落一个 JSON 文件（进程内线程安全）。存储目录默认 ``./storage``，
可用环境变量 ``SAFEMMSA_STORAGE_DIR`` 覆盖；目录被 .gitignore 忽略。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Optional


def storage_dir() -> str:
    return os.environ.get("SAFEMMSA_STORAGE_DIR", os.path.join(os.getcwd(), "storage"))


class JsonListStore:
    """把一组记录持久化为单个 JSON 数组文件，支持按 id 取。进程内线程安全。"""

    def __init__(self, name: str, *, max_items: int = 1000) -> None:
        self._path = os.path.join(storage_dir(), f"{name}.json")
        self._max_items = max_items
        self._lock = threading.Lock()
        os.makedirs(storage_dir(), exist_ok=True)

    def _read(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write(self, items: list[dict]) -> None:
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)  # 原子替换

    def append(self, record: dict[str, Any], *, record_id: Optional[str] = None) -> str:
        """追加一条记录，自动补 id/created_at；返回 id。超出上限按 FIFO 截断。"""
        rid = record_id or record.get("id") or uuid.uuid4().hex[:12]
        with self._lock:
            items = self._read()
            entry = {"id": rid, "created_at": time.time(), **record}
            items.append(entry)
            if len(items) > self._max_items:
                items = items[-self._max_items :]
            self._write(items)
        return rid

    def get(self, record_id: str) -> Optional[dict]:
        with self._lock:
            for item in reversed(self._read()):
                if item.get("id") == record_id:
                    return item
        return None

    def list(self, *, limit: int = 50) -> list[dict]:
        with self._lock:
            items = self._read()
        return list(reversed(items[-limit:]))

    def clear(self) -> None:
        with self._lock:
            self._write([])


# 模块级默认 store。
analyze_history = JsonListStore("analyze_history")
eval_runs = JsonListStore("eval_runs")
