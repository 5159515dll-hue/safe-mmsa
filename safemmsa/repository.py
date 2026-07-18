"""模块11c — 持久化仓储抽象 + JSON 后端 + 后端工厂。

``Repository`` 是统一接口；当前默认 ``JsonRepository``（规范化、每实体一个 JSON 文件），
后期切 ``SqliteRepository``（db/sqlite_repository.py）只换实现、调用方零改动——这正是
「JSON 与 RDBMS 共用同一套契约、平滑迁移」的落点。后端由 ``SAFEMMSA_DB_BACKEND``
（``json`` | ``sqlite``）选择。
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

from . import auth
from .persistence import storage_dir
from .records import (
    AnalysisBundle,
    AnalysisRecord,
    EvalItemRow,
    EvalRun,
    ModalityOutputRow,
    ModalityReliabilityRow,
    OperationLog,
    Role,
    StateDiagnosis,
    User,
)

ANONYMOUS_USER_ID = "usr_anonymous"
ANONYMOUS_USERNAME = "anonymous"


# ── 一个规范化 JSON 表（原子写、进程内线程安全）──────────────────────
class _JsonTable:
    def __init__(self, name: str) -> None:
        self._path = os.path.join(storage_dir(), f"{name}.json")
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

    def _write(self, rows: list[dict]) -> None:
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    def insert(self, row: dict) -> None:
        with self._lock:
            rows = self._read()
            rows.append(row)
            self._write(rows)

    def insert_unique(self, row: dict, *, key: str, value: object) -> bool:
        """原子「查唯一→插入」：单次持锁内检查 ``key==value`` 是否已存在，不存在才追加。
        返回是否插入成功。用于替代「先 get 再 insert」的 TOCTOU（用户名唯一性靠此保证）。"""
        with self._lock:
            rows = self._read()
            if any(r.get(key) == value for r in rows):
                return False
            rows.append(row)
            self._write(rows)
            return True

    def insert_many(self, new_rows: list[dict]) -> None:
        if not new_rows:
            return
        with self._lock:
            rows = self._read()
            rows.extend(new_rows)
            self._write(rows)

    def all(self) -> list[dict]:
        with self._lock:
            return self._read()

    def clear(self) -> None:
        with self._lock:
            self._write([])


# ── 仓储接口 ───────────────────────────────────────────────────────
class Repository(ABC):
    # users
    @abstractmethod
    def create_user(self, username: str, password: str, *, role: Role = "user") -> User: ...
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]: ...
    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[User]: ...

    # analysis
    @abstractmethod
    def save_analysis(self, bundle: AnalysisBundle) -> str: ...
    @abstractmethod
    def get_analysis(self, record_id: str) -> Optional[AnalysisBundle]: ...
    @abstractmethod
    def list_analyses(self, *, user_id: Optional[str] = None, limit: int = 20) -> list[AnalysisRecord]: ...

    # eval
    @abstractmethod
    def save_eval_run(self, run: EvalRun, items: list[EvalItemRow]) -> str: ...
    @abstractmethod
    def get_eval_run(self, run_id: str) -> Optional[tuple[EvalRun, list[EvalItemRow]]]: ...
    @abstractmethod
    def list_eval_runs(self, *, user_id: Optional[str] = None, limit: int = 20) -> list[EvalRun]: ...

    # logs
    @abstractmethod
    def add_log(self, log: OperationLog) -> str: ...
    @abstractmethod
    def list_logs(self, *, limit: int = 50) -> list[OperationLog]: ...

    # 公共：确保存在匿名用户（未登录分析归属它，保证 FK 有效）
    def ensure_anonymous_user(self) -> User:
        existing = self.get_user(ANONYMOUS_USER_ID)
        if existing is not None:
            return existing
        user = User(
            user_id=ANONYMOUS_USER_ID,
            username=ANONYMOUS_USERNAME,
            password_hash=auth.hash_password(os.urandom(16).hex()),  # 不可登录
            role="user",
        )
        self._insert_user(user)
        return user

    @abstractmethod
    def _insert_user(self, user: User) -> None: ...


# ── JSON 后端（规范化，每实体一个文件）──────────────────────────────
class JsonRepository(Repository):
    def __init__(self) -> None:
        self.users = _JsonTable("users")
        self.analysis_records = _JsonTable("analysis_records")
        self.modality_outputs = _JsonTable("modality_outputs")
        self.state_diagnoses = _JsonTable("state_diagnoses")
        self.modality_reliabilities = _JsonTable("modality_reliabilities")
        self.eval_runs_t = _JsonTable("eval_runs_norm")
        self.eval_items = _JsonTable("eval_items")
        self.operation_logs = _JsonTable("operation_logs")

    # users
    def _insert_user(self, user: User) -> None:
        # 幂等按 user_id 唯一插入（供 ensure_anonymous_user 用，避免重复 PK 行）。
        self.users.insert_unique(user.model_dump(), key="user_id", value=user.user_id)

    def create_user(self, username: str, password: str, *, role: Role = "user") -> User:
        username = username.strip()
        if not username:
            raise ValueError("username 不能为空")
        user = User(username=username, password_hash=auth.hash_password(password), role=role)
        # 原子唯一插入：并发同名注册只有一个成功，其余得到 409（与 SQLite UNIQUE 行为一致）。
        if not self.users.insert_unique(user.model_dump(), key="username", value=username):
            raise ValueError(f"username 已存在: {username}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        for r in self.users.all():
            if r.get("user_id") == user_id:
                return User.model_validate(r)
        return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        for r in self.users.all():
            if r.get("username") == username:
                return User.model_validate(r)
        return None

    # analysis
    def save_analysis(self, bundle: AnalysisBundle) -> str:
        self.analysis_records.insert(bundle.record.model_dump())
        self.modality_outputs.insert_many([m.model_dump() for m in bundle.modality_outputs])
        self.state_diagnoses.insert(bundle.state_diagnosis.model_dump())
        self.modality_reliabilities.insert_many([r.model_dump() for r in bundle.reliabilities])
        return bundle.record.record_id

    def get_analysis(self, record_id: str) -> Optional[AnalysisBundle]:
        rec = next(
            (r for r in self.analysis_records.all() if r.get("record_id") == record_id), None
        )
        if rec is None:
            return None
        mo = [
            ModalityOutputRow.model_validate(r)
            for r in self.modality_outputs.all()
            if r.get("record_id") == record_id
        ]
        sd_row = next(
            (r for r in self.state_diagnoses.all() if r.get("record_id") == record_id), None
        )
        if sd_row is None:
            return None
        rel = [
            ModalityReliabilityRow.model_validate(r)
            for r in self.modality_reliabilities.all()
            if r.get("record_id") == record_id
        ]
        return AnalysisBundle(
            record=AnalysisRecord.model_validate(rec),
            modality_outputs=mo,
            state_diagnosis=StateDiagnosis.model_validate(sd_row),
            reliabilities=rel,
        )

    def list_analyses(self, *, user_id: Optional[str] = None, limit: int = 20) -> list[AnalysisRecord]:
        rows = self.analysis_records.all()
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return [AnalysisRecord.model_validate(r) for r in rows[:limit]]

    # eval
    def save_eval_run(self, run: EvalRun, items: list[EvalItemRow]) -> str:
        self.eval_runs_t.insert(run.model_dump())
        self.eval_items.insert_many([it.model_dump() for it in items])
        return run.run_id

    def get_eval_run(self, run_id: str) -> Optional[tuple[EvalRun, list[EvalItemRow]]]:
        row = next((r for r in self.eval_runs_t.all() if r.get("run_id") == run_id), None)
        if row is None:
            return None
        items = [
            EvalItemRow.model_validate(r)
            for r in self.eval_items.all()
            if r.get("run_id") == run_id
        ]
        return EvalRun.model_validate(row), items

    def list_eval_runs(self, *, user_id: Optional[str] = None, limit: int = 20) -> list[EvalRun]:
        rows = self.eval_runs_t.all()
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return [EvalRun.model_validate(r) for r in rows[:limit]]

    # logs
    def add_log(self, log: OperationLog) -> str:
        self.operation_logs.insert(log.model_dump())
        return log.log_id

    def list_logs(self, *, limit: int = 50) -> list[OperationLog]:
        rows = self.operation_logs.all()
        rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return [OperationLog.model_validate(r) for r in rows[:limit]]


# ── 后端工厂（进程内单例）──────────────────────────────────────────
_repo: Optional[Repository] = None
_repo_lock = threading.Lock()


def get_repository() -> Repository:
    global _repo
    with _repo_lock:
        if _repo is None:
            backend = os.environ.get("SAFEMMSA_DB_BACKEND", "json").strip().lower()
            if backend == "sqlite":
                from .db.sqlite_repository import SqliteRepository

                _repo = SqliteRepository()
            else:
                _repo = JsonRepository()
            _repo.ensure_anonymous_user()
        return _repo


def reset_repository() -> None:
    """丢弃单例（测试用：切目录/后端后强制重建）。"""
    global _repo
    with _repo_lock:
        _repo = None
