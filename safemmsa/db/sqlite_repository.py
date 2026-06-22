"""SqliteRepository —— ``Repository`` 接口的 SQLite（stdlib ``sqlite3``）实现。

证明「JSON 与 RDBMS 共用同一套契约、平滑迁移」：与 JsonRepository 同接口、同入参/出参，
仅写入层不同。建表用 db/schema.sql（同一份规范化 DDL）。JSON 型字段以 TEXT 存。
启用 ``PRAGMA foreign_keys=ON``，FK / UNIQUE / 1:1 约束由数据库强制。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from ..persistence import storage_dir
from ..records import (
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
from ..repository import Repository

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _dumps(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, ensure_ascii=False)


def _loads(s: Any) -> Any:
    return None if s is None else json.loads(s)


@contextmanager
def _tx(conn: sqlite3.Connection):
    """多语句写：成功提交，异常回滚（避免半截行留在未提交事务里）。"""
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class SqliteRepository(Repository):
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or os.environ.get(
            "SAFEMMSA_DB_PATH", os.path.join(storage_dir(), "safemmsa.db")
        )
        if self._path != ":memory:":
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.commit()

    # ── users ──────────────────────────────────────────────────────
    def _insert_user(self, user: User) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO user (user_id, username, password_hash, role, created_at) "
                "VALUES (?,?,?,?,?)",
                (user.user_id, user.username, user.password_hash, user.role, user.created_at),
            )
            self._conn.commit()

    def create_user(self, username: str, password: str, *, role: Role = "user") -> User:
        from .. import auth

        username = username.strip()
        if not username:
            raise ValueError("username 不能为空")
        if self.get_user_by_username(username) is not None:
            raise ValueError(f"username 已存在: {username}")
        user = User(username=username, password_hash=auth.hash_password(password), role=role)
        try:
            self._insert_user(user)
        except sqlite3.IntegrityError as e:  # UNIQUE(username) 兜底
            raise ValueError(f"username 已存在: {username}") from e
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user WHERE user_id=?", (user_id,)
            ).fetchone()
        return User.model_validate(dict(row)) if row else None

    def get_user_by_username(self, username: str) -> Optional[User]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user WHERE username=?", (username,)
            ).fetchone()
        return User.model_validate(dict(row)) if row else None

    # ── analysis ───────────────────────────────────────────────────
    def save_analysis(self, bundle: AnalysisBundle) -> str:
        r = bundle.record
        sd = bundle.state_diagnosis
        with self._lock, _tx(self._conn):
            c = self._conn
            c.execute(
                "INSERT INTO analysis_record (record_id, user_id, language, final_emotion, "
                "enhancement, correction_magnitude, bounded, initial_top, boundary_score, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    r.record_id, r.user_id, r.language, r.final_emotion, r.enhancement,
                    r.correction_magnitude, int(r.bounded), r.initial_top, r.boundary_score,
                    r.created_at,
                ),
            )
            c.executemany(
                "INSERT INTO modality_output (mo_id, record_id, modality, available, backend, "
                "distribution, logits, quality) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        m.mo_id, m.record_id, m.modality, int(m.available), m.backend,
                        _dumps(m.distribution), _dumps(m.logits), _dumps(m.quality),
                    )
                    for m in bundle.modality_outputs
                ],
            )
            c.execute(
                "INSERT INTO state_diagnosis (sd_id, record_id, boundary_max_confidence, "
                "boundary_top2_margin, boundary_norm_entropy, boundary_score, comp_top1, comp_top2, "
                "comp_margin, comp_entropy, comp_vector, comp_per_modality_direction, "
                "consistency_pairwise) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sd.sd_id, sd.record_id, sd.boundary_max_confidence, sd.boundary_top2_margin,
                    sd.boundary_norm_entropy, sd.boundary_score, sd.comp_top1, sd.comp_top2,
                    sd.comp_margin, sd.comp_entropy, _dumps(sd.comp_vector),
                    _dumps(sd.comp_per_modality_direction), _dumps(sd.consistency_pairwise),
                ),
            )
            c.executemany(
                "INSERT INTO modality_reliability (mr_id, record_id, modality, reliability, "
                "quality_part, uncertainty_part) VALUES (?,?,?,?,?,?)",
                [
                    (x.mr_id, x.record_id, x.modality, x.reliability, x.quality_part, x.uncertainty_part)
                    for x in bundle.reliabilities
                ],
            )
        return r.record_id

    def get_analysis(self, record_id: str) -> Optional[AnalysisBundle]:
        with self._lock:
            c = self._conn
            rec = c.execute(
                "SELECT * FROM analysis_record WHERE record_id=?", (record_id,)
            ).fetchone()
            if rec is None:
                return None
            mo_rows = c.execute(
                "SELECT * FROM modality_output WHERE record_id=?", (record_id,)
            ).fetchall()
            sd = c.execute(
                "SELECT * FROM state_diagnosis WHERE record_id=?", (record_id,)
            ).fetchone()
            rel_rows = c.execute(
                "SELECT * FROM modality_reliability WHERE record_id=?", (record_id,)
            ).fetchall()
        if sd is None:
            return None
        record = AnalysisRecord.model_validate({**dict(rec), "bounded": bool(rec["bounded"])})
        modality_outputs = [
            ModalityOutputRow.model_validate(
                {
                    **dict(m),
                    "available": bool(m["available"]),
                    "distribution": _loads(m["distribution"]),
                    "logits": _loads(m["logits"]),
                    "quality": _loads(m["quality"]) or {},
                }
            )
            for m in mo_rows
        ]
        state = StateDiagnosis.model_validate(
            {
                **dict(sd),
                "comp_vector": _loads(sd["comp_vector"]) or [],
                "comp_per_modality_direction": _loads(sd["comp_per_modality_direction"]) or {},
                "consistency_pairwise": _loads(sd["consistency_pairwise"]) or {},
            }
        )
        reliabilities = [ModalityReliabilityRow.model_validate(dict(x)) for x in rel_rows]
        return AnalysisBundle(
            record=record,
            modality_outputs=modality_outputs,
            state_diagnosis=state,
            reliabilities=reliabilities,
        )

    def list_analyses(
        self, *, user_id: Optional[str] = None, limit: int = 20
    ) -> list[AnalysisRecord]:
        sql = "SELECT * FROM analysis_record"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id=?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            AnalysisRecord.model_validate({**dict(r), "bounded": bool(r["bounded"])}) for r in rows
        ]

    # ── eval ───────────────────────────────────────────────────────
    def save_eval_run(self, run: EvalRun, items: list[EvalItemRow]) -> str:
        with self._lock, _tx(self._conn):
            c = self._conn
            c.execute(
                "INSERT INTO eval_run (run_id, user_id, dataset, n_items, macro_f1, accuracy, "
                "delta_macro_f1, delta_accuracy, mean_correction_magnitude, confusion, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id, run.user_id, run.dataset, run.n_items, run.macro_f1, run.accuracy,
                    run.delta_macro_f1, run.delta_accuracy, run.mean_correction_magnitude,
                    _dumps(run.confusion), run.created_at,
                ),
            )
            c.executemany(
                "INSERT INTO eval_item (item_id, run_id, gold_label, video_path, audio_path, text, "
                "pred_initial, pred_final, correction_magnitude) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        it.item_id, it.run_id, it.gold_label, it.video_path, it.audio_path, it.text,
                        it.pred_initial, it.pred_final, it.correction_magnitude,
                    )
                    for it in items
                ],
            )
        return run.run_id

    def get_eval_run(self, run_id: str) -> Optional[tuple[EvalRun, list[EvalItemRow]]]:
        with self._lock:
            c = self._conn
            row = c.execute("SELECT * FROM eval_run WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            items = c.execute("SELECT * FROM eval_item WHERE run_id=?", (run_id,)).fetchall()
        run = EvalRun.model_validate({**dict(row), "confusion": _loads(row["confusion"])})
        return run, [EvalItemRow.model_validate(dict(it)) for it in items]

    def list_eval_runs(
        self, *, user_id: Optional[str] = None, limit: int = 20
    ) -> list[EvalRun]:
        sql = "SELECT * FROM eval_run"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id=?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [EvalRun.model_validate({**dict(r), "confusion": _loads(r["confusion"])}) for r in rows]

    # ── logs ───────────────────────────────────────────────────────
    def add_log(self, log: OperationLog) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT INTO operation_log (log_id, user_id, action, detail, created_at) "
                "VALUES (?,?,?,?,?)",
                (log.log_id, log.user_id, log.action, _dumps(log.detail), log.created_at),
            )
            self._conn.commit()
        return log.log_id

    def list_logs(self, *, limit: int = 50) -> list[OperationLog]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM operation_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            OperationLog.model_validate({**dict(r), "detail": _loads(r["detail"]) or {}})
            for r in rows
        ]
