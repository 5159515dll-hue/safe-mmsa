-- =============================================================================
-- SAFE-MMSA 关系型持久化 schema —— 与 fig4-5 的 E-R 一一对应（数据持久化层）。
--
-- 方言：SQLite 为主，尽量保持与 PostgreSQL 兼容（避免方言专属类型；JSON 以 TEXT
-- 存储，迁 PG 时可整列改 JSONB；REAL→DOUBLE PRECISION，INTEGER 布尔→BOOLEAN）。
-- 规范化：3NF——多值（各模态、各样本）拆子表并以 FK 回指，无重复组、无传递依赖。
-- 关系：user 1—N {analysis_record, eval_run, operation_log}；
--       analysis_record 1—N {modality_output, modality_reliability}，1—1 state_diagnosis；
--       eval_run 1—N eval_item。
-- =============================================================================

PRAGMA foreign_keys = ON;

-- ── user 用户 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user (
    user_id        TEXT    PRIMARY KEY,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    role           TEXT    NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at     REAL    NOT NULL
);

-- ── analysis_record 分析记录（N:1 → user）───────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_record (
    record_id            TEXT    PRIMARY KEY,
    user_id              TEXT    NOT NULL REFERENCES user (user_id),
    language             TEXT    NOT NULL,
    final_emotion        TEXT    NOT NULL,
    enhancement          TEXT    NOT NULL,
    correction_magnitude REAL    NOT NULL,
    bounded              INTEGER NOT NULL DEFAULT 0,   -- 布尔（迁 PG 改 BOOLEAN）
    initial_top          TEXT    NOT NULL,
    boundary_score       REAL    NOT NULL,
    created_at           REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_record_user ON analysis_record (user_id);
CREATE INDEX IF NOT EXISTS ix_record_created ON analysis_record (created_at);

-- ── modality_output 模态输出（N:1 → analysis_record）────────────────────────
CREATE TABLE IF NOT EXISTS modality_output (
    mo_id        TEXT    PRIMARY KEY,
    record_id    TEXT    NOT NULL REFERENCES analysis_record (record_id) ON DELETE CASCADE,
    modality     TEXT    NOT NULL,
    available    INTEGER NOT NULL,
    backend      TEXT    NOT NULL,
    distribution TEXT,                 -- JSON: {label: prob}
    logits       TEXT,                 -- JSON: [float]
    quality      TEXT,                 -- JSON: ModalitySignalQuality 摊平
    UNIQUE (record_id, modality)
);
CREATE INDEX IF NOT EXISTS ix_mo_record ON modality_output (record_id);

-- ── state_diagnosis 状态诊断（1:1 → analysis_record，UNIQUE FK 落实一对一）───
CREATE TABLE IF NOT EXISTS state_diagnosis (
    sd_id                       TEXT PRIMARY KEY,
    record_id                   TEXT NOT NULL UNIQUE
                                     REFERENCES analysis_record (record_id) ON DELETE CASCADE,
    boundary_max_confidence     REAL,
    boundary_top2_margin        REAL,
    boundary_norm_entropy       REAL,
    boundary_score              REAL,
    comp_top1                   TEXT,
    comp_top2                   TEXT,
    comp_margin                 REAL,
    comp_entropy                REAL,
    comp_vector                 TEXT,  -- JSON: [float]
    comp_per_modality_direction TEXT,  -- JSON: {modality: float}
    consistency_pairwise        TEXT   -- JSON: {"video-audio": float, ...}
);

-- ── modality_reliability 模态可靠性（N:1 → analysis_record）─────────────────
CREATE TABLE IF NOT EXISTS modality_reliability (
    mr_id            TEXT PRIMARY KEY,
    record_id        TEXT NOT NULL REFERENCES analysis_record (record_id) ON DELETE CASCADE,
    modality         TEXT NOT NULL,
    reliability      REAL NOT NULL,
    quality_part     REAL,
    uncertainty_part REAL,
    UNIQUE (record_id, modality)
);
CREATE INDEX IF NOT EXISTS ix_mr_record ON modality_reliability (record_id);

-- ── eval_run 评测运行（N:1 → user）──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_run (
    run_id                    TEXT    PRIMARY KEY,
    user_id                   TEXT    NOT NULL REFERENCES user (user_id),
    dataset                   TEXT,
    n_items                   INTEGER NOT NULL DEFAULT 0,
    macro_f1                  REAL,
    accuracy                  REAL,
    delta_macro_f1            REAL,
    delta_accuracy            REAL,
    mean_correction_magnitude REAL,
    confusion                 TEXT,    -- JSON: [[int]]
    created_at                REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_run_user ON eval_run (user_id);

-- ── eval_item 评测样本（N:1 → eval_run）─────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_item (
    item_id              TEXT PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES eval_run (run_id) ON DELETE CASCADE,
    gold_label           TEXT NOT NULL,
    video_path           TEXT,
    audio_path           TEXT,
    text                 TEXT,
    pred_initial         TEXT,
    pred_final           TEXT,
    correction_magnitude REAL
);
CREATE INDEX IF NOT EXISTS ix_item_run ON eval_item (run_id);

-- ── operation_log 操作日志（N:1 → user；系统事件 user_id 可空）──────────────
CREATE TABLE IF NOT EXISTS operation_log (
    log_id     TEXT PRIMARY KEY,
    user_id    TEXT REFERENCES user (user_id),
    action     TEXT NOT NULL,
    detail     TEXT,                   -- JSON
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_log_user ON operation_log (user_id);
CREATE INDEX IF NOT EXISTS ix_log_created ON operation_log (created_at);
