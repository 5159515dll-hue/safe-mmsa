# SAFE-MMSA 数据持久化层 —— 关系型 E-R 设计与落地

> 对应论文 **图 4-5**（数据持久化层·关系型数据库 E-R 设计）。本文件说明：8 个实体的
> 规范化设计、PK/FK 与 1-N/1-1 关系、3NF 论证，以及「JSON 现用 / RDBMS 后期落地、
> 共用同一套数据契约、平滑迁移」的工程路径。

---

## 1. 定位：一套契约，两种落地

系统有两类契约，分层清晰：

| 契约 | 文件 | 角色 |
|---|---|---|
| **compute 契约** | `safemmsa/contracts.py` | 系统稳定接口：编码器/主干/增强等训练产物满足签名即 `register_*` 零改动接入。 |
| **持久化契约** | `safemmsa/records.py` | 与 E-R 一一对应的 8 个规范化实体；**JSON 后端与 RDBMS 后端共用同一套**。 |

`AnalyzeResult`（compute 契约）经纯函数 `analyze_result_to_bundle()` 映射为规范化的
`AnalysisBundle`（持久化契约），再由仓储写入。**后端是可换的**：

```
api / eval  ──►  Repository(接口)  ──►  JsonRepository   (默认, 现在)
                                   └►  SqliteRepository  (db/schema.sql, 后期/可选)
```

调用方只依赖 `Repository` 接口，切换后端零改动 —— 这就是「平滑迁移」的落点。
后端由环境变量 `SAFEMMSA_DB_BACKEND`（`json` | `sqlite`）选择，SQLite 路径用
`SAFEMMSA_DB_PATH`（默认 `storage/safemmsa.db`）。

---

## 2. E-R 概览（实体与关系）

```
            ┌──────────┐
            │   user   │  PK user_id
            └────┬─────┘
       1—N  ┌────┼─────────────────┐  1—N
            ▼    ▼ 1—N             ▼
   analysis_record        eval_run            operation_log
   PK record_id           PK run_id           PK log_id
   FK user_id             FK user_id           FK user_id
        │                      │ 1—N
   ┌────┼───────────┐          ▼
   │1—N │1—1   │1—N        eval_item
   ▼    ▼      ▼           PK item_id  FK run_id
 modality  state_     modality_
 _output   diagnosis  reliability
```

| 关系 | 基数 | 落实方式 |
|---|---|---|
| user → analysis_record | 1—N | `analysis_record.user_id` FK |
| user → eval_run | 1—N | `eval_run.user_id` FK |
| user → operation_log | 1—N | `operation_log.user_id` FK（系统事件可空） |
| analysis_record → modality_output | 1—N | `modality_output.record_id` FK + `UNIQUE(record_id, modality)` |
| analysis_record → state_diagnosis | **1—1** | `state_diagnosis.record_id` FK **UNIQUE** |
| analysis_record → modality_reliability | 1—N | `modality_reliability.record_id` FK + `UNIQUE(record_id, modality)` |
| eval_run → eval_item | 1—N | `eval_item.run_id` FK |

子表对父表的 FK 均带 `ON DELETE CASCADE`，删除一条分析/评测即级联清理其明细。

---

## 3. 表逐一说明

### 3.1 `user` 用户
`user_id`(PK) · `username`(UNIQUE) · `password_hash` · `role`(`user`|`admin`) · `created_at`。
口令以 `pbkdf2_sha256$rounds$salt$hash` 存（`safemmsa/auth.py`，标准库，无新依赖）。
未登录的分析归属种子用户 `anonymous`（`usr_anonymous`），保证 FK 始终有效。

### 3.2 `analysis_record` 分析记录
`record_id`(PK) · `user_id`(FK) · `language` · `final_emotion` · `enhancement` ·
`correction_magnitude` · `bounded` · `initial_top` · `boundary_score` · `created_at`。
单值的端到端结论；多值部分拆到下面三张子表。

### 3.3 `modality_output` 模态输出（每记录最多 3 行）
`mo_id`(PK) · `record_id`(FK) · `modality` · `available` · `backend` ·
`distribution`(JSON) · `logits`(JSON) · `quality`(JSON)。`UNIQUE(record_id, modality)`。

### 3.4 `state_diagnosis` 状态诊断（每记录恰 1 行）
`sd_id`(PK) · `record_id`(FK,UNIQUE) · 边界 B（max_confidence/top2_margin/norm_entropy/score）·
类别竞争 C（top1/top2/margin/entropy/vector/per_modality_direction）· 跨模态一致性。
三状态里的**记录级**诊断集中于此；`record_id` 上的 UNIQUE 正是 1—1 的关系约束。

### 3.5 `modality_reliability` 模态可靠性（每模态 1 行）
`mr_id`(PK) · `record_id`(FK) · `modality` · `reliability` · `quality_part` ·
`uncertainty_part`。R 状态里**逐模态**的可靠性分量。`UNIQUE(record_id, modality)`。

### 3.6 `eval_run` 评测运行
`run_id`(PK) · `user_id`(FK) · `dataset` · `n_items` · `macro_f1` · `accuracy` ·
`delta_macro_f1` · `delta_accuracy` · `mean_correction_magnitude` · `confusion`(JSON) · `created_at`。

### 3.7 `eval_item` 评测样本（每 run N 行）
`item_id`(PK) · `run_id`(FK) · `gold_label` · `video_path` · `audio_path` · `text` ·
`pred_initial` · `pred_final` · `correction_magnitude`。

### 3.8 `operation_log` 操作日志
`log_id`(PK) · `user_id`(FK,可空) · `action`（register/login/analyze/…）· `detail`(JSON) · `created_at`。

---

## 4. 规范化（3NF）论证

- **1NF**：所有列原子。多值的「各模态输出 / 各模态可靠性 / 各评测样本」不塞进数组列，
  而是各自成表、以 FK 回指父行 —— 消除重复组。
- **2NF**：每张表均为单列代理主键（`*_id`），不存在「非键列只依赖复合键一部分」的部分依赖。
  `modality_output` 等用 `UNIQUE(record_id, modality)` 作业务键约束，主键仍是单列 `mo_id`。
- **3NF**：非键列只依赖本表主键，无传递依赖。逐模态可靠性依赖 `(record_id, modality)`
  而非 `analysis_record` 的其他属性，故独立成 `modality_reliability` 表；记录级的
  状态诊断依赖 `record_id`、放 `state_diagnosis`。
- **受控冗余**：`analysis_record.boundary_score`、`initial_top` 是为列表/排序便利冗余的派生值，
  其权威来源仍是 `state_diagnosis` / `modality_output`，不破坏 3NF 的判断（派生而非传递依赖）。
- JSON 列（`distribution`/`logits`/`quality`/`confusion`/`comp_vector`/`detail`）保存的是
  **对分析无需按元素查询**的向量/字典，作为整体读写；如后期需按元素检索，迁 PG 改 `JSONB` 即可加索引。

---

## 5. 后期落地路径（→ RDBMS）

1. `db/schema.sql` 已是同一 schema 的规范化 DDL（SQLite 为主、PG 兼容）。
2. 置 `SAFEMMSA_DB_BACKEND=sqlite`（可选 `SAFEMMSA_DB_PATH`），`get_repository()` 即返回
   `SqliteRepository`，建表、写读全部走库；**API / eval / 前端零改动**。
3. 迁 PostgreSQL：DDL 按注释替换少量类型（`REAL→DOUBLE PRECISION`、布尔 `INTEGER→BOOLEAN`、
   JSON 列 `TEXT→JSONB`），新增一个 `PgRepository` 实现同一 `Repository` 接口即可，其余不动。
4. 历史 JSON 数据迁移：逐行读 `storage/*.json` → 经持久化契约 `model_validate` → 调
   `save_*` 写入新库（同一契约，转换无损）。

---

## 6. 对外接口（已接入）

- `POST /api/auth/register` · `POST /api/auth/login` · `GET /api/auth/me`
- `POST /api/analyze`（带 `Authorization: Bearer` 则归属该用户，否则 anonymous）→ 落规范化 bundle + 操作日志
- `GET /api/history` / `GET /api/history/{record_id}`（按用户隔离）
- `GET /api/admin/logs`（仅 `admin`）
- `POST /api/eval/run` 同时落 `eval_run` / `eval_item`
