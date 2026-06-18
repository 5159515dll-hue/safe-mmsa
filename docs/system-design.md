# SAFE-MMSA 系统设计方案（具体实施蓝图）

> 本文档是“从 AIoT Copilot 抽取情感核心 + 新增内容，做一个独立的蒙古语多模态情感分析系统”的**具体工程方案**，供新会话直接照此实现。
> 本机本地文档，不外发、不入任何远程仓库；不复制开题报告原文。

---

## 0. 题目 / 定位

- **系统中文名**：基于状态感知驱动融合增强的蒙古语多模态情感分析系统研发
  - 「状态感知」与「融合增强」融为一体：三类样本级状态(边界/竞争/可靠性)**驱动**有界融合增强（决定是否修正/幅度/方向/取信哪个模态），与开题报告「状态感知的融合增强方法」一致。
- **代号 / 仓库名**：**SAFE-MMSA** = **S**tate-**A**ware **F**usion-**E**nhanced **M**ongolian **M**ulti-modal **S**entiment **A**nalysis
  - `SAFE` 对应论文两个创新点：状态感知 **(State-Aware)** 驱动 融合增强 **(Fusion-Enhanced)**。
- **任务**：对【视频 + 音频 + 蒙古文文本】联合输入做 **7 类离散情感分类**（厌恶/悲伤/愤怒/恐惧/快乐/惊讶/中性），低资源蒙古语。
- **形态**：独立的 Web 系统，**无硬件**；三模态通过**文件上传 + 文本框**进入；含**批量数据集评测**入口。
- **本期目标**：**把除可训练神经模型以外的所有模块做完**，并定义好稳定接口，使模型训练完成后通过 `register_*` 直接挂载、其余代码零改动。

---

## 1. 核心架构：predict-then-correct（与论文 图4-1 对齐）

```
原始输入(视频?/音频?/蒙文文本?)
   │  ── 上传 + 预处理 + 信号质量度量(确定性, 现在就做)
   ▼
[模态编码器 ×3]  CINO(文) / wav2vec2-XLSR-53(声) / ViT-FER(视)   ← 可训练, 现在用桩
   │  每模态: 特征 f + 单模态head logits + 7类分布 + 质量指标
   ▼
[融合主干]  跨模态注意力 + 低秩张量融合                          ← 可训练, 现在用“晚融合基线”
   │  初始预测 z⁰/p⁰ + 融合表征 h⁰  （initial prediction）
   ▼
[状态感知 ×3]  边界评分B / 类别竞争C / 模态可靠性R               ← **确定性, 现在全部做掉**
   │
   ▼
[融合增强网络 D]  以 p⁰ 为锚, 受 B 门控的有界残差修正           ← 可训练, 现在用 identity
   │  特征层低秩张量交互项 + 决策层有界调整量, 叠加于 p⁰
   ▼
最终预测 p*  →  7 类情感
```

**关键工程洞察（决定本方案可行性）**：
1. **三个“状态”几乎都是确定性计算**，不需要训练，**现在就能完整实现**，而且就是论文的核心创新：
   - **边界评分 B**：由 p⁰ 的「最大置信度 / Top-2 间隔 / 归一化熵」三统计量加权、squash 到 [0,1]。纯数学。
   - **类别竞争状态 C**：取 p⁰ 的 Top-2 类别，算各模态单模态 logits 在这对类别上的方向差，拼成状态向量。纯数学。
   - **模态可靠性 R**：质量(信号级) + 单模态不确定性 + 跨模态一致性 三层 → 每模态可靠性∈[0,1]。质量是信号处理；不确定性/一致性是 logits 上的数学。默认确定性聚合即可跑；其中“每模态可靠性融合映射”那一小段可学习头，留 `register_reliability_head` 后插。
2. **真正要训练的只有 4 类东西**，全部藏在 `register_*` 插点后面，**现在给默认桩/基线**让系统端到端先跑起来：
   - 3 个模态编码器、融合主干、增强网络 D、(可选)可靠性可学习头。
3. **系统从第一天起就完整可观测**：即便编码器是桩、增强是 identity，编排器仍会算出并展示「初始预测→三状态→最终预测+修正幅度」。模型训练好注册进来后，**同一条管线**自然产出非零修正——这正是答辩/论文配图最想要的。
4. **抽出来的「晚融合」既是默认主干，又恰好是论文消融实验的 baseline**——直接复用，一举两得。

---

## 2. 从 AIoT Copilot 抽什么 / 怎么改 / 丢什么

### 抽取并改造（来源 → 去向）
| AIoT 源文件 | 复用内容 | 改造 |
|---|---|---|
| `services/api/app/emotion_perception.py` | **`register_face/voice/text_model` 插件机制**（正是“模型后插”的范式）、`detect_language`（已能识别蒙文/西里尔）、7 类标签、关键词桩 | 扩展为 `encoders.py`：插点改为“编码器后端”，输出从“仅分布”升级为“特征+单模态logits+分布+质量” |
| `services/api/app/emotion_fusion.py` | **`fuse_distribution` 晚融合 = 默认主干 + 消融 baseline**；`EmotionSmoother` 滞回（留给实时视图）；valence/arousal 映射 | 包成 `backbone.py` 的默认 `late_weighted_baseline`，输出对齐 `InitialPrediction(z⁰,p⁰,h⁰)` |
| `services/api/app/models.py`（情感部分）| `EmotionModalityInput/Summary/State` 等 | 迁入 `contracts.py` 并**扩展**（见 §3） |
| `services/api/app/routes/emotion.py` | ingest/state 路由骨架 | **重写**为文件上传 `POST /api/analyze` |
| `apps/web/app/emotion/page.tsx` + `components/emotion-timeline-panel.tsx` + `lib/{api,types}.ts` 情感部分 | 上传/展示前端基础 | 改造为“上传三模态 + 可视化整条框架”的演示页（见 §4 模块10） |

### 丢弃（不进新系统）
`robots/`、`firmware/`、`services/mqtt-ingestor/`、`device_*`（连接/适配/凭据/限流）、`companion_*`（手势/TTS/人格/记忆/语音流）、`policy.py`、`live_stream.py`、`media_store`、节点/遥测/空间/规则/审计/评测台等全部 IoT 与陪伴功能。

### 形态建议
- **后端核心 = 纯 Python 包 `safemmsa/`（不依赖 torch）**：契约、预处理质量、三状态、编排、桩/基线全部 numpy/纯 python → “除模型外的模块”可独立运行、可单测、轻量。**torch 只在 `models/` 真模型挂载时才进来。**
- **API = FastAPI**（与 AIoT 一致，迁移顺滑）。
- **前端**：**Next.js（已定）**，复用 AIoT 现成的上传/展示组件改造。

---

## 3. 核心数据契约（让模型“后插即用”的关键）

`safemmsa/contracts.py`（pydantic v2）。这是整套系统的**稳定接口**，训练产物只要满足这些签名即可零改动接入。

```python
EMOTION_LABELS = ("happy","sad","angry","fear","surprise","disgust","neutral")  # 固定 7 类
Modality = Literal["video","audio","text"]

class ModalitySignalQuality(BaseModel):
    # 信号级、与情感弱相关、不依赖深度模型 —— 现在就能算（缺测项留 None）
    face_detection_rate: float | None = None   # 视频
    blur: float | None = None
    brightness: float | None = None
    frame_loss_rate: float | None = None
    snr_db: float | None = None                # 音频
    voiced_ratio: float | None = None
    silence_ratio: float | None = None
    text_length: int | None = None             # 文本
    valid_token_ratio: float | None = None
    low_freq_subword_ratio: float | None = None

class ModalityOutput(BaseModel):
    modality: Modality
    available: bool
    distribution: dict[str, float] | None      # 单模态 head softmax（7 类）
    logits: list[float] | None                 # 单模态 logits（状态计算需要）
    feature_ref: str | None = None             # 高层特征的缓存键/引用（融合主干用；桩为 None）
    quality: ModalitySignalQuality

class InitialPrediction(BaseModel):
    distribution: dict[str, float]             # p⁰
    logits: list[float]                        # z⁰
    fused_feature_ref: str | None = None       # h⁰ 引用
    backbone: str                              # "late_weighted_baseline" | "trained_<name>"

class BoundaryState(BaseModel):
    max_confidence: float
    top2_margin: float
    norm_entropy: float
    score: float                               # B ∈ [0,1]，门控修正幅度

class CompetitionState(BaseModel):
    top1: str
    top2: str
    per_modality_direction: dict[str, float]   # 各模态对(top1,top2)方向支持；>0 偏向 top1
    margin: float
    entropy: float
    vector: list[float]                        # C 状态向量（供 D 输入）

class ReliabilityState(BaseModel):
    per_modality: dict[str, float]             # R：每模态可靠性 ∈ [0,1]
    quality_part: dict[str, float]
    uncertainty_part: dict[str, float]
    consistency_pairwise: dict[str, float]     # {"video-audio":.., "video-text":.., "audio-text":..}

class EnhancementResult(BaseModel):
    final_distribution: dict[str, float]       # p*
    final_logits: list[float]
    correction_magnitude: float                # ‖p*−p⁰‖，受 B 上界约束
    bounded: bool
    enhancement: str                           # "identity_baseline" | "trained_<name>"

class AnalyzeResult(BaseModel):
    language: str                              # 固定 "mn"（系统定位蒙古语）
    modalities: list[ModalityOutput]
    initial: InitialPrediction
    boundary: BoundaryState
    competition: CompetitionState
    reliability: ReliabilityState
    enhancement: EnhancementResult
    final_emotion: str
    timings_ms: dict[str, float] | None = None
```

---

## 4. 模块清单：本期“除模型外”全部要做（含接口 + 默认桩 + 模型插点）

> 标 **[现在做]** = 本期实现；**[插点]** = 现在给默认实现 + 留 `register_*`，训练后替换。

**模块1 — 契约与标签** `safemmsa/contracts.py` `labels.py` `language.py` **[现在做]**
迁移并扩展 §3；`detect_language` 直接复用（已识别蒙文）。系统对外固定 `language="mn"`。

**模块2 — 预处理与信号质量** `safemmsa/preprocessing/{video,audio,text}_quality.py` **[现在做]**
- 视频：解码取帧（opencv）→ 亮度/模糊(Laplacian 方差)/帧缺失率；人脸检测率用轻量检测器(opencv haar 或 mediapipe，**注意这是信号级检测，非 FER 情感模型**)，无依赖时可桩为 None。
- 音频：librosa 载入 → SNR 估计、有声帧比例、静音比例（可选 webrtcvad）。
- 文本：长度、有效 token 比例、低频子词比例（用 CINO tokenizer；无 transformers 时退化为简单分词桩）。
- 产出 `ModalitySignalQuality`。**纯信号处理，不依赖情感模型。**

**模块3 — 三状态计算（论文创新点核心）** `safemmsa/states.py` **[现在做]**
确定性函数，按论文公式实现，配高覆盖单测：
```python
def compute_boundary_state(p0: dict[str,float], weights=(1/3,1/3,1/3)) -> BoundaryState
    # max_conf = max(p0); top2_margin = p1 - p2; norm_entropy = H(p0)/ln(K)
    # score = clip( w1*(1-max_conf) + w2*(1-margin) + w3*norm_entropy , 0, 1 )
def compute_competition_state(p0, unimodal_logits: dict[str,list[float]]) -> CompetitionState
    # top1,top2 = p0 的前二；每模态 dir_m = logit_m[top1] - logit_m[top2]
    # vector = [dir_video,dir_audio,dir_text, top2_margin, entropy]
def compute_reliability_state(qualities, unimodal_outputs) -> ReliabilityState
    # quality_part: 质量指标归一聚合; uncertainty_part: 单模态 conf/熵/top2间隔;
    # consistency_pairwise: 模态对在竞争类别对上的方向一致性;
    # per_modality = sigmoid(默认线性聚合)  ← 可学习头见模块6插点
```

**模块4 — 模态编码器注册表** `safemmsa/encoders.py` **[插点]**
```python
EncoderBackend = Callable[[Modality, RawInput, ModalitySignalQuality], ModalityOutput]
def register_encoder(modality: Modality, backend: EncoderBackend | None) -> None
```
默认桩：文本走关键词/CINO-tokenizer 桩分布，视/声返回 `available + quality` 但 `distribution=None/uniform、feature_ref=None`。训练后注册 CINO / wav2vec2-XLSR-53 / ViT-FER 真实编码器（输出特征 + 单模态 logits + 分布）。

**模块5 — 融合主干注册表** `safemmsa/backbone.py` **[插点]**
```python
FusionBackboneBackend = Callable[[list[ModalityOutput]], InitialPrediction]
def register_fusion_backbone(backend) -> None
```
默认 `late_weighted_baseline`：复用 AIoT `fuse_distribution`（置信度×模态先验加权）→ p⁰，z⁰=log(p⁰)，`backbone="late_weighted_baseline"`。训练后注册“跨模态注意力 + 低秩张量融合”主干，输出 h⁰/z⁰/p⁰。

**模块6 — 融合增强注册表 + 有界修正数学** `safemmsa/enhancement.py` **[插点]**
```python
EnhancementBackend = Callable[[InitialPrediction, BoundaryState, CompetitionState, ReliabilityState], EnhancementResult]
def register_enhancement(backend) -> None
def apply_bounded_correction(p0, raw_adjustment, boundary_score, eta) -> (p_final, magnitude)
    # 残差叠加 + 由 B 决定的幅度上界：B≈0 → 几乎不改; B≈1 → 允许较大修正
def register_reliability_head(head) -> None   # 可靠性可学习融合映射(可选)
```
默认 `identity_baseline`：p*=p⁰、correction=0、`enhancement="identity_baseline"`。`apply_bounded_correction` **现在就实现**（确定性数学），真增强网 D 训练后只产出 raw_adjustment 交给它做有界叠加。

**模块7 — 编排器** `safemmsa/orchestrator.py` **[现在做]**
`analyze(video?, audio?, text?) -> AnalyzeResult`：预处理→质量→编码器(3)→主干→三状态→增强→最终。**全程容错降级**（某模态缺失/低质，其余继续；沿用 AIoT 的“模态 unavailable 不影响其他”）。

**模块8 — API** `api/main.py` **[现在做]**
- `POST /api/analyze`：multipart（video/audio 文件可选 + text 字段可选）→ `AnalyzeResult`。
- `GET /api/health`；`GET /api/models/status`：报告各插点是“桩/基线”还是“真模型”（一眼看出系统当前用真模型与否）。
- `POST /api/eval/run`、`GET /api/eval/{id}`：批量评测（见模块9）。

**模块9 — 数据集适配 + 批量评测台** `eval/` **[现在做]**
- **清单格式**（`eval/manifest_schema.md`）：JSONL，每行 `{"id","label","video_path?","audio_path?","text?"}`；标签为 7 类之一。
- `runner.py`：遍历清单→`analyze`→收集 initial 与 final。
- `metrics.py`：**macro-F1、accuracy、混淆矩阵**；分别对 initial(主干基线) 与 final(增强后) 出指标。
- `ablation.py`：开关三状态/增强模块跑对照（默认 identity 时 final==initial，框架就绪，真模型到位即生效）。
- `robustness.py`：音/视频加噪、模态缺失注入，记录退化与恢复指标。
- `report.py`：导出 JSON/CSV + 混淆矩阵图（论文可直接用）。

**模块10 — 演示前端** `web/` **[现在做]**
上传【视频/音频/蒙文文本】→ 一屏可视化整条框架：各模态分布+质量条 → 初始预测 p⁰ → **三状态面板**（边界评分仪表、竞争类别对、可靠性条形）→ 最终预测 p* 与**修正幅度**。用 **Next.js** 复用 AIoT 上传/展示组件改造。

**模块11 — 结果/历史持久化** `safemmsa/persistence.py` **[现在做]**
分析历史 + 评测 run 落 SQLite 或 JSON（沿用 AIoT JsonListStore 思路）。

**模块12 — 模型挂载约定** `models/` **[插点 + 现在写约定]**
`models/__init__.py` 暴露 `load_models(config)`：按配置 `register_encoder/fusion_backbone/enhancement/reliability_head`。默认 no-op（全用桩/基线）。ENV/config 切换 stub↔real。`models/README.md` 写清训练产物需满足的签名与权重放置路径 → 训练完把 checkpoint 丢进来、实现 backend、`load_models` 一注册即生效。

---

## 5. 仓库结构（目标）

```
safe-mmsa/
├─ README.md
├─ docs/system-design.md            ← 本文档
├─ pyproject.toml                    # 核心不含 torch；preprocessing 依赖 opencv/librosa
├─ safemmsa/                         # 纯 Python 核心（不依赖 torch）
│  ├─ labels.py  language.py  contracts.py
│  ├─ preprocessing/{video_quality,audio_quality,text_quality}.py
│  ├─ encoders.py   backbone.py   states.py   enhancement.py
│  ├─ orchestrator.py   persistence.py
├─ api/{main.py, eval_routes.py}     # FastAPI
├─ eval/{manifest_schema.md, runner.py, metrics.py, ablation.py, robustness.py, report.py}
├─ models/{__init__.py(load_models), README.md}   # ← 训练产物挂载点（本期只写约定+桩）
├─ web/                              # Next.js(复用) 或 Gradio
├─ tests/{test_states.py, test_contracts.py, test_orchestrator_stub.py, test_quality.py}
└─ data/                            # gitignore；数据集放这
```

---

## 6. 实施顺序（新会话按此推进）

1. **基础**：`labels` + `language` + `contracts`（从 AIoT 迁移情感模型并扩展）。
2. **质量**：`preprocessing/*` 三个质量提取器 + 单测。
3. **状态**：`states.py` 三状态（确定性，**论文创新点**）+ 高覆盖单测（对照论文公式逐项验证）。
4. **插点**：`encoders/backbone/enhancement` 注册表 + 默认桩/晚融合基线 + `apply_bounded_correction`。
5. **编排**：`orchestrator.analyze` + 端到端桩测（上传→完整 `AnalyzeResult`，全部走桩也能跑通）。
6. **API**：`/api/analyze` + `/api/models/status`。
7. **前端**：上传 + 框架可视化页。
8. **评测台**：清单→runner→metrics→ablation/robustness→report。
9. **模型约定**：`models/load_models` + README（为训练产物就位）。
10. **持久化 + 打磨**。

> 1–5 步做完，系统已“除模型外全部就绪”、端到端可跑可视化；6–10 补齐 API/UI/评测/挂载约定。

---

## 7. 技术选型 / 依赖

- Python 3.11；FastAPI + pydantic v2 + numpy（核心）。
- 预处理：opencv-python、librosa、(可选)webrtcvad、(可选)transformers 仅用 CINO **tokenizer**（不是模型推理）。
- 前端：**Next.js**（复用 AIoT 上传/展示组件）。
- **核心包不引入 torch**；torch/真实编码器仅在 `models/` 挂载阶段进入。
- 不建 git remote、不 push（本地研究项目，避免泄漏）。

## 8. 明确推迟（= 你说的“先不训练”的部分）

- 3 个模态编码器（CINO / wav2vec2-XLSR-53 / ViT-FER）的真实推理权重。
- 融合主干（跨模态注意力 + 低秩张量融合）训练。
- 增强网络 D（有界残差修正）训练。
- 可靠性可学习融合头（可选）。

以上全部藏在 `register_*` 之后；本期交付的系统用桩/基线端到端运行，训练完成后逐个注册即生效，**其余模块零改动**。
```
