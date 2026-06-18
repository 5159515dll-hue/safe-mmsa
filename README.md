# SAFE-MMSA

**基于状态感知驱动融合增强的蒙古语多模态情感分析系统研发**
State-Aware (Fusion-Enhanced) Mongolian Multi-modal Sentiment Analysis

对【视频 + 音频 + 蒙古文文本】做 7 类离散情感分类（厌恶/悲伤/愤怒/恐惧/快乐/惊讶/中性）。
无硬件，三模态经文件上传 + 文本框进入；含批量数据集评测。

实现 **“先预测 → 再修正”（predict-then-correct）** 框架：
模态编码器 → 融合主干 → 初始预测 p⁰ → **三状态（边界 B / 类别竞争 C / 模态可靠性 R）**
→ **有界融合增强** → 最终预测 p*。三状态为确定性计算（论文核心创新，本期已实现）；
编码器/主干/增强网络为可训练插件，本期用桩/基线让系统端到端先跑通，模型训练完成后经
`register_*` 直接挂载、**其余代码零改动**。

> 完整实施蓝图见 [`docs/system-design.md`](docs/system-design.md)。
> 本期目标：把“除可训练模型外的所有模块”做完并定义稳定接口。**已全部完成。**

## 本期完成度（除模型外全部就绪）

| # | 模块 | 位置 | 状态 |
|---|---|---|---|
| 1 | 契约 / 标签 / 语言 | `safemmsa/{contracts,labels,language}.py` | ✅ |
| 2 | 预处理与信号质量 | `safemmsa/preprocessing/` | ✅（opencv/librosa 可选，缺失降级） |
| 3 | **三状态（确定性创新点）** | `safemmsa/states.py` | ✅ 高覆盖单测 |
| 4–6 | 编码器/主干/增强 注册表 + 桩/基线 + 有界修正 | `safemmsa/{encoders,backbone,enhancement}.py` | ✅ 插点就绪 |
| 7 | 编排器（全程容错降级） | `safemmsa/orchestrator.py` | ✅ |
| 8 | API（FastAPI） | `api/` | ✅ |
| 9 | 数据集适配 + 批量评测台 | `eval/` | ✅ |
| 10 | 演示前端（Next.js） | `web/` | ✅ |
| 11 | 结果/历史持久化 | `safemmsa/persistence.py` | ✅ |
| 12 | 模型挂载约定 | `models/` | ✅ 约定+桩（权重待训练） |

可训练神经模型（CINO / wav2vec2-XLSR-53 / ViT-FER / 融合主干 / 增强网络 D / 可选可靠性头）
**本期推迟**，全部藏在 `register_*` 之后——训练完成后注册即生效。

## 快速开始（核心，不依赖 torch）

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[api,dev]"        # 核心 numpy+pydantic；api=FastAPI；dev=pytest
pytest -q                          # 53 个单测

python -c "import safemmsa, json; print(safemmsa.analyze(text='I am so happy and great').final_emotion)"
```

启动 API + 前端：

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8100      # 后端
cd web && npm install && npm run build && npm run start # 前端 :3100
```

- `POST /api/analyze`（multipart：video/audio 文件可选 + text 字段可选）→ 完整 `AnalyzeResult`
- `GET /api/models/status` —— 一眼看出当前是桩/基线还是真模型
- `POST /api/eval/run` + `GET /api/eval/{id}` —— 批量评测（macro-F1/accuracy/混淆矩阵，initial vs final）

## 仓库结构

```
safemmsa/        # 纯 Python 核心（不依赖 torch）：契约/质量/三状态/编排/桩/基线
api/             # FastAPI：/api/analyze, /api/models/status, /api/history, /api/eval/*
eval/            # 评测台：清单→runner→metrics→ablation/robustness→report
models/          # 训练产物挂载约定（load_models）+ README（签名/权重放置）—— 本期桩
web/             # Next.js 可视化演示页（初始→三状态→最终+修正幅度）
deploy/          # systemd + nginx（独立端口，与服务器上的 AIoT 工程隔离）
tests/           # 53 个单测
docs/            # 系统设计方案
```

## 部署

与既有工程隔离，使用独立端口（API 8100 / Web 3100 / 对外 nginx 8080）。
详见 [`deploy/README.md`](deploy/README.md)。

## 训练完成后如何接入真实模型

1. 把 checkpoint 放进 `models/weights/`；
2. 在 `models/backends/<component>.py` 实现 `build(spec)`（签名见 [`models/README.md`](models/README.md)）；
3. 在 config 里把对应组件 `enabled=true`；`load_models()` 一调用即生效。

`GET /api/models/status` 随时确认当前桩/真模型；**核心代码零改动**。
