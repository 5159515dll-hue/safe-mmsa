# models/ — 训练产物挂载约定

本目录是**唯一**引入 torch / 真实权重的地方。核心包 `safemmsa/` 永不依赖 torch。
训练完成后，把模型放进来、实现 backend、在 config 里 enable，`load_models()` 一调用即生效，
**其余代码零改动**。

## 目录约定

```
models/
├─ __init__.py          # load_models(config) / unload_models()（已就绪）
├─ README.md            # 本文件
├─ config.example.json  # 配置样例
├─ weights/             # 放 checkpoint（.gitignore 忽略，不入库）
└─ backends/            # 各组件的真实实现：<component>.py 暴露 build(spec)->Backend
   ├─ encoder_text.py        # CINO
   ├─ encoder_audio.py       # wav2vec2-XLSR-53
   ├─ encoder_video.py       # ViT-FER
   ├─ fusion_backbone.py     # 跨模态注意力 + 低秩张量融合
   ├─ enhancement.py         # 有界残差增强网络 D
   └─ reliability_head.py    # 可选可学习可靠性融合头
```

每个 `backends/<component>.py` 暴露：

```python
def build(spec: dict) -> Backend:
    """spec 来自 config[component]；返回满足下方签名的可调用后端。"""
```

## 各组件必须满足的签名（= safemmsa 的稳定契约）

> 标签顺序固定 `EMOTION_LABELS = (happy, sad, angry, fear, surprise, disgust, neutral)`；
> 所有 logits / list[float] 一律按此顺序。

### 编码器 `encoder_text/video/audio`
```python
EncoderBackend = Callable[[Modality, RawInput, ModalitySignalQuality], ModalityOutput]
```
- 入：模态、原始输入（`RawInput.path` 指向媒体文件 / `RawInput.text` 文本）、已算好的信号质量。
- 出：`ModalityOutput(modality, available=True, distribution=<7类softmax>, logits=<7>,
  feature_ref=<高层特征缓存键>, quality=<透传>, backend="trained_<name>")`。
- `feature_ref` 指向一处特征缓存（供融合主干取 h），核心不搬运大张量。

### 融合主干 `fusion_backbone`
```python
FusionBackboneBackend = Callable[[list[ModalityOutput]], InitialPrediction]
```
- 出：`InitialPrediction(distribution=p⁰, logits=z⁰, fused_feature_ref=h⁰, backbone="trained_<name>")`。

### 增强网络 `enhancement`
```python
EnhancementBackend = Callable[[InitialPrediction, BoundaryState, CompetitionState, ReliabilityState], EnhancementResult]
```
- 只需产出 logit 空间残差 `raw_adjustment`(len=7)，再交给
  `safemmsa.apply_bounded_correction(p0, raw_adjustment, boundary.score, eta)` 做**由 B 门控的有界叠加**，
  保证 `‖p*−p⁰‖ ≤ eta·B`。返回 `EnhancementResult(..., enhancement="trained_<name>")`。

### 可靠性头 `reliability_head`（可选）
```python
ReliabilityHead = Callable[[dict quality_part, dict uncertainty_part, dict consistency_pairwise], dict per_modality]
```
- 出：每模态可靠性 ∈ [0,1]，覆盖默认确定性线性聚合。

## 配置

`config.example.json`：
```json
{
  "encoder_text":    {"enabled": true,  "weights": "models/weights/cino"},
  "encoder_audio":   {"enabled": false, "weights": "models/weights/xlsr53"},
  "encoder_video":   {"enabled": false, "weights": "models/weights/vit_fer"},
  "fusion_backbone": {"enabled": false, "weights": "models/weights/backbone.pt"},
  "enhancement":     {"enabled": false, "weights": "models/weights/enhance.pt"},
  "reliability_head":{"enabled": false}
}
```

挂载：
```python
from models import load_models
print(load_models("models/config.example.json"))
# 或 ENV：SAFEMMSA_MODELS_CONFIG=models/config.example.json
```
`enabled=false` 或 backend 未实现 → 保持桩/基线，系统照常端到端运行（不崩）。
`GET /api/models/status` 可随时确认当前是桩还是真模型。
