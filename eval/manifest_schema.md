# 评测清单格式（manifest）

批量评测的输入是一个 **JSONL** 文件（每行一个 JSON 对象），每行描述一条样本：

```json
{"id": "s001", "label": "happy", "video_path": "clips/s001.mp4", "audio_path": "clips/s001.wav", "text": "ᠰᠠᠶᠢᠨ ᠪᠠᠶᠢᠨ᠎ᠠ ᠤᠤ"}
```

## 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 样本唯一标识（字符串） |
| `label` | 是 | 真值情感，取值 ∈ 7 类之一：`happy` `sad` `angry` `fear` `surprise` `disgust` `neutral` |
| `video_path` | 否 | 视频文件路径；相对路径按**清单文件所在目录**解析 |
| `audio_path` | 否 | 音频文件路径；同上 |
| `text` | 否 | 蒙古文（或任意）文本，直接进文本框 |

- 三个模态字段**至少给一个**；缺的模态在分析时按 unavailable 降级，不影响其余模态。
- `label` 必须是 7 类英文标签之一；非法标签的样本在指标统计中被忽略。

## 用法

```python
from eval import load_manifest, run_manifest
from eval.report import export

items = load_manifest("data/dev.jsonl")
result = run_manifest(items, collect_per_item=True)
export(result, outdir="runs")          # 写 JSON/CSV/混淆矩阵图
print(result.paired.final.macro_f1)
```

或经 API：`POST /api/eval/run`（上传清单或传服务器端路径）→ `GET /api/eval/{run_id}`。

## 指标

对 **initial（融合主干基线）** 与 **final（增强后）** 分别给 accuracy / macro-F1 / 混淆矩阵，
并报告增量 `delta_macro_f1` / `delta_accuracy`、平均修正幅度、预测改变样本数。
默认桩/identity 下 final == initial（delta≈0），框架就绪；真模型注册后产生差异。
