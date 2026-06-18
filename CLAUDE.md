# CLAUDE.md — SAFE-MMSA 项目交接

> 给在本仓库工作的 Claude 会话的上下文。系统全貌见 [README.md](README.md) 与
> [docs/system-design.md](docs/system-design.md)（权威方案，12 模块）。

## 是什么
**SAFE-MMSA** = 状态感知驱动融合增强的**蒙古语多模态情感分析**（毕设）。无硬件，三模态
（视频+音频+蒙文文本）文件上传，**predict-then-correct**：编码器 → 融合主干 → 初始预测 p⁰ →
**三状态（边界 B / 类别竞争 C / 模态可靠性 R）** → 有界融合增强 → 最终 p*。7 类离散情感。

## 完成度
除可训练神经模型外**12 模块全部完成**，53 单测全过，已上线。模型藏在 `register_*` 之后，
现给桩/基线；核心包 `safemmsa/` **不依赖 torch**（torch 只在 `models/` 挂载真实模型时进入）。
训练完成后按 [models/README.md](models/README.md) 实现 `models/backends/<component>.py` 的
`build(spec)`、config 设 `enabled=true`、`load_models()` 一调即生效，**其余代码零改动**。

## 仓库 / 推送
- 私有 repo `https://github.com/5159515dll-hue/safe-mmsa.git`（user 自己的 `5159515dll-hue`）。
- **分模块/分批次提交**，commit 末尾带 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **保密红线**：开题报告（`~/Desktop/开题报告.docx`）绝不入库/外发/复制原文进任何会 push 的文件；
  `data/` 与训练权重已 `.gitignore`。

## 本地开发
```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[api,dev]"
pytest -q                       # 53 passed
```
（本地不装 node 依赖；前端构建在服务器上做。）

## 服务器部署（与服务器上的 AIoT 工程完全隔离，独立端口，无需停 AIoT）
- SSH：`ssh gpu-relay`（密钥登录；服务器 `~/safe-mmsa/`，venv `.venv` python3.12，无 GitHub 凭据→用 rsync 同步，别在服务器 git pull）。
- 端口：FastAPI `127.0.0.1:8100`、Next.js `127.0.0.1:3100`、对外 nginx **8080**（公网已放行）。
- systemd：`safe-mmsa-api`、`safe-mmsa-web`；nginx site `safe-mmsa`。部署细节见 [deploy/README.md](deploy/README.md)。
- 公网：**http://82.157.148.249:8080/**

**改完发布流程**（`node_modules`/`.next` 仅在服务器）：
```bash
unset HTTP_PROXY HTTPS_PROXY
rsync -az --exclude '.venv' --exclude 'node_modules' --exclude '.git' --exclude '.next' \
  --exclude 'storage' --exclude 'data' --exclude '__pycache__' -e ssh ./ gpu-relay:/home/ubuntu/safe-mmsa/
ssh gpu-relay 'cd ~/safe-mmsa/web && npm run build'          # 前端
ssh gpu-relay 'sudo systemctl restart safe-mmsa-web safe-mmsa-api'
curl --noproxy '*' http://82.157.148.249:8080/api/health     # 验证
# 然后本地 git commit + git push origin main
```

## 前端（web/，Next.js 14，`next@14.2.35`）
已自带 "SAFE-MMSA Design System"：IBM Plex + Noto Sans Mongolian 字体、color/spacing/motion
tokens、7 情感专属配色 `--emo-*`；组件 `Bars`/`ModalityCard`/`StatesPanel`/`AnalyzingPipeline`/`FileDrop`。
一屏可视化「模态+质量 → 初始 p⁰ → 三状态 → 最终 p*+修正幅度」。

## 待办：Claude Design 导入
把设计项目 `https://claude.ai/design/p/8f37096b-8dfa-4d4b-9d5d-6bf38834125e` 经
**DesignSync 连接器 + `/design-sync` skill** 导入实现到 `web/`。
**阻塞**：需在交互式 `claude` 终端 `/login` 登录拥有该设计项目的 claude.ai 账号拿设计 scope
（`CLAUDE_CODE_OAUTH_TOKEN` 无法扩展该 scope）。拿到后 `get_project`→`list_files`→`get_file`，
与既有 token 体系**增量逐组件对齐**，别整体覆盖；动手前先确认是“重塑现有 UI”还是“新增页面”。
