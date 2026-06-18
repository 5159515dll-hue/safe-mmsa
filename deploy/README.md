# 部署（与服务器上的 AIoT 工程隔离，使用独立端口）

SAFE-MMSA 用**全新端口**与 AIoT 并存，互不干扰：

| 组件 | 端口 | 绑定 | 说明 |
|---|---|---|---|
| FastAPI (uvicorn) | 8100 | 127.0.0.1 | AIoT 用 8000，互不冲突 |
| Next.js (next start) | 3100 | 127.0.0.1 | AIoT 用 3000，互不冲突 |
| nginx 对外反代 | **8080** | 0.0.0.0 | 独立 server 块，不碰 AIoT 的 :80 |

AIoT 占用：80(nginx)/3000(web)/8000(api)/1883(mqtt)/5432(pg)。SAFE-MMSA 不触碰其中任何一个，**无需停 AIoT**。

## 一键部署（在服务器上）

```bash
cd ~/safe-mmsa
# 1) Python 核心 + API
python3.12 -m venv .venv
.venv/bin/pip install -e ".[api,dev,preprocessing]"   # preprocessing 可选(opencv/librosa)
.venv/bin/python -m pytest -q                          # 自测

# 2) 前端构建
cd web && npm install && npm run build && cd ..

# 3) systemd 服务
sudo cp deploy/safe-mmsa-api.service /etc/systemd/system/
sudo cp deploy/safe-mmsa-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now safe-mmsa-api safe-mmsa-web

# 4) nginx 反代（新端口 8080，不动 AIoT）
sudo cp deploy/nginx-safe-mmsa.conf /etc/nginx/sites-available/safe-mmsa
sudo ln -sf /etc/nginx/sites-available/safe-mmsa /etc/nginx/sites-enabled/safe-mmsa
sudo nginx -t && sudo systemctl reload nginx
```

## 云安全组

需在腾讯云控制台**放行入站 TCP 8080**，否则外网不可达（与 AIoT 的 8000 同理：内部可用、外网默认不通）。
放行后访问：`http://<服务器IP>:8080/`

## 更新部署

```bash
# 本地 rsync 或服务器 git pull 后：
cd ~/safe-mmsa && .venv/bin/pip install -e ".[api,dev]" && cd web && npm install && npm run build && cd ..
sudo systemctl restart safe-mmsa-api safe-mmsa-web
```

## 排错

- `systemctl status safe-mmsa-api safe-mmsa-web`、`journalctl -u safe-mmsa-web -n 50`
- API 自检：`curl -s 127.0.0.1:8100/api/health`
- 模型状态：`curl -s 127.0.0.1:8100/api/models/status`（看是桩还是真模型）
