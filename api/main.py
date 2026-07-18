"""模块8 — FastAPI 应用：文件上传 → 端到端分析 + 健康检查 + 模型状态（+ 评测路由）。

后端只绑 127.0.0.1（部署时由 nginx 反代）。三模态经 multipart 进入：
video/audio 作为文件上传落临时文件，text 作为表单字段。
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import safemmsa
from safemmsa import auth
from safemmsa.contracts import AnalyzeResult
from safemmsa.records import OperationLog, analyze_result_to_bundle
from safemmsa.repository import get_repository

from .auth_routes import resolve_user_id, router as auth_router
from .models_status import models_status

_log = logging.getLogger("safemmsa.api")

# 单文件大小上限（部署机内存有限，防 OOM）。
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时按 ENV(SAFEMMSA_MODELS_CONFIG) 挂载真实模型；无配置则全用桩/基线。"""
    # 生产前置校验：仍用内置默认弱密钥（令牌可伪造）则拒绝启动。
    auth.assert_secret_configured()
    with suppress(Exception):
        from models import load_models  # type: ignore

        load_models()
    yield


app = FastAPI(
    title="SAFE-MMSA API",
    version=safemmsa.__version__,
    description="状态感知驱动融合增强的蒙古语多模态情感分析（predict-then-correct）",
    lifespan=_lifespan,
)

# 允许前端跨域（Next.js 演示页）。生产经 nginx 同源反代（/api 与页面同源），本不需 CORS；
# 默认仅放行本地开发源，跨源需求用 SAFEMMSA_CORS_ORIGINS 显式列举（不再默认通配 *）。
_DEFAULT_CORS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:3100,http://127.0.0.1:3100"
)
_origins = os.environ.get("SAFEMMSA_CORS_ORIGINS", _DEFAULT_CORS).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 认证 / 历史 / 管理日志路由（模块11：持久化层对外接口）。
app.include_router(auth_router)

# 评测路由（模块9）；eval 包就绪时挂载，缺失不影响核心 API。
with suppress(Exception):
    from eval.routes import router as eval_router  # type: ignore

    app.include_router(eval_router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": safemmsa.__version__, "language": safemmsa.SYSTEM_LANGUAGE}


@app.get("/api/models/status")
def models_status_endpoint() -> dict:
    """各 register_* 插点的桩/真模型状态。"""
    return models_status()


def _spill_to_temp(upload: Optional[UploadFile], suffix: str) -> Optional[str]:
    """把上传文件落到临时路径（带大小上限）；无文件返回 None。调用方负责删除。"""
    if upload is None or not upload.filename:
        return None
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="safemmsa_")
    written = 0
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                out.close()
                with suppress(OSError):
                    os.remove(path)
                raise ValueError(f"上传文件超过上限 {_MAX_UPLOAD_BYTES // (1024*1024)}MB")
            out.write(chunk)
    if written == 0:
        with suppress(OSError):
            os.remove(path)
        return None
    return path


@app.post("/api/analyze", response_model=AnalyzeResult)
async def analyze(
    text: Optional[str] = Form(default=None),
    video: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
) -> AnalyzeResult:
    """三模态（任意子集）→ 完整 AnalyzeResult：初始预测 → 三状态 → 最终 + 修正幅度。"""
    # 先解析归属：无鉴权头→anonymous；带头但令牌无效→401（在做分析/落临时文件前就拒绝）。
    user_id = resolve_user_id(authorization)
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    try:
        # 落临时文件放进 try：任一路超限时，finally 仍会清掉已落盘的另一路（防临时文件泄漏）。
        try:
            video_path = _spill_to_temp(video, ".mp4")
            audio_path = _spill_to_temp(audio, ".wav")
        except ValueError as e:
            raise HTTPException(status_code=413, detail=str(e))
        result = safemmsa.analyze(
            video=video_path,
            audio=audio_path,
            text=(text.strip() if text and text.strip() else None),
        )
        # 持久化为规范化记录（analysis_record + modality_output + state_diagnosis +
        # modality_reliability），并记一条操作日志。存储失败绝不影响分析返回，
        # 但必须记日志（不再静默吞掉，以免审计记录/落库失败无迹可查）。
        try:
            repo = get_repository()
            bundle = analyze_result_to_bundle(result, user_id=user_id)
            record_id = repo.save_analysis(bundle)
            repo.add_log(
                OperationLog(
                    user_id=user_id,
                    action="analyze",
                    detail={"record_id": record_id, "final_emotion": result.final_emotion},
                )
            )
        except Exception:
            _log.exception("持久化分析结果失败（分析结果仍照常返回）")
        return result
    finally:
        for p in (video_path, audio_path):
            if p:
                with suppress(OSError):
                    os.remove(p)


# 运行：uvicorn api.main:app --host 127.0.0.1 --port 8100
