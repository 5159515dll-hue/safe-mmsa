"""模块8 — 认证 / 历史 / 管理日志路由（挂在 /api 下）。

最小认证：注册 / 登录 / me；分析历史按登录用户隔离（未登录归 anonymous）；
管理员可读操作日志。令牌为无状态 HMAC 签名（见 safemmsa.auth）。
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from safemmsa import auth
from safemmsa.records import OperationLog, User
from safemmsa.repository import ANONYMOUS_USER_ID, get_repository

router = APIRouter(prefix="/api", tags=["auth"])


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password: str


def current_user(authorization: Optional[str]) -> Optional[User]:
    uid = auth.parse_bearer(authorization)
    if not uid:
        return None
    return get_repository().get_user(uid)


def resolve_user_id(authorization: Optional[str]) -> str:
    """登录则返回其 user_id，否则归到 anonymous（保证 FK 有效）。"""
    uid = auth.parse_bearer(authorization)
    if uid and get_repository().get_user(uid) is not None:
        return uid
    return ANONYMOUS_USER_ID


def _auth_payload(user: User) -> dict:
    return {"token": auth.issue_token(user.user_id), "user": user.public()}


@router.post("/auth/register")
def register(body: Credentials) -> dict:
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="口令至少 6 位")
    repo = get_repository()
    try:
        user = repo.create_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    repo.add_log(OperationLog(user_id=user.user_id, action="register", detail={"username": user.username}))
    return _auth_payload(user)


@router.post("/auth/login")
def login(body: Credentials) -> dict:
    repo = get_repository()
    user = repo.get_user_by_username(body.username.strip())
    if user is None or not auth.verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或口令错误")
    repo.add_log(OperationLog(user_id=user.user_id, action="login"))
    return _auth_payload(user)


@router.get("/auth/me")
def me(authorization: Optional[str] = Header(default=None)) -> dict:
    user = current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或令牌无效")
    return user.public()


@router.get("/history")
def history(limit: int = 20, authorization: Optional[str] = Header(default=None)) -> dict:
    """当前用户（或 anonymous）的分析历史摘要。"""
    user_id = resolve_user_id(authorization)
    records = get_repository().list_analyses(user_id=user_id, limit=limit)
    return {"items": [r.model_dump() for r in records]}


@router.get("/history/{record_id}")
def history_detail(record_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
    bundle = get_repository().get_analysis(record_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"找不到分析记录: {record_id}")
    # 归属校验：只能读自己的记录（未登录则只能读 anonymous 名下）。防越权 IDOR。
    if bundle.record.user_id != resolve_user_id(authorization):
        raise HTTPException(status_code=404, detail=f"找不到分析记录: {record_id}")
    return bundle.model_dump()


@router.get("/admin/logs")
def admin_logs(limit: int = 50, authorization: Optional[str] = Header(default=None)) -> dict:
    user = current_user(authorization)
    if user is None or user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return {"logs": [log.model_dump() for log in get_repository().list_logs(limit=limit)]}
