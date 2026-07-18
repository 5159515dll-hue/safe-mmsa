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

# 输入边界（防 pbkdf2 放大 DoS / 异常长输入）。
_MIN_PASSWORD = 6
_MAX_PASSWORD = 128
_MAX_USERNAME = 64
_MAX_LIST_LIMIT = 200
# 登录时若用户不存在，用它跑一次等价的 pbkdf2，抹平「存在/不存在」的响应时间差。
_TIMING_DUMMY_HASH = auth.hash_password("safe-mmsa-nonexistent-user-timing-guard")


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password: str


def _clamp_limit(limit: int) -> int:
    """把外部 limit 收敛到 [1, _MAX_LIST_LIMIT]，避免负数/超大值导致跨后端行为分歧或无界读取。"""
    if limit < 1:
        return 1
    return min(limit, _MAX_LIST_LIMIT)


def current_user(authorization: Optional[str]) -> Optional[User]:
    uid = auth.parse_bearer(authorization)
    if not uid:
        return None
    return get_repository().get_user(uid)


def resolve_user_id(authorization: Optional[str]) -> str:
    """归属判定（严格）：
    - 无鉴权头 → anonymous（允许游客，保证 FK 有效）；
    - 有头但令牌无效/过期/用户不存在 → 401，**不静默降级到 anonymous**。

    静默降级会把过期登录用户的分析写进所有人共享的 anonymous 桶（隐私泄露），
    并造成 /history 与 /auth/me 对同一坏令牌行为不一致。
    """
    if authorization is None:
        return ANONYMOUS_USER_ID
    uid = auth.parse_bearer(authorization)
    if uid and get_repository().get_user(uid) is not None:
        return uid
    raise HTTPException(status_code=401, detail="令牌无效或已过期")


def _auth_payload(user: User) -> dict:
    return {"token": auth.issue_token(user.user_id), "user": user.public()}


@router.post("/auth/register")
def register(body: Credentials) -> dict:
    username = body.username.strip()
    if not (1 <= len(username) <= _MAX_USERNAME):
        raise HTTPException(status_code=400, detail=f"用户名长度需为 1–{_MAX_USERNAME} 字符")
    if not (_MIN_PASSWORD <= len(body.password) <= _MAX_PASSWORD):
        raise HTTPException(status_code=400, detail=f"口令长度需为 {_MIN_PASSWORD}–{_MAX_PASSWORD} 字符")
    repo = get_repository()
    try:
        user = repo.create_user(username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    repo.add_log(OperationLog(user_id=user.user_id, action="register", detail={"username": user.username}))
    return _auth_payload(user)


@router.post("/auth/login")
def login(body: Credentials) -> dict:
    repo = get_repository()
    user = repo.get_user_by_username(body.username.strip())
    if user is None:
        # 抹平时序：用户不存在也跑一次等价 pbkdf2，避免「存在/不存在」的响应时间侧信道。
        auth.verify_password(body.password, _TIMING_DUMMY_HASH)
        raise HTTPException(status_code=401, detail="用户名或口令错误")
    if not auth.verify_password(body.password, user.password_hash):
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
    records = get_repository().list_analyses(user_id=user_id, limit=_clamp_limit(limit))
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
    return {"logs": [log.model_dump() for log in get_repository().list_logs(limit=_clamp_limit(limit))]}
