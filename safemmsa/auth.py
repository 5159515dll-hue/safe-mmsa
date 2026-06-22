"""模块11b — 最小认证：口令哈希（pbkdf2_hmac）+ 无状态签名令牌（hmac）。

仅用标准库（hashlib / hmac / secrets / base64），不引入任何新依赖、不依赖 torch。
令牌是无状态的（payload = user_id:exp，HMAC-SHA256 签名），因此无需会话表，
与 fig4-5 的 E-R（无 session 实体）一致。生产环境务必用 SAFEMMSA_SECRET 覆盖密钥。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

_ALGO = "pbkdf2_sha256"
_ROUNDS = 200_000
_DEFAULT_TTL = 7 * 24 * 3600  # 7 天


def hash_password(password: str) -> str:
    """返回 ``pbkdf2_sha256$rounds$salt$hash`` 格式的可校验串。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ROUNDS)
    return f"{_ALGO}${_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """恒定时间比对，格式不符或异常一律返回 False。"""
    try:
        algo, rounds_s, salt_hex, hash_hex = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds_s)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def _secret() -> bytes:
    return os.environ.get("SAFEMMSA_SECRET", "safe-mmsa-dev-secret-change-me").encode("utf-8")


def issue_token(user_id: str, *, ttl_seconds: int = _DEFAULT_TTL) -> str:
    """签发 base64url(payload + 签名)。payload = user_id:exp。"""
    exp = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{exp}"
    sig = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    raw = f"{payload}:{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def verify_token(token: str) -> Optional[str]:
    """校验签名与过期；通过返回 user_id，否则 None。"""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        user_id, exp_s, sig = raw.rsplit(":", 2)
        payload = f"{user_id}:{exp_s}"
        expected = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        if int(exp_s) < int(time.time()):
            return None
        return user_id
    except Exception:
        return None


def parse_bearer(authorization: Optional[str]) -> Optional[str]:
    """从 ``Authorization: Bearer <token>`` 头解析 user_id；无/非法返回 None。"""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return verify_token(parts[1].strip())
