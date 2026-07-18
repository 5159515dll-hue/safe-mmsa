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
_DEFAULT_SECRET = "safe-mmsa-dev-secret-change-me"  # 仅本地/测试；生产必须用 SAFEMMSA_SECRET 覆盖
_MIN_SECRET_LEN = 16  # 生产密钥最小长度（过短同样易被伪造）


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
    return os.environ.get("SAFEMMSA_SECRET", _DEFAULT_SECRET).encode("utf-8")


def _allow_default_secret() -> bool:
    return os.environ.get("SAFEMMSA_ALLOW_DEFAULT_SECRET") == "1"


def secret_is_weak() -> bool:
    """密钥是否弱到令牌可被伪造：未设 / 空 / 全空白 / 仍是内置默认 / 过短（<16）。
    不能只比默认字面量——``SAFEMMSA_SECRET=""`` 或 `" "` 同样让 HMAC 密钥可预测。"""
    s = os.environ.get("SAFEMMSA_SECRET")
    return (not s) or (not s.strip()) or (s == _DEFAULT_SECRET) or (len(s) < _MIN_SECRET_LEN)


def assert_secret_configured() -> None:
    """启动前置校验：密钥弱到令牌可伪造则拒绝启动——默认密钥公开在源码里，
    空/空白/过短同样可预测，任何人都能据此伪造任意 user_id 的合法令牌（绕过认证）。

    仅当显式设置 ``SAFEMMSA_ALLOW_DEFAULT_SECRET=1`` 时豁免（限本地开发/测试）。
    """
    if secret_is_weak() and not _allow_default_secret():
        raise RuntimeError(
            "拒绝启动：SAFEMMSA_SECRET 未配置或过弱（空/空白/默认/<16 字符），"
            "令牌可被任意伪造。请设置强随机 SAFEMMSA_SECRET（如 "
            "`export SAFEMMSA_SECRET=$(python -c \"import secrets;print(secrets.token_urlsafe(48))\")`）"
            "后再启动；本地开发/测试可显式设 SAFEMMSA_ALLOW_DEFAULT_SECRET=1 豁免。"
        )


def issue_token(user_id: str, *, ttl_seconds: int = _DEFAULT_TTL) -> str:
    """签发 base64url(payload + 签名)。payload = user_id:exp。"""
    if secret_is_weak() and not _allow_default_secret():
        raise RuntimeError("拒绝签发令牌：SAFEMMSA_SECRET 未配置或过弱（令牌可被伪造）。")
    exp = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{exp}"
    sig = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    raw = f"{payload}:{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def verify_token(token: str) -> Optional[str]:
    """校验签名与过期；通过返回 user_id，否则 None。"""
    if secret_is_weak() and not _allow_default_secret():
        return None  # 弱密钥下 fail-closed：拒绝一切令牌（含伪造），不依赖 lifespan 守卫
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        user_id, exp_s, sig = raw.rsplit(":", 2)
        if not user_id:  # 空 user_id 不是合法主体，直接拒绝（勿依赖下游 falsy 兜底）
            return None
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
