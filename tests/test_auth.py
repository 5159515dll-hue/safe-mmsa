"""模块11 — 认证单元测试 + API 流程（注册/登录/历史隔离/权限）。"""
from __future__ import annotations

import pytest

from safemmsa import auth


# ── 单元：口令哈希 & 令牌 ──────────────────────────────────────────
def test_password_hash_verify():
    h = auth.hash_password("hunter2x")
    assert h != "hunter2x"
    assert auth.verify_password("hunter2x", h)
    assert not auth.verify_password("hunter2", h)
    assert not auth.verify_password("hunter2x", "garbage")


def test_token_round_trip_and_expiry():
    tok = auth.issue_token("usr_1")
    assert auth.verify_token(tok) == "usr_1"
    assert auth.verify_token(auth.issue_token("usr_1", ttl_seconds=-1)) is None  # 过期
    assert auth.verify_token(tok + "x") is None  # 篡改
    assert auth.verify_token("not-base64!!") is None


def test_parse_bearer():
    tok = auth.issue_token("usr_2")
    assert auth.parse_bearer(f"Bearer {tok}") == "usr_2"
    assert auth.parse_bearer(f"bearer {tok}") == "usr_2"
    assert auth.parse_bearer(tok) is None  # 缺 scheme
    assert auth.parse_bearer(None) is None


# ── API 流程 ───────────────────────────────────────────────────────
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_register_login_me_flow():
    r = client.post("/api/auth/register", json={"username": "dave", "password": "secret123"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    assert r.json()["user"]["role"] == "user"

    # 重复注册 → 409
    assert client.post("/api/auth/register", json={"username": "dave", "password": "secret123"}).status_code == 409
    # 弱口令 → 400
    assert client.post("/api/auth/register", json={"username": "x", "password": "123"}).status_code == 400

    # 登录
    r2 = client.post("/api/auth/login", json={"username": "dave", "password": "secret123"})
    assert r2.status_code == 200
    # 错误口令 → 401
    assert client.post("/api/auth/login", json={"username": "dave", "password": "nope"}).status_code == 401

    # me
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["username"] == "dave"
    assert client.get("/api/auth/me").status_code == 401


def test_analyze_persists_and_history_is_user_scoped():
    reg = client.post("/api/auth/register", json={"username": "erin", "password": "secret123"})
    token = reg.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    a = client.post("/api/analyze", data={"text": "I am so happy"}, headers=headers)
    assert a.status_code == 200

    # 登录用户能在历史看到自己的记录
    h = client.get("/api/history", headers=headers)
    assert h.status_code == 200
    items = h.json()["items"]
    assert len(items) == 1 and items[0]["final_emotion"] == "happy"
    rid = items[0]["record_id"]

    # 明细可取（本人），含 3 模态 + 状态诊断
    d = client.get(f"/api/history/{rid}", headers=headers)
    assert d.status_code == 200
    body = d.json()
    assert len(body["modality_outputs"]) == 3
    assert body["state_diagnosis"]["record_id"] == rid

    # 匿名历史不含该用户记录
    anon_hist = client.get("/api/history").json()["items"]
    assert all(it["record_id"] != rid for it in anon_hist)


def test_history_detail_blocks_cross_user_idor():
    """他人 / 匿名不得通过 record_id 读取登录用户的分析明细（防越权）。"""
    owner = client.post("/api/auth/register", json={"username": "grace", "password": "secret123"})
    otok = owner.json()["token"]
    client.post("/api/analyze", data={"text": "I am so happy"}, headers={"Authorization": f"Bearer {otok}"})
    rid = client.get("/api/history", headers={"Authorization": f"Bearer {otok}"}).json()["items"][0]["record_id"]

    # 本人可读
    assert client.get(f"/api/history/{rid}", headers={"Authorization": f"Bearer {otok}"}).status_code == 200
    # 匿名（无令牌）读 → 404（不泄露存在性）
    assert client.get(f"/api/history/{rid}").status_code == 404
    # 另一登录用户读 → 404
    other = client.post("/api/auth/register", json={"username": "heidi", "password": "secret123"})
    htok = other.json()["token"]
    assert client.get(f"/api/history/{rid}", headers={"Authorization": f"Bearer {htok}"}).status_code == 404


def test_admin_logs_requires_admin():
    # 普通用户被拒
    reg = client.post("/api/auth/register", json={"username": "frank", "password": "secret123"})
    token = reg.json()["token"]
    assert client.get("/api/admin/logs", headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert client.get("/api/admin/logs").status_code == 403

    # 直接造一个 admin（注册接口不暴露提权），其令牌可读日志
    from safemmsa import auth as _auth
    from safemmsa.repository import get_repository

    admin = get_repository().create_user("root", "secret123", role="admin")
    atoken = _auth.issue_token(admin.user_id)
    ok = client.get("/api/admin/logs", headers={"Authorization": f"Bearer {atoken}"})
    assert ok.status_code == 200 and "logs" in ok.json()
