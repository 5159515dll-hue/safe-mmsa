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


# ── 安全加固回归（对应本次安全审查坐实的问题）─────────────────────────

def test_assert_secret_configured_refuses_default(monkeypatch):
    """默认弱密钥（未配置 SAFEMMSA_SECRET）必须拒绝启动，除非显式豁免。"""
    monkeypatch.delenv("SAFEMMSA_SECRET", raising=False)
    monkeypatch.delenv("SAFEMMSA_ALLOW_DEFAULT_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        auth.assert_secret_configured()
    monkeypatch.setenv("SAFEMMSA_ALLOW_DEFAULT_SECRET", "1")  # 本地开发显式豁免
    auth.assert_secret_configured()
    monkeypatch.delenv("SAFEMMSA_ALLOW_DEFAULT_SECRET", raising=False)
    monkeypatch.setenv("SAFEMMSA_SECRET", "a-strong-random-secret")  # 已配置真实密钥
    auth.assert_secret_configured()


def test_secret_guard_rejects_blank_default_and_short(monkeypatch):
    """空/空白/默认/过短密钥都算弱密钥，必须被守卫拒绝（不能只比默认字面量）。"""
    import base64
    import hashlib
    import hmac
    import time

    monkeypatch.delenv("SAFEMMSA_ALLOW_DEFAULT_SECRET", raising=False)
    for weak in ("", "   ", "safe-mmsa-dev-secret-change-me", "short"):
        monkeypatch.setenv("SAFEMMSA_SECRET", weak)
        assert auth.secret_is_weak() is True
        with pytest.raises(RuntimeError):
            auth.assert_secret_configured()
        with pytest.raises(RuntimeError):
            auth.issue_token("usr_x")  # 弱密钥下拒绝签发
        # 关键：即便手工用当前(弱)密钥签出一枚"签名有效、未过期"的令牌，verify_token 也必须拒绝。
        # （用 "any-token" 断言 None 是空断言——旧码经 except 同样返回 None，抓不到守卫回退。）
        exp = int(time.time()) + 3600
        payload = f"usr_x:{exp}"
        sig = hmac.new(auth._secret(), payload.encode(), hashlib.sha256).hexdigest()
        signed = base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()
        assert auth.verify_token(signed) is None  # 弱密钥 fail-closed：合法签名也拒
    monkeypatch.delenv("SAFEMMSA_SECRET", raising=False)  # 未设 → 弱
    assert auth.secret_is_weak() is True
    monkeypatch.setenv("SAFEMMSA_SECRET", "x" * 16)  # 恰好达标
    assert auth.secret_is_weak() is False
    auth.assert_secret_configured()


def test_verify_token_rejects_empty_user_id():
    """空 user_id 的合法签名令牌也必须拒绝（不依赖下游 falsy 兜底）。"""
    import base64
    import hashlib
    import hmac
    import time

    exp = int(time.time()) + 3600
    payload = f":{exp}"  # 空 user_id
    sig = hmac.new(auth._secret(), payload.encode(), hashlib.sha256).hexdigest()
    tok = base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()
    assert auth.verify_token(tok) is None


def test_register_rejects_out_of_bound_credentials():
    assert client.post("/api/auth/register", json={"username": "u" * 65, "password": "secret123"}).status_code == 400
    assert client.post("/api/auth/register", json={"username": "okname", "password": "x" * 129}).status_code == 400
    assert client.post("/api/auth/register", json={"username": "okname", "password": "12345"}).status_code == 400


def test_history_and_analyze_reject_invalid_token_but_allow_guest():
    """有鉴权头但令牌无效 → 401（不静默降级到 anonymous）；无头 → 游客 200。"""
    assert client.get("/api/history").status_code == 200  # 无头=游客
    assert client.get("/api/history", headers={"Authorization": "Bearer garbage.token"}).status_code == 401
    assert client.post("/api/analyze", data={"text": "hi"}).status_code == 200  # 无头 analyze=游客
    assert client.post(
        "/api/analyze", data={"text": "hi"}, headers={"Authorization": "Bearer garbage.token"}
    ).status_code == 401


def test_history_limit_is_clamped():
    """负数/超大 limit 不得 500，也不得导致跨后端行为分歧。"""
    assert client.get("/api/history?limit=-1").status_code == 200
    assert client.get("/api/history?limit=999999").status_code == 200
    assert client.get("/api/history?limit=0").status_code == 200


def test_clamp_limit_semantics():
    """直接钉死收敛语义：透传实现（return limit）会让本用例失败。"""
    from api.auth_routes import _MAX_LIST_LIMIT, _clamp_limit

    assert _clamp_limit(-1) == 1
    assert _clamp_limit(0) == 1
    assert _clamp_limit(5) == 5
    assert _clamp_limit(_MAX_LIST_LIMIT) == _MAX_LIST_LIMIT
    assert _clamp_limit(999999) == _MAX_LIST_LIMIT


def test_history_huge_and_negative_limit_return_same_bounded_set():
    """超大与负数 limit 返回的条数受钉；透传实现下 limit=-1 会返回 rows[:-1]（缺一条），据此可抓回退。"""
    reg = client.post("/api/auth/register", json={"username": "limituser", "password": "secret123"})
    headers = {"Authorization": f"Bearer {reg.json()['token']}"}
    for _ in range(3):
        client.post("/api/analyze", data={"text": "I am so happy"}, headers=headers)
    n_all = len(client.get("/api/history?limit=999999", headers=headers).json()["items"])
    n_neg = len(client.get("/api/history?limit=-1", headers=headers).json()["items"])
    assert n_all == 3, f"超大 limit 应返回全部 3 条，得 {n_all}"
    assert n_neg == 1, f"limit=-1 应被收敛到 1（而非透传的 rows[:-1]=2 条），得 {n_neg}"


def test_json_create_user_username_unique_under_concurrency():
    """JSON 后端用户名唯一性在并发下也成立（修复前会产生重复账号）。"""
    import threading

    from safemmsa.repository import get_repository

    repo = get_repository()
    n = 20
    barrier = threading.Barrier(n)
    oks: list[str] = []
    errs: list[int] = []

    def worker() -> None:
        barrier.wait()  # 最大化 check-then-insert 的并发重叠
        try:
            oks.append(repo.create_user("racer", "secret123").user_id)
        except ValueError:
            errs.append(1)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = [r for r in repo.users.all() if r.get("username") == "racer"]
    assert len(rows) == 1, f"用户名唯一性被破坏：存在 {len(rows)} 个 'racer'"
    assert len(oks) == 1 and len(errs) == n - 1
