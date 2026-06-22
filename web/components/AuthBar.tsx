"use client";

import { useEffect, useRef, useState } from "react";
import type { AuthUser } from "@/lib/types";
import { fetchMe, login, logout, register } from "@/lib/api";

// hero 右上角的认证栏：未登录展开内联登录/注册表单；登录后显示用户名 + 退出。
// 父组件通过 onUser 感知登录态变化（用于刷新「我的历史」）。
export function AuthBar({ onUser }: { onUser?: (u: AuthUser | null) => void }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  function emit(u: AuthUser | null) {
    setUser(u);
    onUser?.(u);
  }

  useEffect(() => {
    fetchMe().then(emit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 点击外部收起表单
  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  async function submit(kind: "login" | "register") {
    setErr(null);
    setBusy(true);
    try {
      const u = await (kind === "login" ? login : register)(username.trim(), password);
      emit(u);
      setOpen(false);
      setUsername("");
      setPassword("");
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  if (user) {
    return (
      <div className="authbar">
        <span className="auth-user" title={`role=${user.role}`}>
          <span className="auth-avatar">{user.username.slice(0, 1).toUpperCase()}</span>
          {user.username}
          {user.role === "admin" && <em className="auth-role">admin</em>}
        </span>
        <button
          type="button"
          className="auth-link"
          onClick={() => {
            logout();
            emit(null);
          }}
        >
          退出
        </button>
      </div>
    );
  }

  return (
    <div className="authbar" ref={ref}>
      <button type="button" className="auth-toggle" onClick={() => setOpen((v) => !v)}>
        登录 / 注册
      </button>
      {open && (
        <div className="auth-pop" role="dialog" aria-label="登录或注册">
          <input
            className="auth-input"
            placeholder="用户名"
            value={username}
            autoComplete="username"
            onChange={(e) => setUsername(e.target.value)}
          />
          <input
            className="auth-input"
            type="password"
            placeholder="口令（≥6 位）"
            value={password}
            autoComplete="current-password"
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit("login")}
          />
          {err && <div className="auth-err">{err}</div>}
          <div className="auth-actions">
            <button type="button" className="auth-btn primary" disabled={busy} onClick={() => submit("login")}>
              登录
            </button>
            <button type="button" className="auth-btn" disabled={busy} onClick={() => submit("register")}>
              注册
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
