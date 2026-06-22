"use client";

import { useState } from "react";
import type { AuthUser } from "@/lib/types";
import { login, register } from "@/lib/api";

// 首屏登录/注册落地页。onAuthed：登录或注册成功；onGuest：游客继续（不登录，归 anonymous）。
export function LandingAuth({
  onAuthed,
  onGuest,
}: {
  onAuthed: (u: AuthUser) => void;
  onGuest: () => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    if (!username.trim()) return setErr("请输入用户名");
    if (mode === "register" && password.length < 6) return setErr("口令至少 6 位");
    setBusy(true);
    try {
      const u = await (mode === "login" ? login : register)(username.trim(), password);
      onAuthed(u);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="landing">
      <div className="landing-mongol" aria-hidden="true">ᠮᠣᠩᠭᠣᠯ</div>
      <div className="landing-card reveal">
        <span className="eyebrow">蒙古语 · 多模态情感分析</span>
        <h1 className="landing-word">SAFE-MMSA</h1>
        <p className="landing-sub">State-Aware Fusion-Enhanced · Mongolian Multi-modal Sentiment Analysis</p>

        <div className="landing-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "login"}
            className={"lt-tab" + (mode === "login" ? " active" : "")}
            onClick={() => { setMode("login"); setErr(null); }}
          >
            登录
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "register"}
            className={"lt-tab" + (mode === "register" ? " active" : "")}
            onClick={() => { setMode("register"); setErr(null); }}
          >
            注册
          </button>
        </div>

        <div className="landing-form">
          <input
            className="landing-input"
            placeholder="用户名"
            value={username}
            autoComplete="username"
            onChange={(e) => setUsername(e.target.value)}
          />
          <input
            className="landing-input"
            type="password"
            placeholder={mode === "register" ? "口令（≥6 位）" : "口令"}
            value={password}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          {err && <div className="landing-err" role="alert">{err}</div>}
          <button type="button" className="landing-submit" disabled={busy} onClick={submit}>
            {busy ? "处理中…" : mode === "login" ? "登录" : "注册并登录"}
          </button>
        </div>

        <button type="button" className="landing-guest" onClick={onGuest}>
          游客继续，直接体验分析 →
        </button>
      </div>
      <p className="landing-foot mono">predict-then-correct · 三状态 B·C·R 驱动有界融合增强</p>
    </main>
  );
}
