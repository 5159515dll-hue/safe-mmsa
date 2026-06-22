"use client";

import type { AuthUser } from "@/lib/types";

// hero 右上角的账户区（受控）：登录后显示用户名 + 退出；游客显示「登录 / 注册」回到落地页。
// 表单逻辑在 LandingAuth；这里只做展示与触发。
export function AuthBar({
  user,
  onLogout,
  onLogin,
}: {
  user: AuthUser | null;
  onLogout: () => void;
  onLogin: () => void;
}) {
  if (user) {
    return (
      <div className="authbar">
        <span className="auth-user" title={`role=${user.role}`}>
          <span className="auth-avatar">{user.username.slice(0, 1).toUpperCase()}</span>
          {user.username}
          {user.role === "admin" && <em className="auth-role">admin</em>}
        </span>
        <button type="button" className="auth-link" onClick={onLogout}>
          退出
        </button>
      </div>
    );
  }
  return (
    <div className="authbar">
      <span className="auth-guest-tag">游客</span>
      <button type="button" className="auth-toggle" onClick={onLogin}>
        登录 / 注册
      </button>
    </div>
  );
}
