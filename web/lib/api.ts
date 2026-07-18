import type { AnalyzeResult, AuthUser, HistoryItem, ModelsStatus } from "./types";

// 同源（nginx 反代 /api）；本地开发用 NEXT_PUBLIC_API_BASE 指向 127.0.0.1:8100。
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";
const TOKEN_KEY = "safemmsa_token";

// ── 令牌存储（localStorage）+ 鉴权头 ──────────────────────────────
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}
function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}
function authHeader(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export async function analyze(input: {
  text?: string;
  video?: File | null;
  audio?: File | null;
}): Promise<AnalyzeResult> {
  const form = new FormData();
  if (input.text && input.text.trim()) form.append("text", input.text.trim());
  if (input.video) form.append("video", input.video);
  if (input.audio) form.append("audio", input.audio);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
    headers: authHeader(),
  });
  if (!res.ok) {
    // 不在此清除令牌：清了会让下一次 analyze 静默以游客身份提交，把本应属于该用户的
    // 输入写进所有人共享的 anonymous 桶（隐私回退）。令牌过期由 fetchMe 统一清除并登出。
    const detail = await res.text().catch(() => "");
    throw new Error(`分析失败 (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── 认证 ──────────────────────────────────────────────────────────
async function postCredentials(path: string, username: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || `${res.status}`);
  }
  const data = await res.json();
  setToken(data.token);
  return data.user as AuthUser;
}

export const register = (u: string, p: string) => postCredentials("/api/auth/register", u, p);
export const login = (u: string, p: string) => postCredentials("/api/auth/login", u, p);
export function logout() {
  setToken(null);
}

export async function fetchMe(): Promise<AuthUser | null> {
  if (!getToken()) return null;
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeader() });
    if (!res.ok) {
      if (res.status === 401) setToken(null); // 过期/失效令牌自动清除
      return null;
    }
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHistory(limit = 10): Promise<HistoryItem[]> {
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=${limit}`, { headers: authHeader() });
    if (!res.ok) return [];  // 401（令牌过期）静默降级为空历史；令牌清除交由 fetchMe 统一处理
    const data = await res.json();
    return (data.items ?? []) as HistoryItem[];
  } catch {
    return [];
  }
}

export async function fetchModelsStatus(): Promise<ModelsStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/api/models/status`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
