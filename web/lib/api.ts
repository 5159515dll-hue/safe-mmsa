import type { AnalyzeResult, ModelsStatus } from "./types";

// 同源（nginx 反代 /api）；本地开发用 NEXT_PUBLIC_API_BASE 指向 127.0.0.1:8100。
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

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
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`分析失败 (${res.status}): ${detail}`);
  }
  return res.json();
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
