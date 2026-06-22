"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { analyze, fetchHistory, fetchMe, fetchModelsStatus, logout } from "@/lib/api";
import type { AnalyzeResult, AuthUser, HistoryItem, ModelsStatus } from "@/lib/types";
import { LABEL_ZH } from "@/lib/types";
import { DistributionBars } from "@/components/Bars";
import { ModalityCard } from "@/components/ModalityCard";
import { StatesPanel } from "@/components/StatesPanel";
import { AnalyzingPipeline } from "@/components/AnalyzingPipeline";
import { FileDrop } from "@/components/FileDrop";
import { CountUp } from "@/components/CountUp";
import { AuthBar } from "@/components/AuthBar";
import { LandingAuth } from "@/components/LandingAuth";

// 一键填充示例文本，便于不上传文件也能跑通整条流水线（桩模式仅演示流程，不据文本判情感）。
const EXAMPLES: { label: string; text: string; mongol?: boolean }[] = [
  { label: "蒙文 · 问候", text: "ᠰᠠᠶᠢᠨ ᠪᠠᠶᠢᠨ᠎ᠠ ᠤᠤ", mongol: true },
  { label: "中文示例", text: "今天的演出太精彩了，我特别开心！" },
  { label: "英文示例", text: "I can't believe this happened, it's terrible." },
];

// 历史时间戳（后端为 epoch 秒）→ 本地 月-日 时:分。
function fmtTime(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// 空状态引导里复用的四个阶段（与结果区 ①~④ 对应）。
const EMPTY_STEPS = [
  { n: "1", t: "模态编码 + 质量" },
  { n: "2", t: "初始预测 p⁰" },
  { n: "3", t: "三状态 B·C·R" },
  { n: "4", t: "有界增强 p*" },
];

const Pipeline3D = dynamic(() => import("@/components/Pipeline3D").then((m) => m.Pipeline3D), {
  ssr: false,
  loading: () => <div className="card muted small">加载 3D 运行视图…</div>,
});

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fear: "😨",
  surprise: "😲",
  disgust: "🤢",
  neutral: "😐",
};

export default function Page() {
  const [text, setText] = useState("");
  const [video, setVideo] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [resultSeq, setResultSeq] = useState(0);
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  // 登录门：loading→检查令牌中；landing→显示登录/注册落地页；app→进入分析页。
  const [gate, setGate] = useState<"loading" | "landing" | "app">("loading");

  useEffect(() => {
    fetchModelsStatus().then(setStatus);
    fetchMe().then((u) => {
      if (u) {
        setUser(u);
        refreshHistory(u);
        setGate("app");
      } else {
        setGate("landing");
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function refreshHistory(u: AuthUser | null) {
    if (u) fetchHistory(8).then(setHistory);
    else setHistory([]);
  }

  function onAuthed(u: AuthUser) {
    setUser(u);
    refreshHistory(u);
    setGate("app");
  }

  function onLogout() {
    logout();
    setUser(null);
    setHistory([]);
    setGate("landing");
  }

  async function onSubmit() {
    setError(null);
    setLoading(true);
    try {
      const r = await analyze({ text, video, audio });
      setResult(r);
      setResultSeq((s) => s + 1);
      if (user) refreshHistory(user); // 登录用户：分析后刷新历史
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  const correction = result?.enhancement.correction_magnitude ?? 0;

  if (gate === "loading") {
    return (
      <main className="gate-splash">
        <span className="spinner" /> 加载中…
      </main>
    );
  }
  if (gate === "landing") {
    return <LandingAuth onAuthed={onAuthed} onGuest={() => setGate("app")} />;
  }

  return (
    <main className="page">
      <header className="hero">
        <div className="hero-mongol" aria-hidden="true">ᠮᠣᠩᠭᠣᠯ</div>
        <div className="hero-top">
          <span className="eyebrow">蒙古语 · 视频 + 音频 + 文本</span>
          <div className="hero-top-right">
            {status && (
              <span className={"mode-chip " + (status.all_stub ? "stub" : "real")}>
                <span className="dot" />
                {status.all_stub ? "桩 / 基线演示" : "已接入真实模型"}
              </span>
            )}
            <AuthBar user={user} onLogout={onLogout} onLogin={() => setGate("landing")} />
          </div>
        </div>
        <h1 className="wordmark">SAFE-MMSA</h1>
        <p className="expand">State-Aware Fusion-Enhanced · Mongolian Multi-modal Sentiment Analysis</p>
        <p className="thesis">
          先由融合主干给出初始判断 <b>p⁰</b>，再用三状态——边界 / 类别竞争 / 模态可靠性——
          门控一次<b>有界修正</b>，得到最终情感 <b>p*</b>。整条 predict-then-correct 流程逐步可观测。
        </p>
        <div className="pipeline-rule" aria-hidden="true">
          <span>输入</span><i></i><span>编码 ×3</span><i></i><span>融合 p⁰</span><i></i>
          <span>三状态 B·C·R</span><i></i><span>有界增强</span><i></i><span className="end">p*</span>
          <span className="eq">‖p*−p⁰‖ ≤ η·B</span>
        </div>
      </header>

      {/* 输入区 */}
      <section className="card inputs">
        <span className="card-eyebrow">INPUT · 三模态联合输入</span>
        <label className="field">
          <span><em className="chcode">TXT</em>蒙古文文本（也接受中 / 英，用于桩演示）</span>
          <textarea
            className="mongol"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="ᠰᠠᠶᠢᠨ ᠪᠠᠶᠢᠨ᠎ᠠ ᠤᠤ … 或输入中 / 英文本试桩"
          />
        </label>
        <div className="examples">
          <span className="ex-cap">示例</span>
          {EXAMPLES.map((ex) => (
            <button
              type="button"
              key={ex.label}
              className={"ex-chip" + (ex.mongol ? " mongol" : "")}
              onClick={() => setText(ex.text)}
            >
              {ex.label}
            </button>
          ))}
        </div>
        <div className="file-row">
          <FileDrop code="VID" label="视频文件（可选）" accept="video/*" file={video} onFile={setVideo} hint="点击或拖拽视频" />
          <FileDrop code="AUD" label="音频文件（可选）" accept="audio/*" file={audio} onFile={setAudio} hint="点击或拖拽音频" />
        </div>
        <button className="btn" onClick={onSubmit} disabled={loading} aria-busy={loading}>
          {loading ? "分析中…" : "开始分析"}
        </button>
        {error && <div className="error" role="alert">{error}</div>}
      </section>

      {!result && !loading && (
        <section className="card empty-guide">
          <span className="card-eyebrow">READY · 等待输入</span>
          <p className="eg-lead">
            上传任意模态或选一个示例，点击「开始分析」——下面会按 predict-then-correct
            逐阶段展开：
          </p>
          <div className="eg-steps">
            {EMPTY_STEPS.map((s) => (
              <div className="eg-step" key={s.n}>
                <em className="eg-n">{s.n}</em>
                <span>{s.t}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {user && history.length > 0 && (
        <section className="card my-history">
          <span className="card-eyebrow">HISTORY · {user.username} 的最近分析</span>
          <ul className="hist-list">
            {history.map((h) => (
              <li className="hist-row" key={h.record_id}>
                <span className="hist-emo">
                  <span className="hist-emoji">{EMOTION_EMOJI[h.final_emotion]}</span>
                  {LABEL_ZH[h.final_emotion]}
                </span>
                <span className="hist-meta mono">
                  p⁰={LABEL_ZH[h.initial_top]} · B={h.boundary_score.toFixed(3)} · ‖Δ‖=
                  {h.correction_magnitude.toFixed(4)}
                </span>
                <span className="hist-time mono">{fmtTime(h.created_at)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {loading && <AnalyzingPipeline />}

      {result && !loading && (
        <section className="result" key={resultSeq}>
          {/* 最终结论 */}
          <div className="card final-banner reveal" style={{ animationDelay: "0s" }}>
            <div className="final-main">
              <div className="final-label">最终情感</div>
              <div className="final-emotion">
                <span className="final-emoji">{EMOTION_EMOJI[result.final_emotion]}</span>
                {LABEL_ZH[result.final_emotion]}
              </div>
              <div className="final-en">{result.final_emotion} · language={result.language}</div>
            </div>
            <div className="final-meta">
              <div>
                <span>融合主干</span>
                <b>{result.initial.backbone}</b>
              </div>
              <div>
                <span>增强</span>
                <b>{result.enhancement.enhancement}</b>
              </div>
              <div>
                <span>修正幅度 ‖p*−p⁰‖</span>
                <b className="mono">
                  <CountUp value={correction} decimals={4} />
                  {result.enhancement.bounded ? " (有界)" : ""}
                </b>
              </div>
            </div>
          </div>

          {/* 运行可视化（3D，数据驱动） */}
          <div className="reveal" style={{ animationDelay: ".04s" }}>
            <Pipeline3D result={result} seq={resultSeq} />
          </div>

          {/* 阶段1：模态 + 质量 */}
          <h2 className="stage reveal" style={{ animationDelay: ".06s" }}>① 模态编码 + 信号质量</h2>
          <div className="modalities">
            {result.modalities.map((m, i) => (
              <div className="reveal" key={m.modality} style={{ animationDelay: `${0.12 + i * 0.06}s` }}>
                <ModalityCard output={m} />
              </div>
            ))}
          </div>

          {/* 阶段2：初始预测 */}
          <h2 className="stage reveal" style={{ animationDelay: ".3s" }}>② 初始预测 p⁰（融合主干）</h2>
          <div className="card reveal" style={{ animationDelay: ".34s" }}>
            <DistributionBars distribution={result.initial.distribution} />
          </div>

          {/* 阶段3：三状态 */}
          <h2 className="stage reveal" style={{ animationDelay: ".42s" }}>
            ③ 三状态 · 边界 / 类别竞争 / 模态可靠性
          </h2>
          <div className="reveal" style={{ animationDelay: ".46s" }}>
            <StatesPanel
              boundary={result.boundary}
              competition={result.competition}
              reliability={result.reliability}
            />
          </div>

          {/* 阶段4：最终预测 */}
          <h2 className="stage reveal" style={{ animationDelay: ".56s" }}>④ 有界融合增强 → 最终预测 p*</h2>
          <div className="card reveal" style={{ animationDelay: ".6s" }}>
            <DistributionBars
              distribution={result.enhancement.final_distribution}
              highlight={result.final_emotion}
            />
            {correction === 0 && (
              <p className="hint">
                当前增强为 identity 基线：p* = p⁰，修正幅度 0。框架已就绪——增强网络 D 训练后经
                <code> register_enhancement </code>挂载即产生由 B 门控的非零有界修正。
              </p>
            )}
          </div>

          {result.timings_ms && (
            <p className="timings mono reveal" style={{ animationDelay: ".66s" }}>
              耗时(ms): {Object.entries(result.timings_ms).map(([k, v]) => `${k}=${v}`).join("  ")}
            </p>
          )}
        </section>
      )}
    </main>
  );
}
