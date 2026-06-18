"use client";

import { useEffect, useState } from "react";
import { analyze, fetchModelsStatus } from "@/lib/api";
import type { AnalyzeResult, ModelsStatus } from "@/lib/types";
import { LABEL_ZH } from "@/lib/types";
import { DistributionBars } from "@/components/Bars";
import { ModalityCard } from "@/components/ModalityCard";
import { StatesPanel } from "@/components/StatesPanel";

export default function Page() {
  const [text, setText] = useState("");
  const [video, setVideo] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [status, setStatus] = useState<ModelsStatus | null>(null);

  useEffect(() => {
    fetchModelsStatus().then(setStatus);
  }, []);

  async function onSubmit() {
    setError(null);
    setLoading(true);
    try {
      const r = await analyze({ text, video, audio });
      setResult(r);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  const correction = result?.enhancement.correction_magnitude ?? 0;

  return (
    <main className="page">
      <header className="hero">
        <h1>SAFE-MMSA</h1>
        <p className="subtitle">状态感知驱动融合增强的蒙古语多模态情感分析 · predict-then-correct</p>
        {status && (
          <span className={"mode-badge " + (status.all_stub ? "stub" : "real")}>
            {status.all_stub ? "当前：桩/基线演示（未接真实模型）" : "当前：已接入真实模型"}
          </span>
        )}
      </header>

      {/* 输入区 */}
      <section className="card inputs">
        <label className="field">
          <span>蒙古文文本（也接受中/英，用于桩演示）</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="ᠰᠠᠶᠢᠨ ᠪᠠᠶᠢᠨ᠎ᠠ ᠤᠤ … 或输入中/英文本试桩"
          />
        </label>
        <div className="file-row">
          <label className="field">
            <span>视频文件（可选）</span>
            <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            <span>音频文件（可选）</span>
            <input type="file" accept="audio/*" onChange={(e) => setAudio(e.target.files?.[0] ?? null)} />
          </label>
        </div>
        <button className="btn" onClick={onSubmit} disabled={loading}>
          {loading ? "分析中…" : "分析"}
        </button>
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="result">
          {/* 最终结论 */}
          <div className="card final-banner">
            <div className="final-main">
              <div className="final-label">最终情感</div>
              <div className="final-emotion">{LABEL_ZH[result.final_emotion]}</div>
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
                <b className="mono">{correction.toFixed(4)}{result.enhancement.bounded ? " (有界)" : ""}</b>
              </div>
            </div>
          </div>

          {/* 阶段1：模态 + 质量 */}
          <h2 className="stage">① 模态编码 + 信号质量</h2>
          <div className="modalities">
            {result.modalities.map((m) => (
              <ModalityCard key={m.modality} output={m} />
            ))}
          </div>

          {/* 阶段2：初始预测 */}
          <h2 className="stage">② 初始预测 p⁰（融合主干）</h2>
          <div className="card">
            <DistributionBars distribution={result.initial.distribution} />
          </div>

          {/* 阶段3：三状态 */}
          <h2 className="stage">③ 三状态（确定性 · 论文创新点）</h2>
          <StatesPanel
            boundary={result.boundary}
            competition={result.competition}
            reliability={result.reliability}
          />

          {/* 阶段4：最终预测 */}
          <h2 className="stage">④ 有界融合增强 → 最终预测 p*</h2>
          <div className="card">
            <DistributionBars distribution={result.enhancement.final_distribution} highlight={result.final_emotion} />
            {correction === 0 && (
              <p className="hint">
                当前增强为 identity 基线：p* = p⁰，修正幅度 0。框架已就绪——增强网络 D 训练后经
                <code> register_enhancement </code>挂载即产生由 B 门控的非零有界修正。
              </p>
            )}
          </div>

          {result.timings_ms && (
            <p className="timings mono">
              耗时(ms): {Object.entries(result.timings_ms).map(([k, v]) => `${k}=${v}`).join("  ")}
            </p>
          )}
        </section>
      )}
    </main>
  );
}
