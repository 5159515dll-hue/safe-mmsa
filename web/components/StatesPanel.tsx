"use client";

import type {
  BoundaryState,
  CompetitionState,
  ReliabilityState,
} from "@/lib/types";
import { LABEL_ZH } from "@/lib/types";
import { MeterBar } from "./Bars";

const MODALITY_ZH: Record<string, string> = { video: "视频", audio: "音频", text: "文本" };

// 边界评分半圆仪表（SVG）。
function Gauge({ score }: { score: number }) {
  const angle = Math.PI * (1 - Math.max(0, Math.min(1, score))); // 0→右, 1→左
  const cx = 90;
  const cy = 90;
  const r = 70;
  const x = cx + r * Math.cos(angle);
  const y = cy - r * Math.sin(angle);
  const color = score > 0.66 ? "var(--red)" : score > 0.33 ? "var(--amber)" : "var(--green)";
  return (
    <svg viewBox="0 0 180 110" className="gauge">
      <path d="M20 90 A70 70 0 0 1 160 90" fill="none" stroke="var(--line)" strokeWidth="14" />
      <path
        d={`M20 90 A70 70 0 0 1 ${x} ${y}`}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
      />
      <text x="90" y="78" textAnchor="middle" className="gauge-val" fill={color}>
        {score.toFixed(3)}
      </text>
      <text x="90" y="100" textAnchor="middle" className="gauge-cap">
        边界评分 B
      </text>
    </svg>
  );
}

export function StatesPanel({
  boundary,
  competition,
  reliability,
}: {
  boundary: BoundaryState;
  competition: CompetitionState;
  reliability: ReliabilityState;
}) {
  const mods = Object.keys(reliability.per_modality);
  return (
    <div className="states-grid">
      {/* 边界评分 B */}
      <div className="card state">
        <div className="state-title">① 边界评分 B</div>
        <Gauge score={boundary.score} />
        <div className="kv">
          <div><span>最大置信</span><b className="mono">{boundary.max_confidence.toFixed(3)}</b></div>
          <div><span>Top-2 间隔</span><b className="mono">{boundary.top2_margin.toFixed(3)}</b></div>
          <div><span>归一化熵</span><b className="mono">{boundary.norm_entropy.toFixed(3)}</b></div>
        </div>
        <p className="hint">B 越高 → 越接近决策边界 → 允许更大的有界修正。</p>
      </div>

      {/* 类别竞争 C */}
      <div className="card state">
        <div className="state-title">② 类别竞争 C</div>
        <div className="versus">
          <span className="vs-top">{LABEL_ZH[competition.top1]}</span>
          <span className="vs-mid">vs</span>
          <span className="vs-top2">{LABEL_ZH[competition.top2]}</span>
        </div>
        <div className="dir-list">
          {Object.entries(competition.per_modality_direction).length === 0 && (
            <div className="muted">无单模态 logits（视/声为桩）</div>
          )}
          {Object.entries(competition.per_modality_direction).map(([m, d]) => (
            <div className="dir-row" key={m}>
              <span>{MODALITY_ZH[m] ?? m}</span>
              <span className={"dir " + (d >= 0 ? "pos" : "neg")}>
                {d >= 0 ? "▲ 偏 " + LABEL_ZH[competition.top1] : "▼ 偏 " + LABEL_ZH[competition.top2]}
                <b className="mono"> {d.toFixed(2)}</b>
              </span>
            </div>
          ))}
        </div>
        <div className="kv">
          <div><span>margin</span><b className="mono">{competition.margin.toFixed(3)}</b></div>
          <div><span>熵</span><b className="mono">{competition.entropy.toFixed(3)}</b></div>
        </div>
        <p className="hint mono small">C 向量 = [{competition.vector.map((v) => v.toFixed(2)).join(", ")}]</p>
      </div>

      {/* 模态可靠性 R */}
      <div className="card state">
        <div className="state-title">③ 模态可靠性 R</div>
        {mods.length === 0 && <div className="muted">无可用模态</div>}
        {mods.map((m) => (
          <div className="rel-block" key={m}>
            <MeterBar
              label={MODALITY_ZH[m] ?? m}
              value={reliability.per_modality[m]}
              tone="green"
            />
            <div className="rel-parts">
              <span>质量 {fmt(reliability.quality_part[m])}</span>
              <span>确定性 {fmt(reliability.uncertainty_part[m])}</span>
            </div>
          </div>
        ))}
        {Object.keys(reliability.consistency_pairwise).length > 0 && (
          <div className="pairwise">
            <div className="pw-title">跨模态一致性</div>
            {Object.entries(reliability.consistency_pairwise).map(([k, v]) => (
              <div className="pw-row" key={k}>
                <span>{k}</span>
                <b className="mono">{v.toFixed(3)}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function fmt(v: number | undefined) {
  return v == null ? "—" : v.toFixed(2);
}
