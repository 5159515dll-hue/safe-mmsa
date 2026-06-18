"use client";

import { EMOTION_LABELS, LABEL_ZH } from "@/lib/types";

// 7 类分布的水平条形图，高亮最大类。
export function DistributionBars({
  distribution,
  highlight,
  compact = false,
}: {
  distribution: Record<string, number> | null;
  highlight?: string;
  compact?: boolean;
}) {
  if (!distribution) {
    return <div className="muted">（无分布 · 桩/无真实模型）</div>;
  }
  const top = EMOTION_LABELS.reduce(
    (a, b) => ((distribution[b] ?? 0) > (distribution[a] ?? 0) ? b : a),
    EMOTION_LABELS[0]
  );
  return (
    <div className={compact ? "bars compact" : "bars"}>
      {EMOTION_LABELS.map((label) => {
        const v = distribution[label] ?? 0;
        const isTop = (highlight ?? top) === label;
        return (
          <div className="bar-row" key={label}>
            <div className="bar-label">
              {LABEL_ZH[label]} <span className="en">{label}</span>
            </div>
            <div className="bar-track">
              <div
                className={"bar-fill" + (isTop ? " top" : "")}
                style={{ width: `${Math.max(1, v * 100)}%` }}
              />
            </div>
            <div className="bar-val">{(v * 100).toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}

// 单值进度条（0..1），用于可靠性、各分量。
export function MeterBar({
  value,
  label,
  tone = "blue",
}: {
  value: number | null | undefined;
  label: string;
  tone?: "blue" | "green" | "amber" | "red";
}) {
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="mono">{value == null ? "—" : v.toFixed(3)}</span>
      </div>
      <div className="bar-track">
        <div className={`bar-fill tone-${tone}`} style={{ width: `${v * 100}%` }} />
      </div>
    </div>
  );
}
