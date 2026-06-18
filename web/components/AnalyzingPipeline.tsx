"use client";

// 分析时的流水线动画：四个阶段节点依次脉冲，一道光束沿轨道扫过。
const STAGES = ["模态编码", "初始预测 p⁰", "三状态 B/C/R", "有界增强 p*"];

export function AnalyzingPipeline() {
  return (
    <div className="card analyzing">
      <div className="analyzing-track">
        <div className="analyzing-line" />
        <div className="analyzing-beam" />
        {STAGES.map((s, i) => (
          <div className="anode" key={s} style={{ animationDelay: `${i * 0.22}s` }}>
            <span className="anode-dot" />
            <span className="anode-label">{s}</span>
          </div>
        ))}
      </div>
      <div className="analyzing-cap">
        <span className="spinner" /> 分析中 · predict-then-correct 流水线运行中…
      </div>
    </div>
  );
}
