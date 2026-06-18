"use client";

import type { ModalityOutput } from "@/lib/types";
import { DistributionBars } from "./Bars";

const MODALITY_ZH: Record<string, string> = {
  video: "视频",
  audio: "音频",
  text: "文本",
};

const QUALITY_FIELDS: Record<string, [keyof ModalityOutput["quality"], string][]> = {
  video: [
    ["face_detection_rate", "人脸检出率"],
    ["brightness", "亮度"],
    ["blur", "清晰度(Lap方差)"],
    ["frame_loss_rate", "丢帧率"],
  ],
  audio: [
    ["snr_db", "信噪比(dB)"],
    ["voiced_ratio", "有声比例"],
    ["silence_ratio", "静音比例"],
  ],
  text: [
    ["text_length", "长度"],
    ["valid_token_ratio", "有效token比例"],
    ["low_freq_subword_ratio", "低频子词比例"],
  ],
};

export function ModalityCard({ output }: { output: ModalityOutput }) {
  const fields = QUALITY_FIELDS[output.modality] ?? [];
  return (
    <div className={"card modality" + (output.available ? "" : " off")}>
      <div className="card-head">
        <strong>{MODALITY_ZH[output.modality]}</strong>
        <span className={"badge " + (output.available ? "ok" : "muted")}>
          {output.available ? "可用" : "缺失/降级"}
        </span>
        <span className={"badge " + (output.backend.startsWith("trained") ? "real" : "stub")}>
          {output.backend.startsWith("trained") ? "真模型" : "桩"}
        </span>
      </div>

      <div className="quality">
        {fields.map(([key, label]) => {
          const v = output.quality[key];
          return (
            <div className="q-row" key={String(key)}>
              <span className="q-label">{label}</span>
              <span className="mono">{v == null ? "—" : v}</span>
            </div>
          );
        })}
      </div>

      <div className="modality-dist">
        <DistributionBars distribution={output.distribution} compact />
      </div>
    </div>
  );
}
