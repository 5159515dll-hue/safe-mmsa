// 与 safemmsa/contracts.py 对齐的前端类型（标签顺序固定）。

export const EMOTION_LABELS = [
  "happy",
  "sad",
  "angry",
  "fear",
  "surprise",
  "disgust",
  "neutral",
] as const;

export type EmotionLabel = (typeof EMOTION_LABELS)[number];

export const LABEL_ZH: Record<string, string> = {
  happy: "快乐",
  sad: "悲伤",
  angry: "愤怒",
  fear: "恐惧",
  surprise: "惊讶",
  disgust: "厌恶",
  neutral: "中性",
};

export interface ModalitySignalQuality {
  face_detection_rate: number | null;
  blur: number | null;
  brightness: number | null;
  frame_loss_rate: number | null;
  snr_db: number | null;
  voiced_ratio: number | null;
  silence_ratio: number | null;
  text_length: number | null;
  valid_token_ratio: number | null;
  low_freq_subword_ratio: number | null;
}

export interface ModalityOutput {
  modality: "video" | "audio" | "text";
  available: boolean;
  distribution: Record<string, number> | null;
  logits: number[] | null;
  feature_ref: string | null;
  quality: ModalitySignalQuality;
  backend: string;
}

export interface InitialPrediction {
  distribution: Record<string, number>;
  logits: number[];
  fused_feature_ref: string | null;
  backbone: string;
}

export interface BoundaryState {
  max_confidence: number;
  top2_margin: number;
  norm_entropy: number;
  score: number;
}

export interface CompetitionState {
  top1: string;
  top2: string;
  per_modality_direction: Record<string, number>;
  margin: number;
  entropy: number;
  vector: number[];
}

export interface ReliabilityState {
  per_modality: Record<string, number>;
  quality_part: Record<string, number>;
  uncertainty_part: Record<string, number>;
  consistency_pairwise: Record<string, number>;
}

export interface EnhancementResult {
  final_distribution: Record<string, number>;
  final_logits: number[];
  correction_magnitude: number;
  bounded: boolean;
  enhancement: string;
}

export interface AnalyzeResult {
  language: string;
  modalities: ModalityOutput[];
  initial: InitialPrediction;
  boundary: BoundaryState;
  competition: CompetitionState;
  reliability: ReliabilityState;
  enhancement: EnhancementResult;
  final_emotion: string;
  timings_ms: Record<string, number> | null;
}

export interface ModelsStatus {
  mode: string;
  all_stub: boolean;
  points: {
    encoders: Record<string, string>;
    fusion_backbone: string;
    enhancement: string;
    reliability_head: string;
  };
  note: string;
}
