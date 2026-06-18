"""视频信号质量：亮度 / 模糊 / 帧缺失率 / 人脸检测率。

依赖 opencv（可选）。缺失或解码失败时全部降级为 None。**人脸检测率为信号级检测
（haar 级联），不是 FER 情感模型** —— 只衡量"画面里有没有可用的脸"。
"""
from __future__ import annotations

from typing import Optional

from ..contracts import ModalitySignalQuality

# 最多采样的帧数（控制开销；均匀抽样覆盖全片）。
_MAX_SAMPLE_FRAMES = 32


def _maybe_cv2():
    try:  # pragma: no cover - 依赖环境
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def extract_video_quality(path: Optional[str]) -> ModalitySignalQuality:
    """从视频文件算信号质量。无路径/无 opencv/解码失败 → 全 None。"""
    if not path:
        return ModalitySignalQuality()
    cv2 = _maybe_cv2()
    if cv2 is None:  # opencv 未安装 → 质量项留空，系统照常运行。
        return ModalitySignalQuality()

    try:  # pragma: no cover - 依赖环境/IO
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return ModalitySignalQuality()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // _MAX_SAMPLE_FRAMES) if total else 1

        face_cascade = None
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            face_cascade = None

        brightness_vals: list[float] = []
        blur_vals: list[float] = []
        read_ok = 0
        read_attempt = 0
        face_hits = 0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                read_attempt += 1
                read_ok += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_vals.append(float(gray.mean()))
                blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                if face_cascade is not None and not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        face_hits += 1
            idx += 1
            if read_attempt >= _MAX_SAMPLE_FRAMES and step == 1:
                break
        cap.release()

        if read_attempt == 0:
            return ModalitySignalQuality()

        # 帧缺失率：解码声明的总帧数与实际可读帧的差（粗略）。
        frame_loss_rate = None
        if total > 0 and idx > 0:
            frame_loss_rate = round(max(0.0, 1.0 - idx / total), 4)

        return ModalitySignalQuality(
            brightness=round(sum(brightness_vals) / len(brightness_vals), 3),
            blur=round(sum(blur_vals) / len(blur_vals), 3),
            frame_loss_rate=frame_loss_rate,
            face_detection_rate=round(face_hits / read_attempt, 4),
        )
    except Exception:
        return ModalitySignalQuality()
