"""
Détection des joueurs de rugby sur une vidéo via YOLOv8.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def detect_video(
    video_path: str | Path,
    model_path: str | Path = "models/yolov8n.pt",
    output_path: str | Path | None = None,
    conf: float = 0.4,
    classes: list[int] | None = None,
) -> list[dict]:
    """
    Détecte les joueurs dans chaque frame d'une vidéo.

    Args:
        video_path: Chemin vers la vidéo source.
        model_path: Chemin vers le modèle YOLO (.pt).
        output_path: Si fourni, enregistre la vidéo annotée.
        conf: Seuil de confiance (0-1).
        classes: Liste de classes COCO à détecter (0 = person par défaut).

    Returns:
        Liste de dicts {frame, detections} par frame.
    """
    if classes is None:
        classes = [0]  # 0 = person dans COCO

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    results_all = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, classes=classes, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "class": int(box.cls[0]),
                    }
                )

        results_all.append({"frame": frame_idx, "detections": detections})

        if writer:
            annotated = results[0].plot()
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    logger.info("Détection terminée : %d frames, vidéo=%s", frame_idx, video_path)
    return results_all
