"""
Suivi des joueurs entre les frames (tracking).
Utilise ByteTrack intégré dans ultralytics.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def track_video(
    video_path: str | Path,
    model_path: str | Path = "models/yolov8n.pt",
    output_path: str | Path | None = None,
    conf: float = 0.4,
    tracker: str = "bytetrack.yaml",
) -> list[dict]:
    """
    Suit les joueurs dans une vidéo avec ByteTrack.

    Returns:
        Liste de dicts {frame, tracks} où chaque track a un id unique.
    """
    model = YOLO(str(model_path))

    results_all = []
    for frame_idx, result in enumerate(
        model.track(
            source=str(video_path),
            conf=conf,
            classes=[0],
            tracker=tracker,
            save=bool(output_path),
            project=str(Path(output_path).parent) if output_path else None,
            name=Path(output_path).stem if output_path else None,
            stream=True,
            verbose=False,
        )
    ):
        tracks = []
        if result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                tracks.append(
                    {
                        "id": int(track_id),
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                    }
                )
        results_all.append({"frame": frame_idx, "tracks": tracks})

    logger.info("Tracking terminé : %d frames", len(results_all))
    return results_all
