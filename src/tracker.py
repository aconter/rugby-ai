"""
Suivi des joueurs entre les frames (tracking).
Utilise ByteTrack intégré dans ultralytics.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_total_frames(video_path: str | Path) -> int:
    """Retourne le nombre total de frames d'une vidéo."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


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

    total_frames, fps = get_total_frames(video_path)
    duration_s = total_frames / fps if fps > 0 else 0
    duration_str = f"{int(duration_s // 60)}m{int(duration_s % 60):02d}s"

    print(f"\n  Vidéo : {total_frames} frames · {fps:.1f} fps · durée {duration_str}")
    print(f"  Modèle chargé : {Path(str(model_path)).name}\n")

    results_all = []
    t_start = time.time()

    progress = tqdm(
        total=total_frames,
        unit="frame",
        ncols=80,
        bar_format=(
            "  {l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining} · {rate_fmt}]"
        ),
        colour="green",
    )

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
        progress.set_postfix({"joueurs": len(tracks)}, refresh=False)
        progress.update(1)

    progress.close()
    elapsed = time.time() - t_start
    logger.info("Tracking terminé : %d frames en %.1fs", len(results_all), elapsed)
    return results_all
