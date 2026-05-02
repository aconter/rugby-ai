"""
CLI Rugby AI — Détection et tracking YOLOv8 sur une vidéo de rugby.

Usage:
    python3.12 run.py <video> [options]

Exemples:
    python3.12 run.py /data/raw_videos/match.mp4
    python3.12 run.py /data/raw_videos/match.mp4 --mode track --conf 0.35
    python3.12 run.py /data/raw_videos/match.mp4 --classes 0 32 --output /data/outputs/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Classes COCO utiles pour le rugby
COCO_CLASSES = {
    0: "person",
    32: "sports ball",
    38: "tennis racket",  # parfois confondu avec ballon ovale
}


def print_banner() -> None:
    print("""
╔══════════════════════════════════════╗
║        Rugby AI — Baseline YOLO      ║
║   Modèle : YOLOv8 pré-entraîné COCO  ║
╚══════════════════════════════════════╝
""")


def print_stats(results: list[dict], elapsed: float, mode: str) -> None:
    """Affiche un résumé des détections."""
    total_frames = len(results)
    key = "tracks" if mode == "track" else "detections"

    counts_per_frame = [len(r[key]) for r in results]
    avg = sum(counts_per_frame) / total_frames if total_frames else 0
    max_count = max(counts_per_frame) if counts_per_frame else 0
    fps_proc = total_frames / elapsed if elapsed > 0 else 0

    print("\n" + "═" * 42)
    print("  RÉSULTATS")
    print("═" * 42)
    print(f"  Frames analysées  : {total_frames}")
    print(f"  Vitesse traitement : {fps_proc:.1f} fps")
    print(f"  Temps total        : {elapsed:.1f}s")
    print(f"  Détections/frame   : {avg:.1f} en moyenne")
    print(f"  Max simultané      : {max_count} personnes")

    # Distribution des détections
    buckets = {
        "0 joueur     ": sum(1 for c in counts_per_frame if c == 0),
        "1-5 joueurs  ": sum(1 for c in counts_per_frame if 1 <= c <= 5),
        "6-10 joueurs ": sum(1 for c in counts_per_frame if 6 <= c <= 10),
        "11-15 joueurs": sum(1 for c in counts_per_frame if 11 <= c <= 15),
        "16+ joueurs  ": sum(1 for c in counts_per_frame if c > 15),
    }
    print("\n  Distribution :")
    for label, count in buckets.items():
        pct = count / total_frames * 100 if total_frames else 0
        bar = "█" * int(pct / 2)
        print(f"    {label} : {bar} {count} frames ({pct:.0f}%)")
    print("═" * 42)


def main() -> int:
    print_banner()

    parser = argparse.ArgumentParser(
        description="Détection YOLOv8 sur une vidéo de rugby"
    )
    parser.add_argument("video", help="Chemin vers la vidéo d'entrée")
    parser.add_argument(
        "--mode",
        choices=["detect", "track"],
        default="track",
        help="detect = boîtes par frame | track = IDs persistants (défaut: track)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Modèle YOLO : yolov8n.pt (rapide) | yolov8s.pt | yolov8m.pt | yolov8x.pt (précis)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance 0-1 (défaut: 0.35)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0],
        help="Classes COCO à détecter. 0=personne 32=ballon (défaut: 0)",
    )
    parser.add_argument(
        "--output",
        default="/data/outputs",
        help="Dossier de sortie (défaut: /data/outputs)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Ne pas enregistrer la vidéo annotée (plus rapide)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Exporter les détections en JSON",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Vidéo introuvable : %s", video_path)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    output_video = output_dir / f"{stem}_{args.mode}_annotated.mp4" if not args.no_video else None
    output_json = output_dir / f"{stem}_{args.mode}_results.json" if args.json else None

    classes_str = ", ".join(
        COCO_CLASSES.get(c, f"class_{c}") for c in args.classes
    )
    logger.info("Vidéo       : %s", video_path)
    logger.info("Mode        : %s", args.mode)
    logger.info("Modèle      : %s", args.model)
    logger.info("Confiance   : %.2f", args.conf)
    logger.info("Classes     : %s", classes_str)
    if output_video:
        logger.info("Sortie vidéo: %s", output_video)

    start = time.time()

    if args.mode == "detect":
        from detect import detect_video
        results = detect_video(
            video_path=video_path,
            model_path=args.model,
            output_path=output_video,
            conf=args.conf,
            classes=args.classes,
        )
    else:
        from tracker import track_video
        results = track_video(
            video_path=video_path,
            model_path=args.model,
            output_path=output_video,
            conf=args.conf,
        )

    elapsed = time.time() - start
    print_stats(results, elapsed, args.mode)

    if output_json and results:
        output_json.write_text(json.dumps(results, indent=2))
        logger.info("JSON exporté : %s", output_json)

    if output_video and output_video.exists():
        logger.info("Vidéo annotée : %s", output_video)

    return 0


if __name__ == "__main__":
    sys.exit(main())
