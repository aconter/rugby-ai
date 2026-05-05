# Rugby AI — Pipeline d'analyse vidéo

Pipeline de détection et tracking de joueurs de rugby via YOLOv8.

## Architecture

```
src/
  run.py        # CLI principal
  detect.py     # Détection par frame (sans tracking)
  tracker.py    # Tracking avec IDs persistants (ByteTrack)
```

## Utilisation via GitHub Actions

Workflow **Analyser une vidéo** (recommandé) :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `video` | — | Nom du fichier dans `~/rugby-videos/` |
| `mode` | `track` | `track` = IDs persistants / `detect` = boîtes par frame |
| `fps` | `5` | Réduire le FPS avant analyse (`0` = vidéo originale) |
| `model` | `yolov8n.pt` | Modèle YOLO (`n`=rapide → `x`=précis) |
| `classes` | `0` | Classes COCO : `0`=joueurs `32`=ballon |
| `export_json` | `false` | Exporter les détections en JSON |
| `no_video` | `false` | Ne pas générer la vidéo annotée |

## Utilisation en CLI

```bash
python3.12 run.py <video> [options]

# Exemples
python3.12 run.py /data/raw_videos/match.mp4
python3.12 run.py /data/raw_videos/match.mp4 --mode track --conf 0.4
python3.12 run.py /data/raw_videos/match.mp4 --classes 0 32 --json
python3.12 run.py /data/raw_videos/match.mp4 --no-video --json
```

## Paramètres CLI

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--mode` | `track` | `detect` ou `track` |
| `--model` | `yolov8n.pt` | Chemin vers le modèle `.pt` |
| `--conf` | `0.4` | Seuil de confiance (0-1) |
| `--classes` | `0` | Classes COCO à détecter |
| `--output` | `/data/outputs` | Dossier de sortie |
| `--no-video` | `false` | Ne pas enregistrer la vidéo annotée |
| `--json` | `false` | Exporter les détections en JSON |

## Format du JSON de sortie

```json
[
  {
    "frame": 0,
    "tracks": [
      { "id": 1, "bbox": [x1, y1, x2, y2], "conf": 0.87 },
      { "id": 2, "bbox": [x1, y1, x2, y2], "conf": 0.91 }
    ]
  }
]
```

## Classes COCO utiles

| ID | Classe |
|----|--------|
| `0` | person (joueurs, arbitres) |
| `32` | sports ball (ballon — détection approximative pour ballon ovale) |

## Modèles disponibles

| Modèle | Vitesse | Précision |
|--------|---------|-----------|
| `yolov8n.pt` | ⚡⚡⚡⚡ | ⭐⭐ |
| `yolov8s.pt` | ⚡⚡⚡ | ⭐⭐⭐ |
| `yolov8m.pt` | ⚡⚡ | ⭐⭐⭐⭐ |
| `yolov8x.pt` | ⚡ | ⭐⭐⭐⭐⭐ |
