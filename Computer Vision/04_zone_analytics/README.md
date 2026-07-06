# Zone Analytics — Offense Zone Monitoring

People tracking with spatial awareness. Every person in a video is detected and assigned a persistent track ID. The scene is divided into named **offense zones** — areas where people should not be. Anyone whose **feet** (bottom-centre of bounding box) land inside an offense zone gets a red `OFFENSE #id` box and a red trajectory line drawn. Everyone else gets a thin grey box so the full crowd is still visible.

## Dataset

**Oxford TownCentre** — University of Oxford, research licence.  
Fixed overhead street camera recording a busy UK pedestrian shopping street. Ideal for zone analytics because the camera is completely stationary and the overhead angle gives an accurate ground-plane view.

- Download page: [megapixels.cc/oxford_town_centre](https://megapixels.cc/oxford_town_centre/)
- Place the file at `data/TownCentreXVID.mp4` (or pass `--video <path>`)

## How it works

1. **YOLO11n** detects every person in the frame and assigns each one a persistent track ID via **ByteTrack**.
2. The **feet position** (bottom-centre of the bounding box, i.e. where the person actually stands on the ground) is tested against each zone polygon using `cv2.pointPolygonTest`.
3. Using feet rather than the bounding-box centroid gives accurate zone detection under perspective and occlusion.
4. Offense-zone people get a **red box + OFFENSE label + red trajectory line**; everyone else gets a subtle grey box.
5. A **density alert** is shown on-screen when the offense count exceeds `max_capacity`.

## Zones

Zones are arbitrary polygons defined in `config.yaml` with normalised coordinates (0–1).  
The config separates zone shapes (`zones:`) from which zones are monitored (`offense_zones:`).

```yaml
zones:
  "Central Street":
    - [0.29, 0.22]   # top-left
    - [0.70, 0.22]   # top-right
    - [0.74, 1.00]   # bottom-right  (slightly wider — perspective taper)
    - [0.24, 1.00]   # bottom-left

offense_zones:
  - "Central Street"
```

### Interactive zone editor

```bash
python zone_analytics.py --video data/TownCentreXVID.mp4 --define-zones
```

Opens an OpenCV window on the first frame. Click polygon vertices, press **Enter** to confirm each zone and name it in the terminal, then **S** to save. Zones are written back to `config.yaml` automatically.

| Key | Action |
|-----|--------|
| Left click | Add vertex |
| Right click / Backspace | Undo last vertex |
| Enter | Confirm polygon → name it |
| D | Discard current polygon |
| U | Undo last confirmed zone |
| S / Esc | Save & exit |

### Zone preview

```bash
python zone_analytics.py --video data/TownCentreXVID.mp4 --preview
```

Saves `output/zone_preview.png` — the first frame with zone fills and every corner labeled with its pixel coordinate — so you can verify placement and fine-tune `config.yaml` before running the full analysis.

## How to run

```bash
# Full run (processes first 100 frames by default)
python zone_analytics.py --video data/TownCentreXVID.mp4

# Process more frames
python zone_analytics.py --video data/TownCentreXVID.mp4 --max-frames 500

# Process the whole video
python zone_analytics.py --video data/TownCentreXVID.mp4 --max-frames 0

# Override detection confidence
python zone_analytics.py --video data/TownCentreXVID.mp4 --conf 0.4

# Use a custom config
python zone_analytics.py --video data/TownCentreXVID.mp4 --config my_config.yaml
```

## Output

| File | Description |
|------|-------------|
| `output/zone_annotated.mp4` | Annotated video: red boxes on offense-zone people, grey boxes on everyone else, zone outlines, density alerts |
| `output/zone_preview.png` | First-frame zone visualization (only with `--preview`) |

## Code structure

```
zone_analytics.py
├── _resolve_zones(cfg_zones, w, h)      normalised → pixel coords; auto-generates 3 lanes if zones=null
├── _point_in_zone(cx, cy, polygon)      cv2.pointPolygonTest wrapper
├── ZoneAnalytics
│   ├── load_model()                     loads YOLO11n
│   ├── _get_video_path()                resolves video path from config
│   ├── _process_frame(frame, idx)       detect + track + annotate one frame
│   ├── run(video_path, output_dir)      main video loop → writes annotated MP4
│   ├── define_zones_interactive(...)    OpenCV zone editor → saves to config.yaml
│   └── preview_zones(...)              first-frame zone preview → PNG
└── main()                               CLI entry point
```
