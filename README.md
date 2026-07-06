# Computer Vision Projects

Four computer vision projects built with PyTorch, OpenCV, and Ultralytics, covering the core tasks of modern CV: detecting what is in a scene, tracking objects as they move, segmenting each instance at the pixel level, and monitoring restricted zones in a crowd.

All three use pretrained models — no training required. Each one runs out of the box and downloads any required data automatically.

## Projects

| # | Project | Technique | Model |
|---|---------|-----------|-------|
| 01 | [Object Detection](01_object_detection/) | Bounding box prediction | Faster R-CNN (ResNet-50 FPN, COCO) |
| 02 | [Object Tracking](02_object_tracking/) | Multi-object tracking across video frames | YOLO11 + ByteTrack |
| 03 | [Image Segmentation](03_image_segmentation/) | Per-instance pixel masks | Mask R-CNN (ResNet-50 FPN, COCO) |
| 04 | [Zone Analytics](04_zone_analytics/) | Offense-zone people tracking: flag anyone entering a restricted area | YOLO11n + ByteTrack + feet-based zone detection |

## Setup

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> PyTorch CPU wheels are not on PyPI. If `pip install torch torchvision` fails, get the correct install command for your platform at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

## Running a project

```bash
cd 01_object_detection
python object_detection.py

cd 02_object_tracking
python object_tracking.py

cd 03_image_segmentation
python image_segmentation.py

cd 04_zone_analytics
# Draw your offense zones first (optional but recommended):
python zone_analytics.py --video data/TownCentreXVID.mp4 --define-zones
# Then run:
python zone_analytics.py --video data/TownCentreXVID.mp4
```

Projects 01–03 save output to a `plots/` subfolder. Project 04 saves to `output/`.

> **Oxford TownCentre** (project 04): must be downloaded manually from [megapixels.cc/oxford_town_centre](https://megapixels.cc/oxford_town_centre/) and placed at `04_zone_analytics/data/TownCentreXVID.mp4`.

## Repository structure

```
Computer Vision/
├── 01_object_detection/
│   ├── object_detection.py
│   └── README.md
├── 02_object_tracking/
│   ├── object_tracking.py
│   └── README.md
├── 03_image_segmentation/
│   ├── image_segmentation.py
│   └── README.md
├── 04_zone_analytics/
│   ├── zone_analytics.py
│   ├── config.yaml
│   └── README.md
├── requirements.txt
└── README.md
```
