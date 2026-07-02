# Computer Vision Projects

Three computer vision projects built with PyTorch and OpenCV, covering the core tasks of modern CV: detecting what is in a scene, tracking objects as they move, and segmenting each instance at the pixel level.

All three use pretrained models — no training required. Each one runs out of the box and downloads any required data automatically.

## Projects

| # | Project | Technique | Model |
|---|---------|-----------|-------|
| 01 | [Object Detection](01_object_detection/) | Bounding box prediction | Faster R-CNN (ResNet-50 FPN, COCO) |
| 02 | [Object Tracking](02_object_tracking/) | Multi-object tracking across video frames | YOLO11 + ByteTrack |
| 03 | [Image Segmentation](03_image_segmentation/) | Per-instance pixel masks | Mask R-CNN (ResNet-50 FPN, COCO) |

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
```

Each project downloads what it needs on first run (model weights, sample images or video) and saves its output to a `plots/` subfolder.

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
├── requirements.txt
└── README.md
```
