import argparse
import colorsys
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm

# 91-class COCO label map (index 0 = background, 1–90 = object classes)
COCO_LABELS: list[str] = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Three COCO validation images used as built-in demo samples
_SAMPLE_IMAGES: list[tuple[str, str]] = [
    ("sample_cats.jpg",    "http://images.cocodataset.org/val2017/000000039769.jpg"),
    ("sample_kitchen.jpg", "http://images.cocodataset.org/val2017/000000397133.jpg"),
    ("sample_sports.jpg",  "http://images.cocodataset.org/val2017/000000037777.jpg"),
]


def _distinct_colors(n: int) -> list[tuple[int, int, int]]:
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95))
        for i in range(n)
    ]


class ObjectDetector:
    THRESHOLD: float = 0.5
    DATA_DIR: str = "data"

    def __init__(self) -> None:
        self.model: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        try:
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Faster R-CNN pretrained weights.\n"
                "Check your internet connection and try again."
            ) from exc

    def _download_samples(self) -> Path:
        dest = Path(self.DATA_DIR) / "sample_images"
        dest.mkdir(parents=True, exist_ok=True)
        for filename, url in _SAMPLE_IMAGES:
            path = dest / filename
            if not path.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, path)
        return dest

    def predict(self, image: Image.Image) -> dict:
        tensor = TF.to_tensor(image).to(self.device)
        with torch.no_grad():
            output = self.model([tensor])[0]
        keep = output["scores"] >= self.THRESHOLD
        return {
            "boxes":  output["boxes"][keep].cpu(),
            "labels": output["labels"][keep].cpu(),
            "scores": output["scores"][keep].cpu(),
        }

    def annotate(self, image: Image.Image, predictions: dict) -> Image.Image:
        img = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        colors = _distinct_colors(len(predictions["labels"]))
        for i, (box, label, score) in enumerate(
            zip(predictions["boxes"], predictions["labels"], predictions["scores"])
        ):
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            r, g, b = colors[i]
            draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 255), width=3)
            tag = f"{COCO_LABELS[label.item()]} {score.item():.0%}"
            draw.rectangle([x0, y0 - 16, x0 + len(tag) * 7, y0], fill=(r, g, b, 200))
            draw.text((x0 + 2, y0 - 14), tag, fill=(255, 255, 255, 255), font=font)
        return Image.alpha_composite(img, overlay).convert("RGB")

    def run(self, images_dir: Path) -> None:
        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            print(f"No images found in {images_dir}.")
            return
        for p in tqdm(paths, desc="Detecting objects", unit="image"):
            image = Image.open(p).convert("RGB")
            preds = self.predict(image)
            annotated = self.annotate(image, preds)
            out = plots / f"detection_{p.stem}.png"
            annotated.save(out)
            n = len(preds["labels"])
            labels = [COCO_LABELS[lbl.item()] for lbl in preds["labels"]]
            print(
                f"  {p.name}: {n} detection{'s' if n != 1 else ''} "
                f"({', '.join(dict.fromkeys(labels))}) → {out.name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Object detection — Faster R-CNN pretrained on COCO"
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="directory of images to run detection on (default: downloads 3 COCO samples)",
    )
    parser.add_argument(
        "--threshold", type=float, default=ObjectDetector.THRESHOLD,
        help=f"confidence score threshold (default: {ObjectDetector.THRESHOLD})",
    )
    args = parser.parse_args()

    detector = ObjectDetector()
    detector.THRESHOLD = args.threshold

    print("[1/3] Loading Faster R-CNN (COCO pretrained — downloads weights on first run)...")
    detector.load_model()

    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        print("[2/3] Downloading 3 sample images from the COCO validation set...")
        images_dir = detector._download_samples()

    print(f"[3/3] Running detection on images in {images_dir}...")
    detector.run(images_dir)
    print("Done. Annotated images saved to plots/")


if __name__ == "__main__":
    main()
