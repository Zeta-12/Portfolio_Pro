import argparse
import colorsys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
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


class ImageSegmenter:
    THRESHOLD: float = 0.5
    ALPHA: float = 0.5
    DATA_DIR: str = "data"

    def __init__(self) -> None:
        self.model: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        try:
            self.model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Mask R-CNN pretrained weights.\n"
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
            "masks":  output["masks"][keep].squeeze(1).cpu(),  # (N, H, W) float [0, 1]
            "labels": output["labels"][keep].cpu(),
            "scores": output["scores"][keep].cpu(),
        }

    def overlay_masks(self, image: Image.Image, predictions: dict) -> Image.Image:
        img = image.copy().convert("RGBA")
        font = ImageFont.load_default()
        colors = _distinct_colors(len(predictions["labels"]))
        for i, (mask, label, score) in enumerate(
            zip(predictions["masks"], predictions["labels"], predictions["scores"])
        ):
            binary = (mask.numpy() > 0.5).astype(np.uint8)  # (H, W)
            r, g, b = colors[i]
            alpha = int(self.ALPHA * 255)
            color_layer = Image.new("RGBA", img.size, (r, g, b, alpha))
            mask_pil = Image.fromarray(binary * 255, mode="L")
            img.paste(color_layer, mask=mask_pil)
            # Place label at the centroid of the mask
            ys, xs = np.where(binary > 0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                tag = f"{COCO_LABELS[label.item()]} {score.item():.0%}"
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [cx, cy - 14, cx + len(tag) * 7, cy + 2],
                    fill=(r, g, b, 200),
                )
                draw.text((cx + 2, cy - 12), tag, fill=(255, 255, 255, 255), font=font)
        return img.convert("RGB")

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
        for p in tqdm(paths, desc="Segmenting images", unit="image"):
            image = Image.open(p).convert("RGB")
            preds = self.predict(image)
            annotated = self.overlay_masks(image, preds)
            out = plots / f"segmentation_{p.stem}.png"
            annotated.save(out)
            n = len(preds["labels"])
            labels = [COCO_LABELS[lbl.item()] for lbl in preds["labels"]]
            print(
                f"  {p.name}: {n} instance{'s' if n != 1 else ''} "
                f"({', '.join(dict.fromkeys(labels))}) → {out.name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instance segmentation — Mask R-CNN pretrained on COCO"
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="directory of images to segment (default: downloads 3 COCO samples)",
    )
    parser.add_argument(
        "--threshold", type=float, default=ImageSegmenter.THRESHOLD,
        help=f"confidence score threshold (default: {ImageSegmenter.THRESHOLD})",
    )
    parser.add_argument(
        "--alpha", type=float, default=ImageSegmenter.ALPHA,
        help=f"mask overlay transparency 0–1 (default: {ImageSegmenter.ALPHA})",
    )
    args = parser.parse_args()

    segmenter = ImageSegmenter()
    segmenter.THRESHOLD = args.threshold
    segmenter.ALPHA = args.alpha

    print("[1/3] Loading Mask R-CNN (COCO pretrained — downloads weights on first run)...")
    segmenter.load_model()

    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        print("[2/3] Downloading 3 sample images from the COCO validation set...")
        images_dir = segmenter._download_samples()

    print(f"[3/3] Running segmentation on images in {images_dir}...")
    segmenter.run(images_dir)
    print("Done. Annotated images saved to plots/")


if __name__ == "__main__":
    main()
