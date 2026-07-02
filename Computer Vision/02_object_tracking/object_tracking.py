import argparse
import urllib.request
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

_VTEST_URL: str = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
)


class ObjectTracker:
    MODEL_NAME: str = "yolo11n.pt"      # YOLO11 nano — ~2.6 MB, downloads automatically
    DATA_DIR: str = "data"
    VIDEO_URL: str = _VTEST_URL
    CONF_THRESHOLD: float = 0.3

    def __init__(self) -> None:
        self.model: YOLO | None = None
        self._trajectories: dict[int, list[tuple[float, float]]] = {}

    def load_model(self) -> None:
        self.model = YOLO(self.MODEL_NAME)  # weights auto-downloaded to ~/.cache/ultralytics

    def _download_video(self) -> Path:
        dest = Path(self.DATA_DIR) / "vtest.avi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print("  Downloading sample video (vtest.avi, ~2.6 MB)...")
            urllib.request.urlretrieve(self.VIDEO_URL, dest)
        return dest

    def track(self, video_path: Path, output_path: Path) -> None:
        self._trajectories = {}

        cap = cv2.VideoCapture(str(video_path))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        cap = cv2.VideoCapture(str(video_path))
        for _ in tqdm(range(total), desc="Tracking frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,               # keep track IDs consistent across frames
                conf=self.CONF_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # ultralytics draws boxes + class labels + track IDs automatically
            annotated = results[0].plot()

            # Collect centroid per track ID for the trajectory plot
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()   # (N, 4) cx, cy, w, h
                ids   = results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    cx, cy = float(box[0]), float(box[1])
                    self._trajectories.setdefault(track_id, []).append((cx, cy))

            writer.write(annotated)

        cap.release()
        writer.release()
        print(f"  Annotated video saved to {output_path}")
        print(f"  Tracked {len(self._trajectories)} unique object(s) across {total} frames.")

    def save_trajectory_plot(self, output_dir: Path) -> None:
        active = {tid: pts for tid, pts in self._trajectories.items() if len(pts) >= 5}
        if not active:
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.get_cmap("tab20")

        for i, (track_id, centers) in enumerate(active.items()):
            xs = np.array([c[0] for c in centers])
            ys = np.array([c[1] for c in centers])
            color = cmap(i % 20)
            ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.8, label=f"ID {track_id}")
            ax.plot(xs[0],  ys[0],  "o", color=color, markersize=5)
            ax.plot(xs[-1], ys[-1], "s", color=color, markersize=5)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(f"Object Trajectories — YOLO + ByteTrack ({len(active)} tracks)")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8, ncol=3)

        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "trajectory.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Trajectory plot saved to {path}")

    def run(
        self,
        video_path: Path | None = None,
        output_video: Path = Path("plots/tracked_output.mp4"),
        save_plot: bool = True,
    ) -> None:
        if video_path is None:
            video_path = self._download_video()
        self.track(video_path, output_video)
        if save_plot:
            self.save_trajectory_plot(output_video.parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-object tracking — YOLO11 detection + ByteTrack"
    )
    parser.add_argument(
        "--video-path", type=str, default=None,
        help="path to a video file (default: downloads vtest.avi from OpenCV samples)",
    )
    parser.add_argument(
        "--output-video", type=str, default="plots/tracked_output.mp4",
        help="path for the annotated output video (default: plots/tracked_output.mp4)",
    )
    parser.add_argument(
        "--model", type=str, default=ObjectTracker.MODEL_NAME,
        help=f"YOLO model weights (default: {ObjectTracker.MODEL_NAME})",
    )
    parser.add_argument(
        "--conf", type=float, default=ObjectTracker.CONF_THRESHOLD,
        help=f"detection confidence threshold (default: {ObjectTracker.CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="skip saving the trajectory plot",
    )
    args = parser.parse_args()

    tracker = ObjectTracker()
    tracker.MODEL_NAME = args.model
    tracker.CONF_THRESHOLD = args.conf

    print("[1/2] Loading YOLO11 model (downloads ~2.6 MB on first run)...")
    tracker.load_model()

    print("[2/2] Tracking objects in video...")
    tracker.run(
        video_path=Path(args.video_path) if args.video_path else None,
        output_video=Path(args.output_video),
        save_plot=not args.no_plot,
    )
    print("Done.")


if __name__ == "__main__":
    main()
