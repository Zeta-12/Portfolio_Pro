"""Zone Analytics — Crowd Flow and Occupancy Monitoring
========================================================
Dataset : Oxford TownCentre Dataset (University of Oxford, research licence)
          Fixed overhead street camera: stationary viewpoint ideal for zone drawing,
          thousands of annotated pedestrian tracks across a busy UK town centre.
          Download: https://megapixels.cc/oxford_town_centre/ → place at data/TownCentreXVID.mp4

Pipeline
--------
  1. YOLO11n detects every person per frame with persistent track IDs.
  2. The feet position (bottom-centre of bounding box) is tested against zone polygons.
  3. Offense-zone people: red box + trajectory line.
     Everyone else: thin grey box.
  4. Density alert shown in video when offense count exceeds max_capacity.

Output (output/)
-----------------
  zone_annotated.mp4      Annotated video: red boxes on offense-zone people,
                          grey boxes on all others, zone outlines, density alerts

Usage
-----
  python zone_analytics.py --video path/to.mp4  # run analysis
  python zone_analytics.py --video v.mp4 --define-zones  # interactive zone editor
  python zone_analytics.py --video v.mp4 --preview       # verify zone placement
  python zone_analytics.py --max-frames 500     # quick demo (default: 100)
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Distinct BGR colours for up to 8 zones
_ZONE_COLOURS: list[tuple[int, int, int]] = [
    (255, 120,  60), ( 60, 200,  60), ( 60,  60, 240), (200,  60, 200),
    ( 60, 200, 200), (255, 200,   0), (255,  60, 120), (120, 120, 255),
]

# Oxford TownCentre video — expected local path
# Download from: https://megapixels.cc/oxford_town_centre/
_OXFORD_PRIMARY = "data/TownCentreXVID.mp4"


# ── fallback config ────────────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "model": {"name": "yolo11n.pt", "conf": 0.30, "person_class_id": 0},
    "tracker": {"algorithm": "bytetrack.yaml", "persist": True},
    "zones": None,   # auto-generate from frame dimensions
    "max_capacity": 8,
    "output_dir": "output",
    "data_dir":   "data",
    "frame_skip": 2,
    "max_frames": 100,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str | None) -> dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).with_name("config.yaml")
    if cfg_path.exists():
        with cfg_path.open() as f:
            return yaml.safe_load(f)
    log.warning("config.yaml not found — using built-in defaults.")
    return _DEFAULT_CONFIG


def _auto_zones(w: int, h: int) -> dict[str, list[list[int]]]:
    """Generate three equal vertical lanes when no zones are configured."""
    return {
        "Zone A": [[0, 0], [w//3, 0], [w//3, h], [0, h]],
        "Zone B": [[w//3, 0], [2*w//3, 0], [2*w//3, h], [w//3, h]],
        "Zone C": [[2*w//3, 0], [w, 0], [w, h], [2*w//3, h]],
    }


def _resolve_zones(cfg_zones, w: int, h: int) -> dict[str, list[list[int]]]:
    """Convert normalised (0-1) or absolute zone coords to pixel coords."""
    if not cfg_zones:
        return _auto_zones(w, h)
    result: dict[str, list[list[int]]] = {}
    for name, pts in cfg_zones.items():
        pixel_pts = []
        for x, y in pts:
            # Detect normalised (all values <= 1.0) vs absolute
            px = int(x * w) if x <= 1.0 else int(x)
            py = int(y * h) if y <= 1.0 else int(y)
            pixel_pts.append([px, py])
        result[name] = pixel_pts
    return result


def _point_in_zone(cx: float, cy: float, polygon: list[list[int]]) -> bool:
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0


# ── main class ─────────────────────────────────────────────────────────────────

class ZoneAnalytics:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg        = cfg
        self.model: YOLO | None = None
        self.zones: dict[str, list[list[int]]] = {}  # set after first frame

        # Accumulators
        self._trajectories: dict[int, list[tuple[int, int]]] = {}  # track_id -> [feet positions]

    # ── model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        name = self.cfg["model"]["name"]
        log.info("Loading YOLO model: %s", name)
        self.model = YOLO(name)

    # ── video acquisition ─────────────────────────────────────────────────────

    def _get_video_path(self) -> Path:
        """Return the Oxford TownCentre video path; raises if not found."""
        dest = Path(self.cfg.get("video", {}).get("path", _OXFORD_PRIMARY)
                    if isinstance(self.cfg.get("video"), dict)
                    else _OXFORD_PRIMARY)
        if dest.exists():
            return dest
        raise RuntimeError(
            f"Oxford TownCentre video not found at '{dest}'.\n"
            f"Download TownCentreXVID.mp4 from:\n"
            f"  https://megapixels.cc/oxford_town_centre/\n"
            f"Save to '{dest}', or pass --video <path>."
        )

    # ── per-frame processing ───────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Detect + track people, annotate frame, return it."""
        mcfg    = self.cfg["model"]
        tcfg    = self.cfg.get("tracker", {})
        results = self.model.track(
            frame,
            conf=float(mcfg["conf"]),
            classes=[int(mcfg.get("person_class_id", 0))],
            persist=tcfg.get("persist", True),
            tracker=tcfg.get("algorithm", "bytetrack.yaml"),
            verbose=False,
        )

        annotated     = frame.copy()
        zone_counts   = {name: 0 for name in self.zones}
        max_cap       = int(self.cfg.get("max_capacity", 8))
        offense_zones = set(self.cfg.get("offense_zones", []))

        # ── Zone boundaries ──────────────────────────────────────────────────
        # Offense zones: thick red outline + label.
        # All other zones: thin grey outline only (no fill anywhere).
        for name, poly in self.zones.items():
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            if name in offense_zones:
                cv2.polylines(annotated, [pts], True, (0, 0, 220), 2)
                cv2.putText(annotated, name, (poly[0][0] + 6, poly[0][1] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
            else:
                cv2.polylines(annotated, [pts], True, (140, 140, 140), 1)
                cv2.putText(annotated, name, (poly[0][0] + 5, poly[0][1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                feet_x = (x1 + x2) / 2   # bottom-centre = feet ground position
                feet_y = float(y2)

                # Analytics: record feet position for trajectory
                if tid not in self._trajectories:
                    self._trajectories[tid] = []
                self._trajectories[tid].append((int(feet_x), int(feet_y)))
                for zname in self.zones:
                    if _point_in_zone(feet_x, feet_y, self.zones[zname]):
                        zone_counts[zname] += 1

                # Offense test: are the person's FEET inside an offense zone?
                in_offense = any(
                    _point_in_zone(feet_x, feet_y, self.zones[z])
                    for z in offense_zones if z in self.zones
                )

                if in_offense:
                    # Red trajectory line
                    traj = self._trajectories[tid]
                    if len(traj) > 1:
                        for j in range(1, len(traj)):
                            cv2.line(annotated, traj[j-1], traj[j], (0, 60, 255), 1)
                    # Red bounding box + OFFENSE label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
                    label = f"OFFENSE #{tid}"
                    lw    = len(label) * 9
                    cv2.rectangle(annotated, (x1, max(y1 - 22, 0)), (x1 + lw, y1),
                                  (0, 0, 220), -1)
                    cv2.putText(annotated, label, (x1 + 3, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                else:
                    # Thin grey box for everyone else (always visible)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)
                    cv2.putText(annotated, f"#{tid}", (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                # Feet dot — small marker at the test point
                cv2.circle(annotated, (int(feet_x), int(feet_y)), 3,
                           (0, 0, 220) if in_offense else (180, 180, 180), -1)

        # Density alert based on offense-zone count only
        offense_count = sum(zone_counts.get(z, 0) for z in offense_zones)
        if offense_count > max_cap:
            msg = f"ALERT: {offense_count} people in offense zone"
            cv2.putText(annotated, msg, (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Frame info overlay
        cv2.putText(annotated, f"Frame {frame_idx}  Offenses: {offense_count}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    # ── zone definition tool ───────────────────────────────────────────────────

    def define_zones_interactive(self, video_path: Path, config_path: Path) -> None:
        """Interactive zone polygon editor using an OpenCV window.

        Controls
        --------
          Left click   Add vertex to current polygon
          Right click  Undo last vertex
          Backspace    Undo last vertex
          Enter        Confirm polygon → type its name in the terminal
          D            Discard current polygon
          U            Undo the last confirmed zone
          S / Esc      Save all zones to config.yaml and exit
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ret, base_frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read first frame from video.")

        H, W = base_frame.shape[:2]
        WIN  = "Define Zones  [Enter=confirm | S/Esc=save]"

        # Mutable state shared with the mouse callback
        state: dict = {"pts": [], "mouse": (0, 0)}
        completed: dict[str, list[list[int]]] = {}

        _COLOURS = [
            ( 50, 200,  50), ( 50,  50, 220), (220, 150,  50),
            (200,  50, 200), ( 50, 200, 200), (220, 200,  50),
        ]

        def on_mouse(event, x, y, flags, _):
            state["mouse"] = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pts"].append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and state["pts"]:
                state["pts"].pop()

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(W, 1280), min(H, 720))
        cv2.setMouseCallback(WIN, on_mouse)

        sep = "=" * 56
        print(f"\n{sep}")
        print("  INTERACTIVE ZONE DEFINITION")
        print(sep)
        print("  Left click    Add vertex")
        print("  Right click   Undo last vertex")
        print("  Backspace     Undo last vertex")
        print("  Enter         Confirm polygon (type name in terminal)")
        print("  D             Discard current polygon")
        print("  U             Undo last confirmed zone")
        print("  S / Esc       Save all zones to config.yaml and exit")
        print(f"{sep}\n")

        while True:
            frame   = base_frame.copy()
            pts     = state["pts"]
            mx, my  = state["mouse"]

            # Draw completed zones
            for idx, (name, poly) in enumerate(completed.items()):
                col    = _COLOURS[idx % len(_COLOURS)]
                np_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                ov     = frame.copy()
                cv2.fillPoly(ov, [np_pts], col)
                cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
                cv2.polylines(frame, [np_pts], True, col, 2)
                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                cv2.putText(frame, name, (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            # Draw polygon being built (lines + dots + rubberband to cursor)
            cur_col = _COLOURS[len(completed) % len(_COLOURS)]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], cur_col, 2)
            if pts:
                cv2.line(frame, pts[-1], (mx, my), cur_col, 1)
                if len(pts) >= 2:
                    cv2.line(frame, (mx, my), pts[0], cur_col, 1)
            for pt in pts:
                cv2.circle(frame, pt, 5, cur_col, -1)

            # HUD
            cv2.rectangle(frame, (0, 0), (W, 52), (15, 15, 15), -1)
            cv2.putText(
                frame,
                f"Zone {len(completed)+1} | pts: {len(pts)} "
                f"| Enter=confirm | D=discard | U=undo | S=save",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1,
            )
            done_str = str(list(completed.keys())) if completed else "(none yet)"
            cv2.putText(frame, f"Done: {done_str}",
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 180, 140), 1)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):            # Enter / Space — confirm
                if len(pts) < 3:
                    print(f"  Need >=3 points (have {len(pts)}). Keep clicking.")
                    continue
                default = f"Zone {len(completed) + 1}"
                name    = input(f"\n  Zone name [{default}]: ").strip() or default
                completed[name] = [list(p) for p in pts]
                state["pts"] = []
                print(f"  OK  '{name}' confirmed ({len(completed[name])} vertices).")

            elif key == 8:                 # Backspace — undo last point
                if pts:
                    print(f"  Removed vertex {state['pts'].pop()}.")

            elif key in (ord('d'), ord('D')):  # D — discard current
                state["pts"] = []
                print("  Current polygon discarded.")

            elif key in (ord('u'), ord('U')):  # U — undo last zone
                if completed:
                    last = list(completed)[-1]
                    del completed[last]
                    print(f"  Zone '{last}' removed.")

            elif key in (ord('s'), ord('S'), 27):  # S / Esc — save & exit
                break

        cv2.destroyAllWindows()

        if not completed:
            print("\n  No zones defined — config.yaml unchanged.")
            return

        # Ask which zones are offense zones (default: ALL)
        zone_names = list(completed.keys())
        print("\n  Which zones are OFFENSE zones? (red box + tracked in video)")
        for i, n in enumerate(zone_names, 1):
            print(f"    {i}. {n}")
        raw = input("  Enter numbers (e.g. 1,3), 'none', or press Enter for ALL: ").strip()
        if raw.lower() == "none":
            offense = []
        elif raw:
            idxs    = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
            offense = [zone_names[i] for i in idxs if 0 <= i < len(zone_names)]
        else:
            offense = zone_names[:]   # default: every defined zone is an offense zone
        print(f"  Offense zones: {offense}")

        # Convert pixel coords to normalised [0-1]
        norm_zones = {
            name: [[round(p[0] / W, 4), round(p[1] / H, 4)] for p in poly]
            for name, poly in completed.items()
        }

        # Load existing config, update zones + offense_zones, write back
        cfg: dict = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        cfg["zones"]         = norm_zones
        cfg["offense_zones"] = offense
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

        print(f"\n  Zones saved to {config_path}")
        print("  Run --preview to verify, then run normally to process the video.")

        # Print YAML snippet for reference
        print("\n  Zone polygons (normalised):")
        for name, pts in norm_zones.items():
            print(f"    {name}:")
            for p in pts:
                print(f"      [{p[0]}, {p[1]}]")

    # ── zone preview ──────────────────────────────────────────────────────────

    def preview_zones(self, video_path: Path, output_dir: Path) -> None:
        """Save the first video frame with zones overlaid.

        Offense zones are shown with a red semi-transparent fill.
        All other zones have a grey fill.
        Every corner point is marked with a yellow dot and its pixel
        coordinate so you can copy exact values back into config.yaml.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read first frame from video.")

        w, h = frame.shape[1], frame.shape[0]
        self.zones = _resolve_zones(self.cfg.get("zones"), w, h)
        offense_zones = set(self.cfg.get("offense_zones", []))
        annotated = frame.copy()

        for name, poly in self.zones.items():
            pts     = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            is_off  = name in offense_zones
            fill    = (0, 0, 180)   if is_off else (90, 90, 90)
            outline = (0, 0, 220)   if is_off else (150, 150, 150)
            alpha   = 0.30          if is_off else 0.18
            # Semi-transparent fill
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], fill)
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            # Outline
            cv2.polylines(annotated, [pts], True, outline, 2)
            # Zone label
            prefix = "[OFFENSE] " if is_off else ""
            cv2.putText(annotated, f"{prefix}{name}",
                        (poly[0][0] + 6, poly[0][1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, outline, 2)

        # Corner dots + pixel-coordinate labels (yellow)
        # These tell you exactly what values to put in config.yaml
        for poly in self.zones.values():
            for px, py in poly:
                cv2.circle(annotated, (px, py), 5, (0, 220, 220), -1)
                label_x = px + 8 if px < w - 120 else px - 115
                label_y = py - 6 if py > 16 else py + 18
                cv2.putText(annotated, f"{px},{py}",
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

        # Header bar
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(annotated,
                    f"ZONE PREVIEW  {w}x{h}  —  adjust zones in config.yaml, re-run --preview to verify",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        output_dir.mkdir(exist_ok=True)
        out = output_dir / "zone_preview.png"
        cv2.imwrite(str(out), annotated)
        log.info("Zone preview -> %s", out)
        log.info("Corner coordinates (x,y pixels) are marked in cyan.")
        log.info("Convert to normalised: x_norm = x / %d,  y_norm = y / %d", w, h)

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self, video_path: Path, output_dir: Path) -> None:
        output_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip      = max(1, int(self.cfg.get("frame_skip", 2)))
        max_f     = int(self.cfg.get("max_frames", 0))
        eff_total = min(total, max_f) if max_f > 0 else total

        # Resolve zones (depends on frame dimensions)
        self.zones = _resolve_zones(self.cfg.get("zones"), w, h)
        log.info("Zones configured: %s", list(self.zones.keys()))

        out_path = output_dir / "zone_annotated.mp4"
        writer   = cv2.VideoWriter(str(out_path),
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps / skip, (w, h))

        frame_idx = 0
        with tqdm(total=eff_total, desc="Analysing video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_f > 0 and frame_idx >= max_f:
                    break
                pbar.update(1)

                if frame_idx % skip != 0:
                    frame_idx += 1
                    continue

                annotated = self._process_frame(frame, frame_idx)
                writer.write(annotated)

                frame_idx += 1

        cap.release()
        writer.release()
        log.info("Annotated video -> %s", out_path)

# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zone Analytics — Crowd Flow and Occupancy Monitoring"
    )
    parser.add_argument("--video",        type=str, default=None,
                        help="Path to input video. Default: auto-detect from config.")
    parser.add_argument("--define-zones",  action="store_true",
                        help="Interactive zone editor: draw polygons on the first frame, "
                             "choose offense zones, save to config.yaml and exit.")
    parser.add_argument("--preview",       action="store_true",
                        help="Save first frame with zones drawn to output/zone_preview.png "
                             "and exit (no model needed). Use this to verify / tune zone coords.")
    parser.add_argument("--config",        type=str, default=None,
                        help="Path to config.yaml.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only the first N frames (default: 100; 0 = all frames).")
    parser.add_argument("--conf",       type=float, default=None,
                        help="Override person detection confidence threshold.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.conf is not None:
        cfg["model"]["conf"] = args.conf

    analytics  = ZoneAnalytics(cfg)
    output_dir = Path(cfg["output_dir"])
    cfg_path   = Path(args.config) if args.config else Path(__file__).with_name("config.yaml")

    if args.video:
        video_path = Path(args.video)
    else:
        log.info("Locating Oxford TownCentre video…")
        video_path = analytics._get_video_path()

    # ── Zone definition mode: no YOLO needed ───────────────────────────────
    if args.define_zones:
        log.info("Zone definition mode — opening interactive editor…")
        analytics.define_zones_interactive(video_path, cfg_path)
        return

    # ── Preview mode: no YOLO needed ────────────────────────────────────
    if args.preview:
        log.info("Preview mode — drawing zones on first frame…")
        analytics.preview_zones(video_path, output_dir)
        log.info("Open %s/zone_preview.png to verify zone placement.", output_dir)
        return

    log.info("[1/3] Loading YOLO model…")
    analytics.load_model()

    log.info("[2/3] Running zone analytics on %s…", video_path)
    analytics.run(video_path, output_dir)
    log.info("Done. All outputs saved to %s/", output_dir)


if __name__ == "__main__":
    main()

