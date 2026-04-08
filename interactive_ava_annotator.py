from __future__ import annotations

import argparse
import copy
import inspect
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


BBox = Tuple[float, float, float, float]


@dataclass
class Annotation:
    x1: float
    y1: float
    x2: float
    y2: float
    obj_id: Optional[int] = None
    action: Optional[str] = None
    id_source: Optional[str] = None  # input | predicted | manual

    def bbox(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class UndoState:
    frame_no: int
    prev_idx: int
    op: str
    frame_before: List[Annotation]
    suppressed_before: set[Tuple[int, int]]


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_image_dir_for_video(root_dir: Path, video_key: Optional[str]) -> Path:
    """Resolve image directory from rawframes root + csv video key."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {root_dir}")

    image_exts = ImageSequenceSource.IMAGE_EXTS
    direct_images = [p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    if direct_images:
        return root_dir

    subdirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No images or subdirectories found in: {root_dir}")

    candidates = []
    for d in subdirs:
        has_img = any(p.is_file() and p.suffix.lower() in image_exts for p in d.iterdir())
        if has_img:
            candidates.append(d)

    if not candidates:
        raise RuntimeError(f"No image sequences found under: {root_dir}")

    if len(candidates) == 1:
        return candidates[0]

    if not video_key:
        sample = ", ".join([c.name for c in candidates[:8]])
        raise ValueError(
            f"Multiple image subdirectories found in {root_dir}: {sample}. "
            "Please provide --video-key or use a single target subdirectory."
        )

    # 1) Exact match.
    for d in candidates:
        if d.name == video_key:
            return d

    # 2) Normalized match.
    target_norm = _norm_name(video_key)
    for d in candidates:
        if _norm_name(d.name) == target_norm:
            return d

    # 3) Loose containment match.
    loose = [d for d in candidates if target_norm in _norm_name(d.name) or _norm_name(d.name) in target_norm]
    if len(loose) == 1:
        return loose[0]

    sample = ", ".join([c.name for c in candidates[:8]])
    raise ValueError(
        f"Could not map video key '{video_key}' to an image subdirectory under {root_dir}. "
        f"Candidates: {sample}"
    )


class FrameSource:
    def __len__(self) -> int:
        raise NotImplementedError

    def read(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def frame_number(self, idx: int) -> int:
        raise NotImplementedError

    def size(self) -> Tuple[int, int]:
        raise NotImplementedError


class VideoSource(FrameSource):
    def __init__(self, video_path: Path, frame_base: int = 1):
        self.video_path = str(video_path)
        self.frame_base = frame_base
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self) -> int:
        return self.frame_count

    def read(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(idx)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame index {idx}")
        return frame

    def frame_number(self, idx: int) -> int:
        return idx + self.frame_base

    def size(self) -> Tuple[int, int]:
        return self.width, self.height


class ImageSequenceSource(FrameSource):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, image_dir: Path, frame_base: int = 1, frame_from_filename: bool = True):
        self.image_dir = image_dir
        self.frame_base = frame_base
        self.frame_from_filename = frame_from_filename

        files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
        files.sort(key=natural_sort_key)
        if not files:
            raise RuntimeError(f"No images found in {image_dir}")

        self.files = files
        self.width, self.height = self._probe_size(files[0])

    @staticmethod
    def _probe_size(path: Path) -> Tuple[int, int]:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        h, w = img.shape[:2]
        return w, h

    def __len__(self) -> int:
        return len(self.files)

    def read(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(idx)
        frame = cv2.imread(str(self.files[idx]))
        if frame is None:
            raise RuntimeError(f"Failed to read image {self.files[idx]}")
        return frame

    def frame_number(self, idx: int) -> int:
        if not self.frame_from_filename:
            return idx + self.frame_base

        name = self.files[idx].stem
        nums = re.findall(r"\d+", name)
        if not nums:
            return idx + self.frame_base
        return int(nums[-1])

    def size(self) -> Tuple[int, int]:
        return self.width, self.height


class SortAdapter:
    """Wrapper that prefers external SORT, with an internal fallback tracker."""

    class SimpleSortFallback:
        """A lightweight SORT-like fallback based on IoU matching."""

        def __init__(self, max_age: int = 20, min_hits: int = 1, iou_threshold: float = 0.3):
            self.max_age = max_age
            self.min_hits = min_hits
            self.iou_threshold = iou_threshold
            self.next_id = 1
            self.frame_count = 0
            self.tracks: Dict[int, Dict[str, object]] = {}

        @staticmethod
        def _iou(a: np.ndarray, b: np.ndarray) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            denom = area_a + area_b - inter + 1e-6
            return inter / denom

        def update(self, dets: np.ndarray) -> np.ndarray:
            self.frame_count += 1
            boxes = dets[:, :4] if dets.size else np.empty((0, 4), dtype=float)

            for tr in self.tracks.values():
                tr["age"] = int(tr["age"]) + 1
                tr["time_since_update"] = int(tr["time_since_update"]) + 1

            track_ids = list(self.tracks.keys())
            unmatched_dets = set(range(len(boxes)))
            unmatched_tracks = set(track_ids)
            matches: List[Tuple[int, int]] = []

            if len(boxes) > 0 and len(track_ids) > 0:
                pair_scores: List[Tuple[float, int, int]] = []
                for di, dbox in enumerate(boxes):
                    for tid in track_ids:
                        iou = self._iou(dbox, np.asarray(self.tracks[tid]["bbox"], dtype=float))
                        if iou >= self.iou_threshold:
                            pair_scores.append((iou, di, tid))

                pair_scores.sort(key=lambda t: t[0], reverse=True)
                for _, di, tid in pair_scores:
                    if di in unmatched_dets and tid in unmatched_tracks:
                        matches.append((di, tid))
                        unmatched_dets.remove(di)
                        unmatched_tracks.remove(tid)

            for di, tid in matches:
                tr = self.tracks[tid]
                tr["bbox"] = boxes[di].copy()
                tr["hits"] = int(tr["hits"]) + 1
                tr["time_since_update"] = 0

            for di in unmatched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": boxes[di].copy(),
                    "hits": 1,
                    "age": 1,
                    "time_since_update": 0,
                }

            to_drop = [tid for tid, tr in self.tracks.items() if int(tr["time_since_update"]) > self.max_age]
            for tid in to_drop:
                self.tracks.pop(tid, None)

            out_rows = []
            for tid, tr in self.tracks.items():
                tsu = int(tr["time_since_update"])
                hits = int(tr["hits"])
                if tsu == 0 and (hits >= self.min_hits or self.frame_count <= self.min_hits):
                    x1, y1, x2, y2 = np.asarray(tr["bbox"], dtype=float).tolist()
                    out_rows.append([x1, y1, x2, y2, float(tid)])

            if not out_rows:
                return np.empty((0, 5), dtype=float)
            return np.asarray(out_rows, dtype=float)

    def __init__(self, max_age: int = 20, min_hits: int = 1, iou_threshold: float = 0.3, allow_fallback: bool = True):
        self.backend = "sort-tracker"
        self.import_error: Optional[str] = None
        try:
            sort_cls = self._load_sort_class()
            kwargs = self._compatible_kwargs(sort_cls, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
            self.tracker = sort_cls(**kwargs)
        except Exception as e:
            if not allow_fallback:
                raise
            self.backend = "fallback"
            self.import_error = str(e)
            self.tracker = self.SimpleSortFallback(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
            )

    @staticmethod
    def _compatible_kwargs(sort_cls, **candidate_kwargs):
        try:
            sig = inspect.signature(sort_cls)
            params = sig.parameters
            return {k: v for k, v in candidate_kwargs.items() if k in params}
        except Exception:
            return candidate_kwargs

    @staticmethod
    def _load_sort_class():
        candidates = [
            ("sort", "Sort"),
            ("sort_tracker", "Sort"),
            ("sort_tracker.sort", "Sort"),
        ]
        for module_name, cls_name in candidates:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, cls_name):
                    return getattr(module, cls_name)
            except Exception:
                continue
        raise ImportError(
            "Could not import SORT tracker. Please install with `pip install sort-tracker` "
            "or make sure `Sort` is importable."
        )

    def update(self, detections_xyxy: np.ndarray) -> np.ndarray:
        """
        detections_xyxy: shape [N, 4] or [N, 5]
        returns tracks as shape [M, >=5], last column is track id.
        """
        if detections_xyxy.size == 0:
            dets = np.empty((0, 5), dtype=float)
        else:
            if detections_xyxy.shape[1] == 4:
                scores = np.ones((detections_xyxy.shape[0], 1), dtype=float)
                dets = np.concatenate([detections_xyxy.astype(float), scores], axis=1)
            else:
                dets = detections_xyxy.astype(float)

        tracks = self.tracker.update(dets)
        if tracks is None:
            return np.empty((0, 5), dtype=float)
        tracks = np.asarray(tracks, dtype=float)
        if tracks.ndim == 1 and tracks.size > 0:
            tracks = tracks[None, :]
        if tracks.size == 0:
            return np.empty((0, 5), dtype=float)
        return tracks


class BehaviorAnnotator:
    def __init__(
        self,
        source: FrameSource,
        annotations: Dict[int, List[Annotation]],
        output_csv: Path,
        autosave: bool = True,
        autosave_every: int = 20,
        window_name: str = "AVA Behavior Annotator",
        sort_max_age: int = 20,
        sort_min_hits: int = 1,
        sort_iou_threshold: float = 0.3,
        recent_memory_frames: int = 8,
        use_sort: bool = True,
        active_id: Optional[int] = None,
        auto_next_on_click: bool = True,
        round_by_id: bool = True,
        output_input_like: bool = True,
        output_video_key: Optional[str] = None,
        output_id_last: bool = True,
        output_header: bool = False,
    ):
        self.source = source
        self.output_csv = output_csv
        self.autosave = autosave
        self.autosave_every = max(1, int(autosave_every))
        self.window_name = window_name
        self.output_input_like = output_input_like
        self.output_video_key = output_video_key
        self.output_id_last = output_id_last
        self.output_header = output_header

        self.frame_indices = list(range(len(source)))
        self.frame_numbers = [source.frame_number(i) for i in self.frame_indices]
        self.frame_number_to_idx = {f: i for i, f in enumerate(self.frame_numbers)}

        # Ensure all source frames exist in annotation dict.
        self.annotations = {f: list(annotations.get(f, [])) for f in self.frame_numbers}

        self.current_idx = 0
        self.current_frame: Optional[np.ndarray] = None
        self.current_tracks: np.ndarray = np.empty((0, 5), dtype=float)

        self.img_w, self.img_h = source.size()
        self.normalized_coords = self._guess_normalized_coords()

        self.sort_max_age = sort_max_age
        self.sort_min_hits = sort_min_hits
        self.sort_iou_threshold = sort_iou_threshold
        self.recent_memory_frames = max(1, int(recent_memory_frames))
        self.use_sort = use_sort
        self.active_id: Optional[int] = active_id
        self.active_action: Optional[str] = None
        self.auto_next_on_click = auto_next_on_click
        self.round_by_id = round_by_id
        self.sort_available = False
        self.sort_backend = "off"
        self.sort_error: Optional[str] = None
        self._warned_sort_fallback = False
        self._warned_sort_off = False
        self._autosave_counter = 0
        self.undo_stack: List[UndoState] = []
        self.undo_limit = 200
        self.suppressed_pred: set[Tuple[int, int]] = set()

        self._reset_tracking_state()
        init_fixed = 0
        for f in self.frame_numbers:
            init_fixed += self._resolve_duplicate_ids_in_frame(f)
        if init_fixed > 0:
            print(f"Startup cleanup: resolved {init_fixed} duplicate ID assignment(s) from loaded annotations.")

    def _reset_tracking_state(self):
        self.track_to_user_id: Dict[int, int] = {}
        self.track_last_seen_idx: Dict[int, int] = {}
        self.frame_tracks: Dict[int, np.ndarray] = {}
        self.predicted_until_idx = -1

        if not self.use_sort:
            self.sorter = None
            self.sort_available = False
            self.sort_backend = "off"
            self.sort_error = "SORT disabled by user option."
            if not self._warned_sort_off:
                print("SORT is disabled. Running in manual-only mode.")
                self._warned_sort_off = True
            return

        try:
            self.sorter = SortAdapter(
                max_age=self.sort_max_age,
                min_hits=self.sort_min_hits,
                iou_threshold=self.sort_iou_threshold,
                allow_fallback=True,
            )
            self.sort_available = True
            self.sort_backend = self.sorter.backend
            self.sort_error = None
            if self.sort_backend == "fallback":
                reason = self.sorter.import_error or "external SORT not found"
                if not self._warned_sort_fallback:
                    print("Warning: external SORT is unavailable. Using internal fallback tracker.")
                    print(f"Reason: {reason}")
                    self._warned_sort_fallback = True
        except Exception as e:
            self.sorter = None
            self.sort_available = False
            self.sort_backend = "off"
            self.sort_error = str(e)
            if not self._warned_sort_off:
                print("Warning: SORT is unavailable. Running in manual-only mode.")
                print(f"Reason: {e}")
                self._warned_sort_off = True

    def _guess_normalized_coords(self) -> bool:
        for frame_no in self.frame_numbers:
            anns = self.annotations.get(frame_no, [])
            for ann in anns:
                vals = [ann.x1, ann.y1, ann.x2, ann.y2]
                if all(0.0 <= v <= 1.0 for v in vals):
                    return True
                return False
        return False

    def _ann_to_pixel_bbox(self, ann: Annotation) -> Tuple[int, int, int, int]:
        if self.normalized_coords:
            x1 = int(round(ann.x1 * self.img_w))
            y1 = int(round(ann.y1 * self.img_h))
            x2 = int(round(ann.x2 * self.img_w))
            y2 = int(round(ann.y2 * self.img_h))
        else:
            x1 = int(round(ann.x1))
            y1 = int(round(ann.y1))
            x2 = int(round(ann.x2))
            y2 = int(round(ann.y2))

        x1 = max(0, min(self.img_w - 1, x1))
        y1 = max(0, min(self.img_h - 1, y1))
        x2 = max(0, min(self.img_w - 1, x2))
        y2 = max(0, min(self.img_h - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _frame_detections_xyxy(self, frame_no: int) -> np.ndarray:
        anns = self.annotations.get(frame_no, [])
        if not anns:
            return np.empty((0, 4), dtype=float)

        boxes = []
        for ann in anns:
            x1, y1, x2, y2 = self._ann_to_pixel_bbox(ann)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        return np.asarray(boxes, dtype=float)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter + 1e-6
        return inter / denom

    def _match_ann_to_tracks(self, frame_no: int, tracks: np.ndarray, iou_thr: float = 0.2) -> Dict[int, int]:
        """Return annotation index -> track_id."""
        anns = self.annotations.get(frame_no, [])
        if not anns or tracks.size == 0:
            return {}

        ann_boxes = np.asarray([self._ann_to_pixel_bbox(a) for a in anns], dtype=float)
        track_boxes = tracks[:, :4]
        track_ids = tracks[:, -1].astype(int)

        matches: Dict[int, int] = {}
        used_tracks = set()

        for ann_idx, ann_box in enumerate(ann_boxes):
            best_j = -1
            best_iou = 0.0
            for j, tbox in enumerate(track_boxes):
                if j in used_tracks:
                    continue
                iou = self._iou(ann_box, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                matches[ann_idx] = int(track_ids[best_j])
                used_tracks.add(best_j)

        return matches

    def _process_tracking_frame(self, idx: int):
        frame_no = self.frame_numbers[idx]
        if not self.sort_available or self.sorter is None:
            self.frame_tracks[frame_no] = np.empty((0, 5), dtype=float)
            return

        dets = self._frame_detections_xyxy(frame_no)
        tracks = self.sorter.update(dets)
        self.frame_tracks[frame_no] = tracks

        anns = self.annotations[frame_no]
        ann_to_track = self._match_ann_to_tracks(frame_no, tracks)
        used_ids = {int(a.obj_id) for a in anns if a.obj_id is not None}

        # Keep ID memory local to recent frames to reduce long-range drift.
        stale_track_ids = [tid for tid, last_idx in self.track_last_seen_idx.items() if idx - last_idx > self.recent_memory_frames]
        for tid in stale_track_ids:
            self.track_last_seen_idx.pop(tid, None)
            self.track_to_user_id.pop(tid, None)

        for ann_idx, track_id in ann_to_track.items():
            ann = anns[ann_idx]
            can_predict_here = (frame_no, ann_idx) not in self.suppressed_pred
            last_seen = self.track_last_seen_idx.get(track_id, -10_000_000)
            recent_enough = (idx - last_seen) <= self.recent_memory_frames
            if ann.obj_id is None and can_predict_here and track_id in self.track_to_user_id and recent_enough:
                cand_id = int(self.track_to_user_id[track_id])
                if cand_id not in used_ids:
                    ann.obj_id = cand_id
                    ann.id_source = "predicted"
                    used_ids.add(cand_id)
            if ann.obj_id is not None:
                self.track_to_user_id[track_id] = int(ann.obj_id)
                self.track_last_seen_idx[track_id] = idx

        # Guard rail for ID switch moments: keep at most one box per ID in one frame.
        self._resolve_duplicate_ids_in_frame(frame_no)

    def ensure_tracking_to(self, idx: int):
        if not self.sort_available or self.sorter is None:
            return
        if idx <= self.predicted_until_idx:
            return
        for i in range(self.predicted_until_idx + 1, idx + 1):
            self._process_tracking_frame(i)
        self.predicted_until_idx = idx

    def invalidate_from(self, frame_idx: int):
        # Clear only predicted IDs from frame_idx onward.
        for i in range(frame_idx, len(self.frame_numbers)):
            frame_no = self.frame_numbers[i]
            for ann in self.annotations[frame_no]:
                if ann.id_source == "predicted":
                    ann.obj_id = None
                    ann.id_source = None

        if self.sort_available and self.sorter is not None:
            self._reset_tracking_state()
            if frame_idx > 0:
                self.ensure_tracking_to(frame_idx - 1)

    @staticmethod
    def color_for_id(obj_id: Optional[int]) -> Tuple[int, int, int]:
        if obj_id is None:
            return (180, 180, 180)
        # Deterministic pseudo-random color from ID.
        r = (obj_id * 37 + 17) % 255
        g = (obj_id * 73 + 29) % 255
        b = (obj_id * 101 + 53) % 255
        return int(b), int(g), int(r)

    def _find_clicked_annotation(self, x: int, y: int) -> Optional[int]:
        frame_no = self.frame_numbers[self.current_idx]
        anns = self.annotations[frame_no]

        candidates = []
        for i, ann in enumerate(anns):
            x1, y1, x2, y2 = self._ann_to_pixel_bbox(ann)
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                candidates.append((area, i))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _enforce_unique_id_in_frame(self, frame_no: int, keep_ann_idx: int, obj_id: Optional[int]) -> int:
        """Keep clicked box as the only owner of obj_id in this frame."""
        if obj_id is None:
            return 0

        anns = self.annotations.get(frame_no, [])
        tracks = self.frame_tracks.get(frame_no, np.empty((0, 5), dtype=float))
        ann_to_track = self._match_ann_to_tracks(frame_no, tracks) if tracks.size > 0 else {}

        cleared = 0
        for i, ann in enumerate(anns):
            if i == keep_ann_idx:
                continue
            if ann.obj_id == obj_id:
                ann.obj_id = None
                ann.id_source = None
                cleared += 1
                tid = ann_to_track.get(i)
                if tid is not None and self.track_to_user_id.get(tid) == obj_id:
                    self.track_to_user_id.pop(tid, None)
        return cleared

    @staticmethod
    def _frame_signature(anns: List[Annotation]) -> List[Tuple[float, float, float, float, Optional[int], Optional[str], Optional[str]]]:
        return [
            (
                float(a.x1),
                float(a.y1),
                float(a.x2),
                float(a.y2),
                a.obj_id,
                str(a.action) if a.action is not None else None,
                a.id_source,
            )
            for a in anns
        ]

    def _capture_undo_before(self, frame_no: int, op: str) -> UndoState:
        return UndoState(
            frame_no=frame_no,
            prev_idx=self.current_idx,
            op=op,
            frame_before=copy.deepcopy(self.annotations.get(frame_no, [])),
            suppressed_before=set(self.suppressed_pred),
        )

    def _push_undo_if_changed(self, state: UndoState):
        after = self.annotations.get(state.frame_no, [])
        if self._frame_signature(state.frame_before) == self._frame_signature(after):
            return
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.undo_limit:
            self.undo_stack = self.undo_stack[-self.undo_limit :]

    def _undo_last(self):
        if not self.undo_stack:
            print("Nothing to undo.")
            return

        state = self.undo_stack.pop()
        self.annotations[state.frame_no] = copy.deepcopy(state.frame_before)
        self.suppressed_pred = set(state.suppressed_before)
        frame_idx = self.frame_number_to_idx.get(state.frame_no, state.prev_idx)
        self.current_idx = max(0, min(len(self.frame_numbers) - 1, frame_idx))

        # Keep restored frame as-is, only clear future predicted labels.
        for i in range(self.current_idx + 1, len(self.frame_numbers)):
            frame_no = self.frame_numbers[i]
            for ann in self.annotations[frame_no]:
                if ann.id_source == "predicted":
                    ann.obj_id = None
                    ann.id_source = None

        if self.sort_available and self.sorter is not None:
            self._reset_tracking_state()
            if self.current_idx > 0:
                self.ensure_tracking_to(self.current_idx - 1)
        print(f"Undo: restored frame {state.frame_no} ({state.op}).")

    def _count_duplicate_ids_in_frame(self, frame_no: int) -> int:
        seen = set()
        dup = 0
        for ann in self.annotations.get(frame_no, []):
            if ann.obj_id is None:
                continue
            if ann.obj_id in seen:
                dup += 1
            else:
                seen.add(ann.obj_id)
        return dup

    @staticmethod
    def _source_priority(src: Optional[str]) -> int:
        if src == "manual":
            return 3
        if src == "input":
            return 2
        if src == "predicted":
            return 1
        return 0

    def _resolve_duplicate_ids_in_frame(self, frame_no: int) -> int:
        """
        Ensure one ID appears at most once in a frame.
        Keep the strongest box by source priority: manual > input > predicted.
        """
        anns = self.annotations.get(frame_no, [])
        by_id: Dict[int, List[int]] = {}
        for i, ann in enumerate(anns):
            if ann.obj_id is None:
                continue
            by_id.setdefault(int(ann.obj_id), []).append(i)

        total_cleared = 0
        for obj_id, idxs in by_id.items():
            if len(idxs) <= 1:
                continue
            keep_idx = max(idxs, key=lambda i: (self._source_priority(anns[i].id_source), -i))
            total_cleared += self._enforce_unique_id_in_frame(frame_no, keep_idx, obj_id)
        return total_cleared

    def _prompt_set_active_id(self, allow_clear: bool = True) -> bool:
        cur = "None" if self.active_id is None else str(self.active_id)
        prompt = f"Set active ID (current={cur}; empty=cancel"
        if allow_clear:
            prompt += "; c=clear"
        prompt += "): "
        val = input(prompt).strip()
        if not val:
            return False

        if allow_clear and val.lower() in ("c", "clear", "none", "null"):
            self.active_id = None
            print("Active ID cleared.")
            return True

        try:
            self.active_id = int(val)
            print(f"Active ID set to {self.active_id}.")
            return True
        except ValueError:
            print("Invalid ID. Please input an integer.")
            return False

    def on_mouse(self, event, x, y, flags, userdata):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            return

        ann_idx = self._find_clicked_annotation(x, y)
        if ann_idx is None:
            if event == cv2.EVENT_RBUTTONDOWN:
                # Convenient fallback: right-click empty area to undo.
                self._undo_last()
            return

        frame_no = self.frame_numbers[self.current_idx]
        ann = self.annotations[frame_no][ann_idx]

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.active_id is None:
                print(f"[Frame {frame_no}] No active ID. Press 'i' to set active ID, then click.")
                return

            undo_state = self._capture_undo_before(frame_no, "left_assign")
            self.suppressed_pred.discard((frame_no, ann_idx))
            ann.obj_id = self.active_id
            ann.id_source = "manual"
            if self.active_action is not None:
                ann.action = self.active_action
            self.invalidate_from(self.current_idx)
            self.ensure_tracking_to(self.current_idx)
            cleared = self._enforce_unique_id_in_frame(frame_no, ann_idx, ann.obj_id)
            self._push_undo_if_changed(undo_state)
            print(
                f"[Frame {frame_no}] bbox #{ann_idx} -> id {ann.obj_id}"
                + (f", action {ann.action}" if ann.action is not None else "")
            )
            if cleared > 0:
                print(f"[Frame {frame_no}] Fixed duplicate ID {ann.obj_id}: cleared {cleared} other bbox(es).")
            if self.auto_next_on_click:
                self._go_next_frame()

        if event == cv2.EVENT_MBUTTONDOWN:
            if ann.obj_id is not None:
                self.active_id = int(ann.obj_id)
                print(f"[Frame {frame_no}] Active ID picked from bbox #{ann_idx}: {self.active_id}")
            else:
                print(f"[Frame {frame_no}] bbox #{ann_idx} has no ID to pick.")

        if event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: full delete on clicked box (id+action).
            # Ctrl+right-click: clear ID only.
            undo_state = self._capture_undo_before(frame_no, "right_delete")
            clear_action = not bool(flags & cv2.EVENT_FLAG_CTRLKEY)

            ann.obj_id = None
            ann.id_source = None
            if clear_action:
                ann.action = None
            self.suppressed_pred.add((frame_no, ann_idx))

            self.invalidate_from(self.current_idx)
            self.ensure_tracking_to(self.current_idx)
            self._push_undo_if_changed(undo_state)
            if clear_action:
                print(f"[Frame {frame_no}] bbox #{ann_idx} id+action cleared")
            else:
                print(f"[Frame {frame_no}] bbox #{ann_idx} id cleared")

    def _go_next_frame(self):
        self._maybe_autosave_step()
        last_idx = len(self.frame_numbers) - 1
        if self.current_idx >= last_idx:
            if self.round_by_id:
                cur_id = "None" if self.active_id is None else str(self.active_id)
                print(
                    f"[Round Done] ID {cur_id} has reached the last frame. "
                    "Press 'r' to restart from frame 1 for the next pig."
                )
            return

        self.current_idx += 1
        self.ensure_tracking_to(self.current_idx)

    def _go_prev_frame(self):
        self._maybe_autosave_step()
        self.current_idx = max(0, self.current_idx - 1)

    def _maybe_autosave_step(self):
        if not self.autosave:
            return
        self._autosave_counter += 1
        if self._autosave_counter >= self.autosave_every:
            self.save_csv()
            self._autosave_counter = 0

    def draw_current_frame(self, frame: np.ndarray) -> np.ndarray:
        disp = frame.copy()
        frame_no = self.frame_numbers[self.current_idx]
        anns = self.annotations.get(frame_no, [])

        # Ensure current frame has latest tracking suggestion.
        self.ensure_tracking_to(self.current_idx)
        self.current_tracks = self.frame_tracks.get(frame_no, np.empty((0, 5), dtype=float))

        for i, ann in enumerate(anns):
            x1, y1, x2, y2 = self._ann_to_pixel_bbox(ann)
            color = self.color_for_id(ann.obj_id)

            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

            id_text = f"id:{ann.obj_id}" if ann.obj_id is not None else "id:None"
            action_text = f"act:{ann.action}" if ann.action else "act:-"
            src_text = f"({ann.id_source})" if ann.id_source else ""
            label = f"#{i} {id_text} {action_text} {src_text}".strip()

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(disp, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                disp,
                label,
                (x1 + 2, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = (
            f"Frame {frame_no} ({self.current_idx + 1}/{len(self.frame_numbers)}) | "
            f"SORT:{self.sort_backend if self.sort_available else 'OFF'} | "
            f"ActiveID:{self.active_id if self.active_id is not None else 'None'} | "
            f"ActiveAct:{self.active_action if self.active_action is not None else 'None'} | "
            f"Dup:{self._count_duplicate_ids_in_frame(frame_no)} | "
            f"Undo:{len(self.undo_stack)} | "
            f"Mem:{self.recent_memory_frames}f | "
            f"RoundMode:{'ID' if self.round_by_id else 'FRAME'} | "
            f"AutoNext:{'ON' if self.auto_next_on_click else 'OFF'} | "
            "Keys: n/right-next p/left-prev r-nextPig z-undo i-setID c-clearID a-setAct x-clearAct s-save q-quit | "
            "Mouse: left=assign(mid pickID) right=deleteID+action ctrl+right=deleteID right(empty)=undo"
        )
        cv2.putText(
            disp,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 240, 30),
            2,
            cv2.LINE_AA,
        )

        return disp

    def save_csv(self):
        rows = []
        for frame_no in self.frame_numbers:
            for ann in self.annotations.get(frame_no, []):
                obj_id = ann.obj_id if ann.obj_id is not None else -1
                action = ann.action if ann.action is not None else -1
                if self.output_input_like:
                    rows.append(
                        {
                            "video": self.output_video_key if self.output_video_key is not None else "",
                            "frame": frame_no,
                            "x1": ann.x1,
                            "y1": ann.y1,
                            "x2": ann.x2,
                            "y2": ann.y2,
                            "action": action,
                            "id": obj_id,
                        }
                    )
                else:
                    rows.append(
                        {
                            "frame": frame_no,
                            "x1": ann.x1,
                            "y1": ann.y1,
                            "x2": ann.x2,
                            "y2": ann.y2,
                            "id": obj_id,
                            "action": action,
                        }
                    )
        if self.output_input_like:
            if self.output_id_last:
                cols = ["video", "frame", "x1", "y1", "x2", "y2", "action", "id"]
            else:
                cols = ["video", "frame", "x1", "y1", "x2", "y2", "id", "action"]
            df = pd.DataFrame(rows, columns=cols)
        else:
            df = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "id", "action"])
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False, header=self.output_header, float_format="%.6f")
        print(f"Saved {len(df)} rows to: {self.output_csv}")

    def run(self, start_frame: Optional[int] = None):
        if start_frame is not None and start_frame in self.frame_number_to_idx:
            self.current_idx = self.frame_number_to_idx[start_frame]

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        try:
            while True:
                self.current_frame = self.source.read(self.current_idx)
                disp = self.draw_current_frame(self.current_frame)
                cv2.imshow(self.window_name, disp)

                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue

                # Windows arrow key codes from waitKeyEx: right=2555904, left=2424832
                if key in (ord("n"), ord("N"), 2555904):
                    self._go_next_frame()

                elif key in (ord("p"), ord("P"), 2424832):
                    self._go_prev_frame()

                elif key == ord("s"):
                    self.save_csv()

                elif key in (ord("i"), ord("I")):
                    self._prompt_set_active_id(allow_clear=True)

                elif key in (ord("c"), ord("C")):
                    self.active_id = None
                    print("Active ID cleared.")

                elif key in (ord("a"), ord("A")):
                    cur = "None" if self.active_action is None else self.active_action
                    val = input(f"Set active action (current={cur}; empty=clear): ").strip()
                    self.active_action = val if val else None
                    print(f"Active action: {self.active_action}")

                elif key in (ord("x"), ord("X")):
                    self.active_action = None
                    print("Active action cleared.")

                elif key in (ord("z"), ord("Z")):
                    self._undo_last()

                elif key in (ord("r"), ord("R")):
                    if self.round_by_id:
                        cur = "None" if self.active_id is None else str(self.active_id)
                        suggested = str(self.active_id + 1) if self.active_id is not None else ""
                        msg = f"Next pig ID (current={cur}"
                        if suggested:
                            msg += f", enter for {suggested}"
                        msg += ", keep=., clear=c): "
                        val = input(msg).strip()
                        if val == "":
                            if suggested:
                                self.active_id = int(suggested)
                                print(f"Active ID set to {self.active_id}")
                        elif val == ".":
                            pass
                        elif val.lower() in ("c", "clear", "none"):
                            self.active_id = None
                            print("Active ID cleared.")
                        else:
                            try:
                                self.active_id = int(val)
                                print(f"Active ID set to {self.active_id}")
                            except ValueError:
                                print("Invalid ID, keep current.")
                        self.current_idx = 0
                        self.ensure_tracking_to(self.current_idx)
                    else:
                        print("Round-by-ID mode is off.")

                elif key == ord("q"):
                    self.save_csv()
                    break

        finally:
            cv2.destroyAllWindows()


# ---------------- Data loading ----------------


def read_csv_flexible(path: Path) -> pd.DataFrame:
    def infer_last_two_for_ava(raw_df: pd.DataFrame, col_a: str, col_b: str) -> Tuple[str, str]:
        a = pd.to_numeric(raw_df[col_a], errors="coerce")
        b = pd.to_numeric(raw_df[col_b], errors="coerce")
        a_valid = a.notna().sum()
        b_valid = b.notna().sum()
        a_neg = ((a < 0) & a.notna()).sum() / max(1, a_valid)
        b_neg = ((b < 0) & b.notna()).sum() / max(1, b_valid)
        a_uni = a.nunique(dropna=True)
        b_uni = b.nunique(dropna=True)

        # In many pig annotations, ID is the last column and often starts as -1.
        if b_neg > a_neg + 0.05:
            return "action", "id"
        if a_neg > b_neg + 0.05:
            return "id", "action"
        if b_uni > a_uni * 1.5:
            return "action", "id"
        if a_uni > b_uni * 1.5:
            return "id", "action"
        # Default to "last column is id" for AVA-like raw inputs in this project.
        return "action", "id"

    def normalize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for c in df_in.columns:
            cl = str(c).strip().lower()
            if cl in ("video", "video_id", "movie", "clip", "clip_id"):
                rename_map[c] = "video"
            elif cl in ("frame", "frame_id", "frameid", "timestamp"):
                rename_map[c] = "frame"
            elif cl in ("x1", "xmin", "left"):
                rename_map[c] = "x1"
            elif cl in ("y1", "ymin", "top"):
                rename_map[c] = "y1"
            elif cl in ("x2", "xmax", "right"):
                rename_map[c] = "x2"
            elif cl in ("y2", "ymax", "bottom"):
                rename_map[c] = "y2"
            elif cl in ("id", "track_id", "object_id", "person_id", "actor_id"):
                rename_map[c] = "id"
            elif cl in ("action", "label", "action_label", "class"):
                rename_map[c] = "action"
        return df_in.rename(columns=rename_map)

    def is_numeric_value(v) -> bool:
        try:
            float(v)
            return True
        except Exception:
            return False

    required = {"frame", "x1", "y1", "x2", "y2"}

    # 1) Try as normal header CSV first.
    df = normalize_columns(pd.read_csv(path))
    if df.empty:
        return df
    if required.issubset(set(df.columns)):
        return df

    # 2) Fallback for no-header CSV (common AVA export style).
    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return raw

    ncols = raw.shape[1]
    cols = [f"col{i}" for i in range(ncols)]
    raw.columns = cols

    first0 = raw.iloc[0, 0] if ncols >= 1 else None
    first1 = raw.iloc[0, 1] if ncols >= 2 else None
    first_is_text = not is_numeric_value(first0)
    second_is_num = is_numeric_value(first1)

    if ncols >= 8 and first_is_text and second_is_num:
        # video,frame,x1,y1,x2,y2,(id/action),(action/id)
        c6_name, c7_name = infer_last_two_for_ava(raw, "col6", "col7")
        new_cols = ["video", "frame", "x1", "y1", "x2", "y2", c6_name, c7_name] + cols[8:]
    elif ncols >= 7 and first_is_text and second_is_num:
        # video,frame,x1,y1,x2,y2,id
        new_cols = ["video", "frame", "x1", "y1", "x2", "y2", "id"] + cols[7:]
    elif ncols >= 6 and first_is_text and second_is_num:
        # video,frame,x1,y1,x2,y2
        new_cols = ["video", "frame", "x1", "y1", "x2", "y2"] + cols[6:]
    elif ncols >= 7:
        # frame,x1,y1,x2,y2,id,action
        new_cols = ["frame", "x1", "y1", "x2", "y2", "id", "action"] + cols[7:]
    elif ncols >= 6:
        # frame,x1,y1,x2,y2,id
        new_cols = ["frame", "x1", "y1", "x2", "y2", "id"] + cols[6:]
    elif ncols >= 5:
        # frame,x1,y1,x2,y2
        new_cols = ["frame", "x1", "y1", "x2", "y2"] + cols[5:]
    else:
        raise ValueError(f"CSV {path} must contain at least 5 columns for frame and bbox.")

    raw.columns = new_cols
    raw = normalize_columns(raw)
    if required.issubset(set(raw.columns)):
        return raw

    raise ValueError(f"CSV {path} must contain at least columns: frame,x1,y1,x2,y2")


def filter_video_rows(
    df: pd.DataFrame,
    csv_name: str,
    video_key: Optional[str] = None,
    auto_video_key: Optional[str] = None,
) -> pd.DataFrame:
    if "video" not in df.columns:
        return df

    video_col = df["video"].astype(str)
    unique_videos = sorted(v for v in video_col.dropna().unique() if v != "nan")
    if not unique_videos:
        return df

    target = None
    if video_key:
        target = video_key
    elif len(unique_videos) == 1:
        target = unique_videos[0]
    elif auto_video_key and auto_video_key in unique_videos:
        target = auto_video_key

    if target is None:
        preview = ", ".join(unique_videos[:5])
        raise ValueError(
            f"{csv_name} contains multiple videos ({len(unique_videos)}): {preview}. "
            "Please pass --video-key to choose one."
        )

    out = df[video_col == target].copy()
    if out.empty:
        raise ValueError(f"{csv_name} has no rows for video key: {target}")

    print(f"{csv_name}: using video key '{target}' ({len(out)} rows).")
    return out


def infer_video_key_from_bbox_csv(bbox_csv: Path, explicit_video_key: Optional[str]) -> Optional[str]:
    if explicit_video_key:
        return explicit_video_key

    df = read_csv_flexible(bbox_csv)
    if "video" not in df.columns:
        return None

    unique_videos = sorted(v for v in df["video"].astype(str).dropna().unique() if v != "nan")
    if not unique_videos:
        return None
    if len(unique_videos) == 1:
        only = unique_videos[0]
        print(f"Inferred video key from bbox csv: '{only}'")
        return only

    first_row_video = str(df.iloc[0]["video"])
    print(
        f"bbox csv ({bbox_csv}) contains multiple videos ({len(unique_videos)}). "
        f"Auto-selecting first-row video key: '{first_row_video}'. "
        "Use --video-key to override."
    )
    return first_row_video


def bbox_key(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    return (round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4))


def load_annotations(
    bbox_csv: Path,
    source_frame_numbers: List[int],
    action_csv: Optional[Path] = None,
    video_key: Optional[str] = None,
    auto_video_key: Optional[str] = None,
) -> Dict[int, List[Annotation]]:
    frame_set = set(source_frame_numbers)
    anns: Dict[int, List[Annotation]] = {f: [] for f in source_frame_numbers}

    bbox_df = read_csv_flexible(bbox_csv)
    bbox_df = filter_video_rows(
        bbox_df,
        csv_name=f"bbox csv ({bbox_csv})",
        video_key=video_key,
        auto_video_key=auto_video_key,
    )
    if bbox_df.empty:
        return anns

    action_by_bbox: Dict[Tuple[int, Tuple[float, float, float, float]], str] = {}
    action_by_frame_id: Dict[Tuple[int, int], str] = {}

    if action_csv is not None and action_csv.exists():
        action_df = read_csv_flexible(action_csv)
        action_df = filter_video_rows(
            action_df,
            csv_name=f"action csv ({action_csv})",
            video_key=video_key,
            auto_video_key=auto_video_key,
        )
        if not action_df.empty:
            for _, row in action_df.iterrows():
                frame_no = int(row["frame"])
                action_val = str(row["action"]) if "action" in row and pd.notna(row["action"]) else None
                if not action_val:
                    continue
                if all(col in action_df.columns for col in ["x1", "y1", "x2", "y2"]):
                    key = (frame_no, bbox_key(row["x1"], row["y1"], row["x2"], row["y2"]))
                    action_by_bbox[key] = action_val
                if "id" in action_df.columns and pd.notna(row.get("id")):
                    try:
                        obj_id = int(row["id"])
                        action_by_frame_id[(frame_no, obj_id)] = action_val
                    except Exception:
                        pass

    dropped = 0
    for _, row in bbox_df.iterrows():
        frame_no = int(row["frame"])
        if frame_no not in frame_set:
            dropped += 1
            continue

        obj_id = None
        id_source = None
        if "id" in bbox_df.columns and pd.notna(row.get("id")):
            try:
                parsed_id = int(float(row["id"]))
                if parsed_id >= 0:
                    obj_id = parsed_id
                    id_source = "input"
                else:
                    obj_id = None
                    id_source = None
            except Exception:
                obj_id = None
                id_source = None

        action = None
        if "action" in bbox_df.columns and pd.notna(row.get("action")):
            try:
                a_num = float(row["action"])
                if int(a_num) >= 0 and abs(a_num - int(a_num)) < 1e-6:
                    action = int(a_num)
                elif a_num >= 0:
                    action = a_num
                else:
                    action = None
            except Exception:
                a_txt = str(row["action"]).strip()
                action = a_txt if a_txt not in ("", "-1", "none", "None", "nan") else None
        else:
            key = (frame_no, bbox_key(row["x1"], row["y1"], row["x2"], row["y2"]))
            if obj_id is not None and (frame_no, obj_id) in action_by_frame_id:
                action = action_by_frame_id[(frame_no, obj_id)]
            elif key in action_by_bbox:
                action = action_by_bbox[key]

        anns[frame_no].append(
            Annotation(
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                obj_id=obj_id,
                action=action,
                id_source=id_source,
            )
        )

    if dropped > 0:
        print(f"Warning: {dropped} bbox rows were skipped because frame not found in source.")

    return anns


def build_source(args, preferred_video_key: Optional[str] = None) -> FrameSource:
    if args.video is None and args.image_dir is None:
        raise ValueError("Provide either --video or --image-dir")
    if args.video is not None and args.image_dir is not None:
        raise ValueError("Use only one of --video or --image-dir")

    if args.video is not None:
        return VideoSource(Path(args.video), frame_base=args.frame_base)
    resolved_dir = find_image_dir_for_video(Path(args.image_dir), preferred_video_key)
    print(f"Using image directory: {resolved_dir}")
    return ImageSequenceSource(
        resolved_dir,
        frame_base=args.frame_base,
        frame_from_filename=args.frame_from_filename,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive AVA-style behavior annotation tool")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image sequence directory")
    parser.add_argument("--bbox-csv", type=str, required=True, help="BBox CSV path")
    parser.add_argument("--action-csv", type=str, default=None, help="Optional action CSV path")
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Video key used to filter multi-video CSV rows (e.g., '10---27 - Trim')",
    )
    parser.add_argument("--output-csv", type=str, default="annotations_output.csv", help="Output CSV path")
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If output CSV exists, auto-load it to continue labeling (default: true)",
    )
    parser.add_argument("--frame-base", type=int, default=1, help="Frame number base for source indexing")
    parser.add_argument(
        "--frame-from-filename",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For image sequence, parse frame number from filename (default: true)",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Optional frame number to start from")
    parser.add_argument(
        "--autosave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-save CSV on frame navigation",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=20,
        help="Auto-save every N frame transitions/clicks (default: 20)",
    )
    parser.add_argument("--sort-max-age", type=int, default=20)
    parser.add_argument("--sort-min-hits", type=int, default=1)
    parser.add_argument("--sort-iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--recent-memory-frames",
        type=int,
        default=8,
        help="Use only recent N frames for ID memory in prediction (default: 8)",
    )
    parser.add_argument("--active-id", type=int, default=None, help="Initial active ID for fast click labeling")
    parser.add_argument(
        "--auto-next-on-click",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto move to next frame after left/right mouse click (default: true)",
    )
    parser.add_argument(
        "--round-by-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One-pig-per-round workflow: finish timeline, then press r to start next pig from frame 1",
    )
    parser.add_argument(
        "--disable-sort",
        action="store_true",
        help="Disable SORT assistance and run manual-only annotation.",
    )
    parser.add_argument(
        "--output-header",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write header row in output CSV (default: false, same as raw input style)",
    )
    parser.add_argument(
        "--output-input-like",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output as video,frame,x1,y1,x2,y2,action,id (default: true)",
    )
    parser.add_argument(
        "--output-id-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When output-input-like is true, put ID in the last column (default: true)",
    )
    parser.add_argument("--window-name", type=str, default="AVA Behavior Annotator")
    return parser.parse_args()


def main():
    args = parse_args()
    bbox_csv = Path(args.bbox_csv)
    output_csv = Path(args.output_csv)
    ann_csv_for_load = bbox_csv

    if args.resume_existing and output_csv.exists():
        try:
            if output_csv.stat().st_size > 0:
                ann_csv_for_load = output_csv
                print(f"Resume mode: loading annotations from existing output file: {output_csv}")
        except Exception:
            pass

    # Table-driven mode: infer one target video key from selected annotation source.
    inferred_video_key = infer_video_key_from_bbox_csv(ann_csv_for_load, args.video_key)

    source = build_source(args, preferred_video_key=inferred_video_key)
    frame_numbers = [source.frame_number(i) for i in range(len(source))]
    auto_video_key = inferred_video_key or (Path(args.video).stem if args.video else None)

    anns = load_annotations(
        bbox_csv=ann_csv_for_load,
        source_frame_numbers=frame_numbers,
        action_csv=Path(args.action_csv) if args.action_csv else None,
        video_key=inferred_video_key,
        auto_video_key=auto_video_key,
    )

    app = BehaviorAnnotator(
        source=source,
        annotations=anns,
        output_csv=output_csv,
        autosave=args.autosave,
        autosave_every=args.autosave_every,
        window_name=args.window_name,
        sort_max_age=args.sort_max_age,
        sort_min_hits=args.sort_min_hits,
        sort_iou_threshold=args.sort_iou_threshold,
        recent_memory_frames=args.recent_memory_frames,
        use_sort=not args.disable_sort,
        active_id=args.active_id,
        auto_next_on_click=args.auto_next_on_click,
        round_by_id=args.round_by_id,
        output_input_like=args.output_input_like,
        output_video_key=inferred_video_key,
        output_id_last=args.output_id_last,
        output_header=args.output_header,
    )
    app.run(start_frame=args.start_frame)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
