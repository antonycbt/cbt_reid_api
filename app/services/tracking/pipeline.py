# -*- coding: utf-8 -*-
"""
UPDATED: member_embeddings DB schema + 2-slot (0.5MB+0.5MB) rolling daily banks + InsightFace det_score>=0.75 gating

Save as: updated_tracking_member_embeddings_rollingslots.py

Example run:
python updated_tracking_member_embeddings_rollingslots.py   --use-db --db-url "postgresql://USER:PASS@HOST:5432/DB"   --src rtsp://... rtsp://...   --camera-ids 3 7   --use-face   --update-db-embeddings   --embed-min-face-det-score 0.75   --update-face-sim-thresh 0.75   --embeddings-slot-mb 0.5   --embeddings-flush-seconds 10   --embeddings-min-sample-seconds 0.5   --db-refresh-seconds 30   --save-csv   --show
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import math
import os
import random
import sys
import time
import threading
import queue
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, quote, unquote

import cv2
import numpy as np
import torch

# --- Excel (openpyxl) ---
try:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment, Font
    OPENPYXL_OK = True
except Exception:
    Workbook = None
    get_column_letter = None
    Alignment = None
    Font = None
    OPENPYXL_OK = False

# --- YOLO (Ultralytics) ---
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --- DeepSORT (deep-sort-realtime) ---
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None

# --- TorchReID gallery extractor ---
try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception:
    TorchreidExtractor = None

# --- InsightFace ---
try:
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except Exception:
    FaceAnalysis = None
    INSIGHT_OK = False

# --- ONNX Runtime ---
try:
    import onnxruntime as ort
except Exception:
    ort = None

# --- SQLAlchemy / pgvector ---
try:
    from sqlalchemy import Column, Integer, String, Boolean, LargeBinary, create_engine, select, BigInteger, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.dialects.postgresql import ARRAY
    from sqlalchemy import Float
except Exception as e:
    raise RuntimeError("SQLAlchemy is required for --use-db mode") from e

try:
    from pgvector.sqlalchemy import Vector
except Exception:
    Vector = None


# ----------------------------
# Globals (guards)
# ----------------------------
EXPECTED_DIM = 512
_yolo_lock = threading.Lock()   # avoid concurrent CUDA kernels on same YOLO instance
_reid_lock = threading.Lock()   # avoid concurrent kernels on same TorchReID extractor
_face_lock = threading.Lock()   # InsightFace (ORT/CUDA) guard


# ----------------------------
# Utils
# ----------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n


def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where((norms == 0) | (~np.isfinite(norms)), 1.0, norms)
    return m / norms


def safe_iter_faces(obj):
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return [obj]


def extract_face_embedding(face):
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb


def extract_face_det_score(face) -> float:
    """Best-effort InsightFace face detection confidence (0..1)."""
    try:
        s = getattr(face, "det_score", None)
        if s is None:
            s = getattr(face, "score", None)
        if s is None:
            return 1.0
        s = float(s)
        if not math.isfinite(s):
            return 0.0
        return s
    except Exception:
        return 1.0


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _sanitize_rtsp_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.username or parts.password:
        user = quote(unquote(parts.username or ""), safe="")
        pwd = quote(unquote(parts.password or ""), safe="")
        host = parts.hostname or ""
        netloc = f"{user}:{pwd}@{host}"
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


# ----------------------------
# Display helpers (GRID that fits the screen)
# ----------------------------
def _get_screen_resolution(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """Best-effort screen resolution detection for cv2.imshow layouts."""
    # Tkinter
    try:
        import tkinter as tk  # stdlib
        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    # Windows metrics
    try:
        if os.name == "nt":
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass

    return int(default[0]), int(default[1])


def _resize_to_cell_cover(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Resize to FILL the cell (cover), then center-crop."""
    if img is None or img.size == 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    scale = max(cell_w / float(iw), cell_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x1 = max(0, (new_w - cell_w) // 2)
    y1 = max(0, (new_h - cell_h) // 2)
    crop = resized[y1:y1 + cell_h, x1:x1 + cell_w]

    if crop.shape[0] != cell_h or crop.shape[1] != cell_w:
        crop = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
    return crop


def _resize_to_cell_contain(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Resize to FIT inside the cell (contain), then pad."""
    if img is None or img.size == 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    scale = min(cell_w / float(iw), cell_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x1 = (cell_w - new_w) // 2
    y1 = (cell_h - new_h) // 2
    out[y1:y1 + new_h, x1:x1 + new_w] = resized
    return out


def _choose_grid_auto(n: int, screen_w: int, screen_h: int, frame_aspect: float) -> Tuple[int, int]:
    """Choose (rows, cols) automatically based on aspect ratio fit."""
    if n <= 1:
        return 1, 1

    best = None
    for rows in range(1, n + 1):
        cols = int(math.ceil(n / rows))
        cell_w = screen_w / float(cols)
        cell_h = screen_h / float(rows)
        cell_aspect = cell_w / max(1.0, cell_h)
        penalty = abs(math.log(max(1e-6, cell_aspect / max(1e-6, frame_aspect))))
        blanks = (rows * cols) - n
        score = penalty + 0.02 * float(blanks)
        cand = (score, rows, cols)
        if best is None or cand < best:
            best = cand

    assert best is not None
    return int(best[1]), int(best[2])


def make_grid_view(
    frames: List[np.ndarray],
    screen_w: int,
    screen_h: int,
    mode: str = "cover",
    grid_rows: int = 0,
    grid_cols: int = 0,
) -> np.ndarray:
    """Build a grid image sized to (screen_w, screen_h)."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    n = len(frames)
    aspects = []
    for f in frames:
        h, w = f.shape[:2]
        if w > 0 and h > 0:
            aspects.append(w / float(h))
    frame_aspect = float(np.median(aspects)) if aspects else (16.0 / 9.0)

    if grid_rows > 0 and grid_cols > 0:
        rows, cols = int(grid_rows), int(grid_cols)
        if rows * cols < n:
            cols = int(math.ceil(n / rows))
    elif grid_rows > 0:
        rows = int(grid_rows)
        cols = int(math.ceil(n / rows))
    elif grid_cols > 0:
        cols = int(grid_cols)
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = _choose_grid_auto(n, screen_w, screen_h, frame_aspect)

    rows = max(1, int(rows))
    cols = max(1, int(cols))

    cell_w = max(1, int(screen_w // cols))
    cell_h = max(1, int(screen_h // rows))

    resize_fn = _resize_to_cell_cover if str(mode).lower() == "cover" else _resize_to_cell_contain

    tiles: List[np.ndarray] = []
    for i in range(rows * cols):
        if i < n:
            tile = resize_fn(frames[i], cell_w, cell_h)
        else:
            tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        tiles.append(tile)

    row_imgs = []
    idx = 0
    for _r in range(rows):
        row = np.concatenate(tiles[idx:idx + cols], axis=1)
        row_imgs.append(row)
        idx += cols

    vis = np.concatenate(row_imgs, axis=0)

    if vis.shape[1] != screen_w or vis.shape[0] != screen_h:
        vis = cv2.resize(vis, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)

    return vis


# ----------------------------
# ORT CUDA provider probe
# ----------------------------
def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("darwin"):
            return False
        capi_dir = Path(ort.__file__).parent / "capi"
        name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = capi_dir / name
        if not lib_path.exists():
            return False
        ctypes.CDLL(str(lib_path))
        return True
    except Exception:
        return False


# ----------------------------
# DB bank helpers
# ----------------------------
def decode_bank_gzip_npy(raw: bytes | None) -> np.ndarray | None:
    if not raw:
        return None
    try:
        with gzip.GzipFile(fileobj=BytesIO(raw), mode="rb") as gz:
            data = gz.read()
        arr = np.load(BytesIO(data), allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(arr).all():
            return None
        return l2_normalize_rows(arr)
    except Exception:
        return None


def encode_bank_gzip_npy(arr: np.ndarray | None) -> bytes | None:
    """Store bank as: gzip(np.save(.npy))  -> matches decode_bank_gzip_npy()."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        buf = BytesIO()
        np.save(buf, a, allow_pickle=False)
        raw = buf.getvalue()
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as gz:
            gz.write(raw)
        return out.getvalue()
    except Exception:
        return None


def _as_vec512(x) -> np.ndarray | None:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        if a.size != EXPECTED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        return l2_normalize(a)
    except Exception:
        return None


def _bytes_per_embedding() -> int:
    return int(EXPECTED_DIM * 4)  # float32


def _rows_from_mb(mb: float) -> int:
    mb = float(mb or 0.0)
    if mb <= 0:
        return 0
    return max(1, int((mb * 1024.0 * 1024.0) // float(_bytes_per_embedding())))


# ----------------------------
# ✅ CSV Audit loggers (member_id + camera_id)
# ----------------------------
class EmbeddingUpdateLogger:
    """
    Append-only CSV audit log for DB embedding updates.
    """
    def __init__(self, path: str, warn_interval_s: float = 5.0):
        self.path = str(path)
        self._lock = threading.Lock()
        self._warn_interval_s = float(max(0.0, float(warn_interval_s or 0.0)))
        self._last_warn_mono = 0.0

        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception as e:
            self._warn(f"[WARN] Could not create audit log folder for {self.path}: {e}")

        try:
            if self.path and (not os.path.exists(self.path)):
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "timestamp",
                        "member_id",
                        "name",
                        "camera_id",
                        "tracks",
                        "face_sim_max",
                        "body_added",
                        "body_removed",
                        "body_before",
                        "body_after",
                        "face_added",
                        "face_removed",
                        "face_before",
                        "face_after",
                    ])
        except Exception as e:
            self._warn(f"[WARN] Could not init embeddings audit CSV at {self.path}: {e}")

    def _warn(self, msg: str) -> None:
        if not msg:
            return
        if self._warn_interval_s <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_warn_mono)) >= float(self._warn_interval_s):
            self._last_warn_mono = now
            print(msg)

    def log(
        self,
        ts: float,
        member_id: int,
        name: str,
        camera_id: int,
        tracks: str,
        face_sim_max: float,
        body_added: int,
        body_removed: int,
        body_before: int,
        body_after: int,
        face_added: int,
        face_removed: int,
        face_before: int,
        face_after: int,
    ) -> None:
        try:
            with self._lock:
                with open(self.path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S"),
                        int(member_id),
                        str(name),
                        int(camera_id),
                        str(tracks),
                        f"{float(face_sim_max):.4f}",
                        int(body_added),
                        int(body_removed),
                        int(body_before),
                        int(body_after),
                        int(face_added),
                        int(face_removed),
                        int(face_before),
                        int(face_after),
                    ])
        except Exception as e:
            self._warn(f"[WARN] Embeddings audit CSV append failed ({self.path}): {e}")


class EmbeddingSampleLogger:
    """Append-only CSV log for accepted embedding samples."""
    def __init__(self, path: str, warn_interval_s: float = 5.0):
        self.path = str(path)
        self._lock = threading.Lock()
        self._warn_interval_s = float(max(0.0, float(warn_interval_s or 0.0)))
        self._last_warn_mono = 0.0

        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception as e:
            self._warn(f"[WARN] Could not create samples log folder for {self.path}: {e}")

        try:
            if self.path and (not os.path.exists(self.path)):
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "timestamp",
                        "member_id",
                        "name",
                        "camera_id",
                        "track_id",
                        "face_sim",
                        "face_det_score",
                        "has_body_emb",
                        "has_face_emb",
                        "action",
                        "note",
                    ])
        except Exception as e:
            self._warn(f"[WARN] Could not init embeddings samples CSV at {self.path}: {e}")

    def _warn(self, msg: str) -> None:
        if not msg:
            return
        if self._warn_interval_s <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_warn_mono)) < float(self._warn_interval_s):
            return
        self._last_warn_mono = now
        print(msg)

    def log(
        self,
        ts: float,
        member_id: int,
        name: str,
        camera_id: int,
        track_id: int,
        face_sim: float,
        face_det_score: float,
        has_body_emb: bool,
        has_face_emb: bool,
        action: str = "accepted",
        note: str = "",
    ) -> None:
        try:
            with self._lock:
                with open(self.path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        float(ts),
                        int(member_id),
                        str(name),
                        int(camera_id),
                        int(track_id),
                        f"{float(face_sim):.4f}",
                        f"{float(face_det_score):.4f}",
                        int(bool(has_body_emb)),
                        int(bool(has_face_emb)),
                        str(action),
                        str(note),
                    ])
        except Exception as e:
            self._warn(f"[WARN] Embeddings samples CSV append failed ({self.path}): {e}")


# ----------------------------
# ✅ Embedding updater: 2-slot rolling (slot_mb + slot_mb = 1MB total)
# ----------------------------
@dataclass
class EmbeddingSample:
    member_id: int
    name: str
    camera_id: int
    track_id: int
    ts: float
    face_sim: float
    face_det_score: float = 1.0
    body_emb: Optional[np.ndarray] = None
    face_emb: Optional[np.ndarray] = None


class EmbeddingDBUpdater:
    """
    Background DB writer into member_embeddings with 2-slot daily rotation.

    Total bank size = 2 * slot_mb.
    On each new day (UTC):
      - clear the slot corresponding to today's date (today.toordinal()%2)
      - keep the other slot (yesterday)
      - append new samples into today's slot up to capacity
    Bootstrap:
      - if both slots are empty, we allow filling BOTH slots (so day-1 can reach 1MB)
    """


    def __init__(
        self,
        db_url: str,
        slot_mb: float,
        flush_seconds: float,
        min_sample_seconds: float,
        min_face_sim: float,
        min_face_det_score: float,
        log_csv_path: str,
        update_body: bool = True,
        update_face: bool = True,
        max_queue: int = 5000,
        reset_if_gap_days_ge: int = 2,
        reset_on_start: bool = False,
        samples_log_csv_path: str = "",
    ):
        self.db_url = str(db_url)

        self.slot_mb = float(slot_mb or 0.0)
        self.slot_rows = int(_rows_from_mb(self.slot_mb))
        self.total_rows = int(self.slot_rows * 2)

        self.flush_seconds = float(flush_seconds or 10.0)
        self.min_sample_seconds = float(min_sample_seconds or 0.5)

        self.min_face_sim = float(min_face_sim or 0.75)
        self.min_face_det_score = float(min_face_det_score or 0.75)

        self.update_body = bool(update_body)
        self.update_face = bool(update_face)

        self.reset_if_gap_days_ge = int(max(1, int(reset_if_gap_days_ge)))

        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))

        # key = (member_id, camera_id)
        self._body_buf: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
        self._face_buf: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
        self._meta_buf: Dict[Tuple[int, int], List[EmbeddingSample]] = defaultdict(list)

        self._last_sample_ts: Dict[Tuple[int, int], float] = defaultdict(float)
        self._last_flush_ts: Dict[Tuple[int, int], float] = defaultdict(float)

        self._state_lock = threading.Lock()
        self._full_state: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)

        # Logs
        self.logger = EmbeddingUpdateLogger(log_csv_path) if log_csv_path else None
        self.sample_logger = EmbeddingSampleLogger(samples_log_csv_path) if samples_log_csv_path else None

        Base = declarative_base()

        class MemberRow(Base):
            __tablename__ = "members"
            id = Column(Integer, primary_key=True)

            # Your schema:
            member_number = Column(String)
            first_name = Column(String)
            last_name = Column(String)
            is_active = Column(Boolean)

        class MemberEmbeddingRow(Base):
            __tablename__ = "member_embeddings"
            id = Column(BigInteger, primary_key=True)

            member_id = Column(Integer, nullable=False)
            camera_id = Column(Integer, nullable=False)

            if Vector is not None:
                body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
                face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            else:
                body_embedding = Column(ARRAY(Float), nullable=True)
                face_embedding = Column(ARRAY(Float), nullable=True)

            body_embeddings_raw = Column(LargeBinary, nullable=True)
            face_embeddings_raw = Column(LargeBinary, nullable=True)

            last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)

        self.MemberEmbeddingRow = MemberEmbeddingRow
        self.MemberRow = MemberRow

        self.engine = create_engine(self.db_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)

        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

        # Safety: do not wipe by default
        if bool(reset_on_start):
            try:
                self._reset_all_to_empty()
            except Exception:
                pass

    @staticmethod
    def _utc_today() -> datetime.date:
        return datetime.now(timezone.utc).date()

    @staticmethod
    def _utc_date_of(dt: Optional[datetime]) -> Optional[datetime.date]:
        if dt is None:
            return None
        try:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None

    def _slot_idx_for_date(self, d: datetime.date) -> int:
        return int(d.toordinal() % 2)

    def _split_slots(self, bank: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if bank is None:
            bank = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        bank = np.asarray(bank, dtype=np.float32)
        if bank.ndim != 2 or bank.shape[1] != EXPECTED_DIM:
            bank = np.zeros((0, EXPECTED_DIM), dtype=np.float32)

        if self.total_rows > 0 and bank.shape[0] > self.total_rows:
            bank = bank[: self.total_rows]

        s = int(self.slot_rows)
        if s <= 0:
            return bank, np.zeros((0, EXPECTED_DIM), dtype=np.float32)

        slot0 = bank[: min(s, bank.shape[0])]
        slot1 = bank[s: min(2 * s, bank.shape[0])] if bank.shape[0] > s else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        return slot0, slot1

    def _combine_slots(self, slot0: np.ndarray, slot1: np.ndarray) -> np.ndarray:
        slot0 = np.asarray(slot0, dtype=np.float32) if slot0 is not None else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        slot1 = np.asarray(slot1, dtype=np.float32) if slot1 is not None else np.zeros((0, EXPECTED_DIM), dtype=np.float32)

        if self.slot_rows > 0:
            slot0 = slot0[: self.slot_rows]
            slot1 = slot1[: self.slot_rows]

        out = np.concatenate([slot0, slot1], axis=0) if (slot0.size or slot1.size) else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        if self.total_rows > 0 and out.shape[0] > self.total_rows:
            out = out[: self.total_rows]
        return out

    def can_accept(self, member_id: int, camera_id: int) -> bool:
        mid = int(member_id)
        cid = int(camera_id)
        if mid <= 0 or cid <= 0:
            return False
        if self.slot_rows <= 0:
            return False

        today = self._utc_today()
        day_ord = int(today.toordinal())

        key = (mid, cid)
        with self._state_lock:
            st = self._full_state.get(key, {})
            st_day = int(st.get("day_ord", -1))
            if st_day != day_ord:
                return True
            return not bool(st.get("full", False))

    def enqueue(self, s: EmbeddingSample) -> None:
        if s is None:
            return
        try:
            self._q.put_nowait(s)
        except queue.Full:
            return

    def close(self) -> None:
        self._stop.set()
        try:
            self._thr.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._flush_all()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            drained = 0
            while drained < 64:
                try:
                    s: EmbeddingSample = self._q.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                self._handle_sample(s)

            try:
                self._flush_due()
            except Exception:
                pass

            self._stop.wait(0.05)

        try:
            self._flush_all()
        except Exception:
            pass

    def _handle_sample(self, s: EmbeddingSample) -> None:
        mid = int(s.member_id)
        cid = int(s.camera_id)
        if mid <= 0 or cid <= 0:
            return

        if float(s.face_sim) < float(self.min_face_sim):
            return
        if float(getattr(s, "face_det_score", 1.0) or 1.0) < float(self.min_face_det_score):
            return

        key = (mid, cid)

        now = float(time.time())
        last = float(self._last_sample_ts.get(key, 0.0))
        if (now - last) < float(self.min_sample_seconds):
            return
        self._last_sample_ts[key] = now

        body = _as_vec512(s.body_emb) if self.update_body else None
        face = _as_vec512(s.face_emb) if self.update_face else None

        if body is not None:
            self._body_buf[key].append(body.astype(np.float32))
        if face is not None:
            self._face_buf[key].append(face.astype(np.float32))

        if (body is not None) or (face is not None):
            self._meta_buf[key].append(s)

            if self.sample_logger is not None:
                try:
                    self.sample_logger.log(
                        ts=float(s.ts or now),
                        member_id=int(s.member_id),
                        name=str(s.name),
                        camera_id=int(s.camera_id),
                        track_id=int(s.track_id),
                        face_sim=float(s.face_sim),
                        face_det_score=float(s.face_det_score),
                        has_body_emb=bool(body is not None),
                        has_face_emb=bool(face is not None),
                        action="accepted",
                        note="",
                    )
                except Exception:
                    pass

    def _flush_due(self) -> None:
        if self.flush_seconds <= 0:
            return
        now = float(time.time())
        keys = set(self._body_buf.keys()) | set(self._face_buf.keys())
        for key in list(keys):
            has_any = (len(self._body_buf.get(key, [])) > 0) or (len(self._face_buf.get(key, [])) > 0)
            if not has_any:
                continue
            last_flush = float(self._last_flush_ts.get(key, 0.0))
            if (now - last_flush) >= float(self.flush_seconds):
                self._flush_key(key)

    def _flush_all(self) -> None:
        keys = set(self._body_buf.keys()) | set(self._face_buf.keys())
        for key in list(keys):
            try:
                self._flush_key(key)
            except Exception:
                pass

    def _flush_key(self, key: Tuple[int, int]) -> None:
        mid, cid = int(key[0]), int(key[1])

        body_list = self._body_buf.get(key, [])
        face_list = self._face_buf.get(key, [])
        if not body_list and not face_list:
            return

        now_dt = datetime.now(timezone.utc)
        today = now_dt.date()
        slot_idx = self._slot_idx_for_date(today)

        with self.Session() as session:
            stmt = select(self.MemberEmbeddingRow).where(
                (self.MemberEmbeddingRow.member_id == mid) &
                (self.MemberEmbeddingRow.camera_id == cid)
            )
            row = session.execute(stmt).scalars().first()

            if row is None:
                row = self.MemberEmbeddingRow(member_id=mid, camera_id=cid)
                session.add(row)
                session.flush()

            old_body = decode_bank_gzip_npy(getattr(row, "body_embeddings_raw", None)) if self.update_body else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            old_face = decode_bank_gzip_npy(getattr(row, "face_embeddings_raw", None)) if self.update_face else np.zeros((0, EXPECTED_DIM), dtype=np.float32)

            if old_body is None:
                old_body = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            if old_face is None:
                old_face = np.zeros((0, EXPECTED_DIM), dtype=np.float32)

            body_before = int(old_body.shape[0])
            face_before = int(old_face.shape[0])

            b0, b1 = self._split_slots(old_body)
            f0, f1 = self._split_slots(old_face)

            last_date = self._utc_date_of(getattr(row, "last_embedding_update_ts", None))
            removed_body = 0
            removed_face = 0

            if last_date is None:
                # Unknown/brand new -> start clean (bootstrap will fill)
                removed_body = int(b0.shape[0] + b1.shape[0])
                removed_face = int(f0.shape[0] + f1.shape[0])
                b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            else:
                try:
                    gap_days = int((today - last_date).days)
                except Exception:
                    gap_days = 0

                if gap_days >= self.reset_if_gap_days_ge:
                    removed_body = int(b0.shape[0] + b1.shape[0])
                    removed_face = int(f0.shape[0] + f1.shape[0])
                    b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                elif gap_days >= 1:
                    # New day: clear today's slot only
                    if slot_idx == 0:
                        removed_body = int(b0.shape[0])
                        removed_face = int(f0.shape[0])
                        b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                        f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    else:
                        removed_body = int(b1.shape[0])
                        removed_face = int(f1.shape[0])
                        b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                        f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)

            # Active slot pointers for today
            if slot_idx == 0:
                b_active, b_other = b0, b1
                f_active, f_other = f0, f1
            else:
                b_active, b_other = b1, b0
                f_active, f_other = f1, f0

            body_added = 0
            face_added = 0

            # BODY: fill today's slot; bootstrap-fill other slot only if empty
            if self.update_body and body_list:
                avail = max(0, self.slot_rows - int(b_active.shape[0]))
                take = min(len(body_list), avail)
                if take > 0:
                    add = l2_normalize_rows(np.stack(body_list[:take], axis=0))
                    b_active = np.concatenate([b_active, add], axis=0)
                    del body_list[:take]
                    body_added += int(take)

                if len(body_list) > 0 and int(b_other.shape[0]) == 0:
                    avail2 = max(0, self.slot_rows - int(b_other.shape[0]))
                    take2 = min(len(body_list), avail2)
                    if take2 > 0:
                        add2 = l2_normalize_rows(np.stack(body_list[:take2], axis=0))
                        b_other = np.concatenate([b_other, add2], axis=0)
                        del body_list[:take2]
                        body_added += int(take2)

                if body_list:
                    body_list.clear()

            # FACE: fill today's slot; bootstrap-fill other slot only if empty
            if self.update_face and face_list:
                avail = max(0, self.slot_rows - int(f_active.shape[0]))
                take = min(len(face_list), avail)
                if take > 0:
                    add = l2_normalize_rows(np.stack(face_list[:take], axis=0))
                    f_active = np.concatenate([f_active, add], axis=0)
                    del face_list[:take]
                    face_added += int(take)

                if len(face_list) > 0 and int(f_other.shape[0]) == 0:
                    avail2 = max(0, self.slot_rows - int(f_other.shape[0]))
                    take2 = min(len(face_list), avail2)
                    if take2 > 0:
                        add2 = l2_normalize_rows(np.stack(face_list[:take2], axis=0))
                        f_other = np.concatenate([f_other, add2], axis=0)
                        del face_list[:take2]
                        face_added += int(take2)

                if face_list:
                    face_list.clear()

            # Put slots back into correct order
            if slot_idx == 0:
                b0, b1 = b_active, b_other
                f0, f1 = f_active, f_other
            else:
                b1, b0 = b_active, b_other
                f1, f0 = f_active, f_other

            upd_body = self._combine_slots(b0, b1)
            upd_face = self._combine_slots(f0, f1)

            body_after = int(upd_body.shape[0])
            face_after = int(upd_face.shape[0])

            changed = (body_added > 0 or face_added > 0 or removed_body > 0 or removed_face > 0)

            # Get member name for logging (best-effort)
            member_name = ""
            try:
                mrow = session.execute(
                    select(self.MemberRow.first_name, self.MemberRow.last_name, self.MemberRow.member_number).where(self.MemberRow.id == mid)
                ).first()
                if mrow:
                    first = str(mrow[0] or "").strip()
                    last = str(mrow[1] or "").strip()
                    member_name = (first + " " + last).strip() if last else first
                    if not member_name:
                        member_name = str(mrow[2] or "").strip()
            except Exception:
                member_name = ""

            if changed:
                try:
                    if self.update_body:
                        row.body_embeddings_raw = encode_bank_gzip_npy(l2_normalize_rows(upd_body))
                        if body_after > 0:
                            row.body_embedding = l2_normalize(np.mean(upd_body, axis=0)).tolist()

                    if self.update_face:
                        row.face_embeddings_raw = encode_bank_gzip_npy(l2_normalize_rows(upd_face))
                        if face_after > 0:
                            row.face_embedding = l2_normalize(np.mean(upd_face, axis=0)).tolist()

                    row.last_embedding_update_ts = now_dt
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"[DB-UPD] commit failed member_id={mid} camera_id={cid}: {e}")
                    return

            # Update can_accept state
            avail_body_now = 0
            avail_face_now = 0
            if self.slot_rows > 0:
                if self.update_body:
                    active_now = (b0 if slot_idx == 0 else b1)
                    avail_body_now = max(0, self.slot_rows - int(active_now.shape[0]))
                if self.update_face:
                    active_now = (f0 if slot_idx == 0 else f1)
                    avail_face_now = max(0, self.slot_rows - int(active_now.shape[0]))

            is_full = (not self.update_body or avail_body_now <= 0) and (not self.update_face or avail_face_now <= 0)
            with self._state_lock:
                self._full_state[key] = {"day_ord": int(today.toordinal()), "full": bool(is_full)}

            # meta used
            metas = self._meta_buf.get(key, [])
            track_ids = ",".join(str(int(m.track_id)) for m in metas[:20]) if metas else ""
            face_sim_max = max((float(m.face_sim) for m in metas), default=0.0)
            if metas:
                metas.clear()

            if self.logger is not None and (body_added or face_added or removed_body or removed_face):
                self.logger.log(
                    ts=float(time.time()),
                    member_id=int(mid),
                    name=str(member_name),
                    camera_id=int(cid),
                    tracks=track_ids,
                    face_sim_max=float(face_sim_max),
                    body_added=int(body_added),
                    body_removed=int(removed_body),
                    body_before=int(body_before),
                    body_after=int(body_after),
                    face_added=int(face_added),
                    face_removed=int(removed_face),
                    face_before=int(face_before),
                    face_after=int(face_after),
                )

                try:
                    print(
                        f"[DB-UPD] member_id={mid} cam={cid} | "
                        f"body +{body_added}/-{removed_body} ({body_before}->{body_after}) | "
                        f"face +{face_added}/-{removed_face} ({face_before}->{face_after})"
                    )
                except Exception:
                    pass

        self._last_flush_ts[key] = float(time.time())

    def _reset_all_to_empty(self):
        """Dangerous: clears all banks. Not enabled by default."""
        with self.Session() as session:
            rows = session.execute(select(self.MemberEmbeddingRow)).scalars().all()
            for row in rows:
                row.body_embeddings_raw = None
                row.face_embeddings_raw = None
                row.body_embedding = None
                row.face_embedding = None
                row.last_embedding_update_ts = datetime.now(timezone.utc)
            session.commit()


# ----------------------------
# DB gallery types
# ----------------------------
@dataclass
class PersonEntry:
    member_id: int
    name: str
    camera_id: int
    body_bank: np.ndarray | None
    body_centroid: np.ndarray | None


@dataclass
class FaceGallery:
    member_ids: list[int]
    names: list[str]
    mat: np.ndarray

    def is_empty(self) -> bool:
        return (not self.names) or (self.mat is None) or (self.mat.size == 0)


def best_face_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> tuple[int | None, str | None, float, float]:
    """Returns: (best_member_id, best_label, best_sim, second_sim)"""
    if emb is None or face_gallery is None or face_gallery.is_empty():
        return None, None, 0.0, 0.0
    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, None, 0.0, 0.0

    sims = face_gallery.mat @ q
    if sims.size == 0:
        return None, None, 0.0, 0.0

    if sims.size == 1:
        return int(face_gallery.member_ids[0]), str(face_gallery.names[0]), float(sims[0]), 0.0

    idxs = np.argpartition(sims, -2)[-2:]
    i1, i2 = int(idxs[0]), int(idxs[1])
    if sims[i2] > sims[i1]:
        i1, i2 = i2, i1
    best_idx, second_idx = i1, i2
    return (
        int(face_gallery.member_ids[best_idx]),
        str(face_gallery.names[best_idx]),
        float(sims[best_idx]),
        float(sims[second_idx]),
    )


def build_galleries_from_db(
    db_url: str,
    active_only: bool = True,
    max_bank_per_entry: int = 0,
) -> tuple[dict[int, list[PersonEntry]], FaceGallery, dict[str, int]]:
    """
    Loads from:
      - member_embeddings (per member_id + camera_id)
      - members (name/status)
    Returns:
      - people_by_cam[camera_id] -> list[PersonEntry] for body matching
      - face_gallery aggregated per member across cameras
      - name_to_member_id mapping
    """
    Base = declarative_base()

    class MemberRow(Base):
        __tablename__ = "members"
        id = Column(Integer, primary_key=True)

        # Your schema:
        member_number = Column(String)
        first_name = Column(String)
        last_name = Column(String)
        is_active = Column(Boolean)

    class MemberEmbeddingRow(Base):
        __tablename__ = "member_embeddings"
        id = Column(BigInteger, primary_key=True)

        member_id = Column(Integer, nullable=False)
        camera_id = Column(Integer, nullable=False)

        if Vector is not None:
            body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
        else:
            body_embedding = Column(ARRAY(Float), nullable=True)
            face_embedding = Column(ARRAY(Float), nullable=True)

        body_embeddings_raw = Column(LargeBinary, nullable=True)
        face_embeddings_raw = Column(LargeBinary, nullable=True)

        last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)

    people_by_cam: dict[int, list[PersonEntry]] = defaultdict(list)
    face_vecs_by_member: dict[int, list[np.ndarray]] = defaultdict(list)
    member_id_to_name: dict[int, str] = {}
    name_to_member_id: dict[str, int] = {}

    with Session() as session:
        stmt = (
            select(
                MemberEmbeddingRow.member_id,
                MemberEmbeddingRow.camera_id,
                MemberRow.member_number,
                MemberRow.first_name,
                MemberRow.last_name,
                MemberRow.is_active,
                MemberEmbeddingRow.body_embedding,
                MemberEmbeddingRow.face_embedding,
                MemberEmbeddingRow.body_embeddings_raw,
                MemberEmbeddingRow.face_embeddings_raw,
            )
            .join(MemberRow, MemberRow.id == MemberEmbeddingRow.member_id)
        )

        rows = session.execute(stmt).all()

        for r in rows:
            mid = int(r.member_id)
            cam_id = int(r.camera_id)
            first = str(r.first_name or "").strip()
            last = str(r.last_name or "").strip()
            # Display label used for drawing/logging
            name = (first + " " + last).strip() if last else first

            # Active filtering (members.is_active)
            try:
                is_active = bool(r.is_active) if (r.is_active is not None) else True
            except Exception:
                is_active = True
            if active_only and (not is_active):
                continue

            # Fallbacks if name is empty
            if not name:
                name = str(r.member_number or "").strip()
            if not name:
                name = f"member_{mid}"

            if name and name not in name_to_member_id:
                name_to_member_id[name] = mid
            member_id_to_name[mid] = name

            body_cent = _as_vec512(r.body_embedding)
            face_cent = _as_vec512(r.face_embedding)

            body_bank = decode_bank_gzip_npy(r.body_embeddings_raw)
            face_bank = decode_bank_gzip_npy(r.face_embeddings_raw)

            if max_bank_per_entry and max_bank_per_entry > 0:
                m = int(max_bank_per_entry)
                if body_bank is not None and body_bank.shape[0] > m:
                    body_bank = body_bank[:m]
                if face_bank is not None and face_bank.shape[0] > m:
                    face_bank = face_bank[:m]

            if body_cent is None and body_bank is not None and body_bank.shape[0] > 0:
                body_cent = l2_normalize(np.mean(body_bank, axis=0))

            if face_cent is None and face_bank is not None and face_bank.shape[0] > 0:
                face_cent = l2_normalize(np.mean(face_bank, axis=0))

            if body_bank is None and body_cent is not None:
                body_bank = body_cent.reshape(1, -1).astype(np.float32)

            if body_bank is not None and body_bank.ndim == 2 and body_bank.shape[1] == EXPECTED_DIM:
                body_bank = l2_normalize_rows(body_bank.astype(np.float32))

            people_by_cam[cam_id].append(
                PersonEntry(
                    member_id=mid,
                    name=name,
                    camera_id=cam_id,
                    body_bank=body_bank,
                    body_centroid=body_cent,
                )
            )

            if face_cent is not None:
                face_vecs_by_member[mid].append(face_cent.astype(np.float32))

    fg_member_ids: list[int] = []
    fg_names: list[str] = []
    fg_vecs: list[np.ndarray] = []

    for mid, vecs in face_vecs_by_member.items():
        if not vecs:
            continue
        mat = l2_normalize_rows(np.stack(vecs, axis=0))
        v = l2_normalize(np.mean(mat, axis=0))
        fg_member_ids.append(int(mid))
        fg_names.append(str(member_id_to_name.get(mid, f"member_{mid}")))
        fg_vecs.append(v.astype(np.float32))

    face_mat = (
        l2_normalize_rows(np.stack(fg_vecs, axis=0))
        if fg_vecs
        else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
    )
    face_gallery = FaceGallery(fg_member_ids, fg_names, face_mat)

    total_body_entries = sum(len(v) for v in people_by_cam.values())
    print(f"[DB] Loaded member_embeddings: body_entries={total_body_entries} | face_identities={len(fg_names)}")
    return people_by_cam, face_gallery, name_to_member_id


class GalleryManager:
    """Thread-safe DB gallery with optional periodic refresh."""
    def __init__(self, args):
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()

        self.people_by_cam: dict[int, list[PersonEntry]] = defaultdict(list)
        self.face_gallery: FaceGallery = FaceGallery([], [], np.zeros((0, EXPECTED_DIM), dtype=np.float32))
        self.name_to_member_id: dict[str, int] = {}

        self.last_load_ts: float = 0.0
        self.load(args)

    def load(self, args) -> None:
        active_only = not bool(getattr(args, "db_include_inactive", False))
        people_by_cam, fg, name_to_mid = build_galleries_from_db(
            args.db_url,
            active_only=active_only,
            max_bank_per_entry=int(getattr(args, "db_max_bank", 0) or 0),
        )
        with self._lock:
            self.people_by_cam = people_by_cam
            self.face_gallery = fg
            self.name_to_member_id = name_to_mid
            self.last_load_ts = time.time()

    def maybe_reload(self, args) -> None:
        period = float(getattr(args, "db_refresh_seconds", 0.0) or 0.0)
        if period <= 0:
            return
        now = time.time()
        if (now - self.last_load_ts) < period:
            return
        if not self._reload_lock.acquire(blocking=False):
            return
        try:
            if (time.time() - self.last_load_ts) < period:
                return
            try:
                self.load(args)
                print("[DB] Gallery reloaded")
            except Exception as e:
                print("[DB] reload failed:", e)
        finally:
            self._reload_lock.release()

    def snapshot(self) -> tuple[dict[int, list[PersonEntry]], FaceGallery, dict[str, int]]:
        with self._lock:
            return dict(self.people_by_cam), self.face_gallery, dict(self.name_to_member_id)



# ----------------------------
# Tracking helpers (IoU)
# ----------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def ioa_xyxy(inner, outer) -> float:
    """Intersection over AREA(inner)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    inner_area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
    return float(inter / inner_area) if inner_area > 0 else 0.0


def _point_in_xyxy(px: float, py: float, box) -> bool:
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


# ----------------------------
# ✅ One-to-one face-to-track assignment
# ----------------------------
def assign_faces_to_tracks_one_to_one(
    recognized_faces: List[Dict[str, Any]],
    raw_tracks: List[Dict[str, Any]],
    args,
) -> Dict[int, Dict[str, Any]]:
    """Greedy one-to-one matching between recognized faces and person tracks. Returns {tid: face_match_dict}"""
    if not recognized_faces or not raw_tracks:
        return {}

    candidates: List[Tuple[float, int, int]] = []

    for fi, fm in enumerate(recognized_faces):
        fbox = fm.get("bbox", None)
        if not fbox:
            continue

        fx1, fy1, fx2, fy2 = map(float, fbox)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        f_area = max(1.0, fw * fh)

        fc_x = 0.5 * (fx1 + fx2)
        fc_y = 0.5 * (fy1 + fy2)

        for ti, tr in enumerate(raw_tracks):
            x1, y1, x2, y2 = tr["bbox"]
            t_xyxy = (float(x1), float(y1), float(x2), float(y2))
            p_area = float(max(1.0, (x2 - x1) * (y2 - y1)))

            if bool(getattr(args, "face_center_in_person", False)) and not _point_in_xyxy(fc_x, fc_y, t_xyxy):
                continue

            top_limit = float(y1) + float(getattr(args, "face_center_y_max_ratio", 0.70)) * float(max(1, (y2 - y1)))
            if fc_y > top_limit:
                continue

            link_mode = str(getattr(args, "face_link_mode", "ioa"))
            link = ioa_xyxy(fbox, t_xyxy) if link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)

            if link < float(getattr(args, "face_iou_link", 0.35)):
                continue

            ratio = float(f_area / p_area)

            min_ratio = float(getattr(args, "min_face_area_ratio", 0.0) or 0.0)
            if min_ratio > 0.0 and ratio < min_ratio:
                continue

            sim = float(fm.get("sim", 0.0))
            score = float(link) + 0.25 * sim + 0.20 * ratio
            candidates.append((score, fi, ti))

    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_faces = set()
    used_tracks = set()
    out: Dict[int, Dict[str, Any]] = {}

    for score, fi, ti in candidates:
        if fi in used_faces or ti in used_tracks:
            continue
        used_faces.add(fi)
        used_tracks.add(ti)
        tid = int(raw_tracks[ti]["tid"])
        out[tid] = recognized_faces[fi]

    return out


# ----------------------------
# IOU tracker fallback
# ----------------------------
class IOUTrack:
    def __init__(self, tlwh, tid):
        self.tlwh = np.array(tlwh, dtype=np.float32)
        self.tid = int(tid)
        self.miss = 0


class IOUTracker:
    """Minimal IoU tracker fallback (stable IDs)."""
    def __init__(self, max_miss=5, iou_thresh=0.3):
        self.tracks: list[IOUTrack] = []
        self.next_id = 1
        self.max_miss = int(max_miss)
        self.iou_thresh = float(iou_thresh)

    def update(self, dets_tlwh_conf: np.ndarray):
        dets = np.asarray(dets_tlwh_conf, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] < 4:
            dets = dets.reshape((0, 5)).astype(np.float32)

        assigned = set()
        for tr in self.tracks:
            tr.miss += 1
            t_x, t_y, t_w, t_h = tr.tlwh
            t_xyxy = np.array([t_x, t_y, t_x + t_w, t_y + t_h], dtype=np.float32)
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                x, y, w, h = d[:4]
                d_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                s = iou_xyxy(t_xyxy, d_xyxy)
                if s > best_iou:
                    best_iou, best_j = s, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                tr.tlwh = dets[best_j][:4]
                tr.miss = 0
                assigned.add(best_j)

        for j, d in enumerate(dets):
            if j in assigned:
                continue
            self.tracks.append(IOUTrack(d[:4], self.next_id))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        outs = []
        for t in self.tracks:
            x, y, w, h = t.tlwh

            class Dummy:
                pass

            o = Dummy()
            o.track_id = t.tid
            o.is_confirmed = lambda: True
            o.to_tlbr = lambda: (x, y, x + w, y + h)
            o.det_conf = None
            o.last_detection = None
            o.time_since_update = 0
            outs.append(o)

        return outs


# ----------------------------
# Identity smoothing (anti-flicker)
# ----------------------------
def update_track_identity(
    state: dict,
    tid: int,
    candidates: list[tuple[str, float, str]],
    decay: float,
    min_score: float,
    margin: float,
    ttl_reset: int,
    w_face: float,
    w_body: float,
) -> tuple[str, float, dict]:
    """Exponentially decayed per-name score accumulator. Body votes accepted only when they agree with face label."""
    entry = state.setdefault(
        tid,
        {
            "scores": defaultdict(float),
            "last": "",
            "ttl": 0,
            "face_vis_ttl": 0,
            "last_face_label": "",
            "last_face_sim": 0.0,
            "assigned_name": "",
            "assigned_member_id": -1,
            "assigned_score": 0.0,
            "last_seen_frame": -1,
        },
    )
    scores = entry["scores"]

    for k in list(scores.keys()):
        scores[k] *= float(decay)
        if scores[k] < 1e-6:
            del scores[k]

    for label, sim, src in candidates:
        if not label:
            continue
        w = w_face if src == "face" else w_body
        scores[label] += max(0.0, float(sim)) * float(w)

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    else:
        top_label, top_score, second_score = "", 0.0, 0.0

    if top_label and (top_score >= min_score) and (entry["last"] == top_label or (top_score - second_score) >= margin):
        entry["last"] = top_label
        entry["ttl"] = int(ttl_reset)
    else:
        if entry["ttl"] > 0:
            entry["ttl"] -= 1
        else:
            entry["last"] = ""

    return entry["last"], float(scores.get(entry["last"], 0.0)), entry


# ----------------------------
# Models init
# ----------------------------
def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str, ort_log: bool):
    if not use_face:
        return None
    if not INSIGHT_OK:
        print("[WARN] insightface not installed; face recognition disabled.")
        return None
    try:
        is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
        cuda_ok = _cuda_ep_loadable()

        if ort is not None and ort_log:
            try:
                print(f"[INFO] ORT available providers: {ort.get_available_providers()}")
            except Exception:
                pass

        providers = ["CPUExecutionProvider"]
        if face_provider == "cuda":
            if cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[INFO] Requested CUDA EP, but not loadable. Using CPU.")
        elif face_provider == "auto":
            if is_cuda and cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        app = FaceAnalysis(name=face_model, providers=providers)
        ctx_id = 0 if providers[0].startswith("CUDA") else -1
        try:
            app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
        except TypeError:
            app.prepare(ctx_id=ctx_id)
        print(f"[INIT] InsightFace ready (model={face_model}, providers={providers}).")
        return app
    except Exception as e:
        print("[WARN] InsightFace init failed:", e)
        return None


def _yolo_forward_safe(yolo, frame, args):
    """Guarded YOLO call with a per-model lock and FP16->FP32 fallback."""
    with _yolo_lock, torch.inference_mode():
        try:
            return yolo(
                frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=args.device,
                half=args.half,
                imgsz=int(args.yolo_imgsz) if int(args.yolo_imgsz) > 0 else None,
            )
        except TypeError:
            return yolo(
                frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=args.device,
                half=args.half,
            )
        except Exception as e:
            if args.half:
                print("[YOLO] FP16 failed, retrying in FP32 once:", e)
                args.half = False
                return yolo(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                    device=args.device,
                    half=False,
                )
            raise


def extract_body_embeddings_batch(extractor, crops_bgr: List[np.ndarray], device_is_cuda: bool, use_half: bool) -> Optional[np.ndarray]:
    """Batched TorchReID embedding extraction for multiple crops in one call. Returns normalized [N,512]."""
    if extractor is None or not crops_bgr:
        return None

    crops_rgb: List[np.ndarray] = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            crops_rgb.append(np.zeros((1, 1, 3), dtype=np.uint8))
            continue
        crops_rgb.append(_to_rgb(c))

    with _reid_lock, torch.inference_mode():
        if device_is_cuda and use_half:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = extractor(crops_rgb)
            except Exception:
                feats = extractor(crops_rgb)
        else:
            feats = extractor(crops_rgb)

    try:
        if isinstance(feats, (list, tuple)):
            feats_arr = []
            for f in feats:
                f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
                feats_arr.append(np.asarray(f, dtype=np.float32).reshape(-1))
            mat = np.stack(feats_arr, axis=0)
        else:
            f = feats.detach().cpu().numpy() if hasattr(feats, "detach") else np.asarray(feats)
            mat = np.asarray(f, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
        if mat.ndim != 2 or mat.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(mat).all():
            return None
        return l2_normalize_rows(mat)
    except Exception:
        return None


def best_body_label_from_emb(
    emb: np.ndarray | None,
    people: list[PersonEntry],
    topk: int = 3,
) -> tuple[str | None, float, float]:
    """TopK-mean cosine match across each person's body_bank."""
    if emb is None:
        return None, 0.0, 0.0
    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, 0.0, 0.0

    k_req = max(1, int(topk))
    scored: list[tuple[str, float]] = []

    for p in people:
        bank = p.body_bank
        if bank is None or bank.size == 0:
            continue
        try:
            sims = bank @ q
        except Exception:
            continue
        if sims.ndim != 1 or sims.size == 0:
            continue
        k = min(k_req, sims.size)
        if k <= 1:
            score = float(np.max(sims))
        else:
            top_vals = np.partition(sims, -k)[-k:]
            score = float(np.mean(top_vals))
        scored.append((p.name, score))

    if not scored:
        return None, 0.0, 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return best_label, float(best_score), float(second_score)


# ----------------------------
# Stream readers (AdaptiveQueueStream)
# ----------------------------
class AdaptiveQueueStream:
    """
    Ordered reader with a bounded queue.
    - When queue is full, drops the oldest frame to keep latency bounded.
    - Fast reconnect with low timeouts.
    """
    def __init__(
        self,
        src: str,
        queue_size: int,
        rtsp_transport: str,
        use_opencv: bool = True,
        freeze_seconds: float = 2.0,
        open_timeout_ms: int = 1000,
        read_timeout_ms: int = 1000,
        reconnect_base_delay: float = 0.5,
        reconnect_max_delay: float = 3.0,
        reconnect_jitter: float = 0.10,
        reconnect_log_interval: float = 2.0,
    ):
        self.src = src
        self.use_opencv = bool(use_opencv)
        self.queue_size = max(1, int(queue_size))
        self.rtsp_transport = str(rtsp_transport or "tcp")

        self.freeze_seconds = float(max(0.5, float(freeze_seconds or 2.0)))
        self.open_timeout_ms = int(max(0, int(open_timeout_ms or 0)))
        self.read_timeout_ms = int(max(0, int(read_timeout_ms or 0)))

        self.reconnect_base_delay = float(max(0.1, float(reconnect_base_delay or 0.5)))
        self.reconnect_max_delay = float(max(self.reconnect_base_delay, float(reconnect_max_delay or 3.0)))
        self.reconnect_jitter = float(max(0.0, min(0.50, float(reconnect_jitter or 0.0))))
        self.reconnect_log_interval = float(max(0.0, float(reconnect_log_interval or 0.0)))

        if isinstance(src, str) and src.lower().startswith("rtsp"):
            try:
                key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
                opt = os.environ.get(key, "") or ""
                us = int(max(0, int(self.read_timeout_ms)) * 1000) if self.read_timeout_ms > 0 else 0

                parts = [p for p in opt.split("|") if p.strip()]
                kv = []
                for p in parts:
                    if ";" in p:
                        k, v = p.split(";", 1)
                        kv.append((k.strip(), v.strip()))
                    else:
                        kv.append((p.strip(), ""))

                def upsert(k: str, v: str) -> None:
                    for i, (kk, vv) in enumerate(kv):
                        if kk == k:
                            kv[i] = (kk, str(v))
                            return
                    kv.append((k, str(v)))

                if us > 0:
                    upsert("stimeout", us)
                    upsert("rw_timeout", us)

                new_opt = "|".join([f"{k};{v}" if v != "" else k for k, v in kv])
                os.environ[key] = new_opt
            except Exception:
                pass

        src_use = src
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            sep = "&" if "?" in src_use else "?"
            src_use = f"{src_use}{sep}rtsp_transport={self.rtsp_transport}"
        self.src_use = src_use

        self.q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.stop_flag = threading.Event()

        self.dropped = 0
        self.read_dropped = 0

        self._cap_lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self._connected = False

        self._last_frame_mono = time.monotonic()
        self._next_reconnect_mono = 0.0
        self._reconnect_failures = 0
        self._last_log_mono = 0.0

        self._open_capture(initial=True)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _log(self, msg: str) -> None:
        if not msg:
            return
        if self.reconnect_log_interval <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_log_mono)) >= float(self.reconnect_log_interval):
            self._last_log_mono = now
            print(msg)

    def _drop_oldest(self) -> None:
        try:
            _ = self.q.get_nowait()
            self.dropped += 1
        except queue.Empty:
            return

    def _drain_queue(self) -> None:
        try:
            while True:
                _ = self.q.get_nowait()
        except queue.Empty:
            return

    def _create_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture()
        try:
            if self.open_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.open_timeout_ms))
        except Exception:
            pass
        try:
            if self.read_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.read_timeout_ms))
        except Exception:
            pass
        return cap

    def _open_capture(self, initial: bool = False) -> bool:
        new_cap = self._create_capture()
        opened = False
        try:
            if self.use_opencv:
                opened = bool(new_cap.open(self.src_use, cv2.CAP_FFMPEG))
            else:
                opened = bool(new_cap.open(self.src_use))
        except Exception:
            opened = False

        if not opened:
            try:
                new_cap.release()
            except Exception:
                pass
            if initial:
                self._log(f"[SRC] cannot open source initially: {self.src}")
            return False

        try:
            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.queue_size))
        except Exception:
            pass

        old_cap: Optional[cv2.VideoCapture] = None
        with self._cap_lock:
            old_cap = self.cap
            self.cap = new_cap
            self._connected = True

        try:
            if old_cap is not None:
                old_cap.release()
        except Exception:
            pass

        self._last_frame_mono = time.monotonic()
        self._drain_queue()
        return True

    def _backoff_delay(self, failures: int) -> float:
        n = max(0, int(failures))
        base = float(self.reconnect_base_delay)
        cap = float(self.reconnect_max_delay)
        try:
            exp = min(max(0, n - 1), 10)
            delay = base * (2.0 ** exp)
        except Exception:
            delay = base

        delay = float(min(cap, max(base, delay)))

        if self.reconnect_jitter > 0:
            j = (random.random() * 2.0 - 1.0) * float(self.reconnect_jitter) * delay
            delay = max(0.0, delay + j)
        return float(delay)

    def _maybe_reconnect(self, reason: str) -> None:
        now = time.monotonic()
        if now < float(self._next_reconnect_mono):
            return

        with self._cap_lock:
            cap = self.cap
            self.cap = None
            self._connected = False

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        ok = self._open_capture(initial=False)
        if ok:
            self._reconnect_failures = 0
            self._next_reconnect_mono = 0.0
            self._log(f"[SRC] reconnected: {self.src} (reason={reason})")
        else:
            self._reconnect_failures += 1
            delay = self._backoff_delay(self._reconnect_failures)
            self._next_reconnect_mono = time.monotonic() + delay
            self._log(
                f"[SRC] reconnect failed: {self.src} (reason={reason}, failures={self._reconnect_failures}, next_retry={delay:.2f}s)"
            )

    def _loop(self):
        while not self.stop_flag.is_set():
            if not self._connected or self.cap is None:
                self._maybe_reconnect(reason="disconnected")
                self.stop_flag.wait(0.02)
                continue

            with self._cap_lock:
                cap = self.cap

            ok = False
            frame = None
            try:
                ok, frame = cap.read() if cap is not None else (False, None)
            except Exception:
                ok, frame = False, None

            now_mono = time.monotonic()

            if ok and frame is not None and getattr(frame, "size", 0) != 0:
                self._last_frame_mono = now_mono
                item = (frame, time.time())
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    self._drop_oldest()
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        self._drop_oldest()
                continue

            if (now_mono - float(self._last_frame_mono)) >= float(self.freeze_seconds):
                self._maybe_reconnect(reason=f"freeze>{self.freeze_seconds:.2f}s")
            else:
                self.stop_flag.wait(0.002)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        try:
            frame, ts = self.q.get(timeout=0.1)
            return True, frame, float(ts)
        except queue.Empty:
            return False, None, 0.0

    def qsize(self) -> int:
        try:
            return int(self.q.qsize())
        except Exception:
            return 0

    def is_opened(self) -> bool:
        return bool(self._connected) and (self.cap is not None)

    def release(self):
        self.stop_flag.set()
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass
        with self._cap_lock:
            cap = self.cap
            self.cap = None
            self._connected = False
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


# ----------------------------
# Live pipeline holders
# ----------------------------
class RenderedFrame:
    """
    Thread-safe latest-frame buffer with optional seq + cached-JPEG support.

    Compatibility:
      - set(frame, meta) and get() behave like the original v2 RenderedFrame:
          get() -> (frame, ts, meta)

    Additions for smoother MJPEG (no impact on tracking logic):
      - wait_for_seq(last_seq, timeout) -> (frame, ts, meta, seq)
      - wait_jpeg(last_ts, timeout, jpeg_quality) -> (jpg_bytes, ts)
      - add_client()/remove_client() to enable/disable JPEG caching
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._frm: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._meta: Dict[str, Any] = {}
        self._seq: int = 0

        # cached JPEG (for MJPEG)
        self._jpeg: Optional[bytes] = None
        self._jpeg_ts: float = 0.0
        self._encode_lock = threading.Lock()
        self._clients: int = 0

    def add_client(self) -> None:
        with self._lock:
            self._clients += 1

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def set(self, frame: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        with self._cond:
            self._frm = frame
            self._ts = time.time()
            self._meta = dict(meta or {})
            self._seq += 1
            # invalidate cached jpeg
            self._jpeg = None
            self._jpeg_ts = 0.0
            self._cond.notify_all()

    def get(self) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        with self._lock:
            return self._frm, self._ts, dict(self._meta)

    def wait_for_seq(self, last_seq: int, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
        last_seq = int(last_seq)
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout=float(timeout))
            return self._frm, float(self._ts), dict(self._meta), int(self._seq)

    def wait_jpeg(self, last_ts: float, timeout: float = 0.5, jpeg_quality: int = 80) -> Tuple[Optional[bytes], float]:
        last_ts = float(last_ts)
        with self._cond:
            if self._ts <= last_ts:
                self._cond.wait(timeout=float(timeout))
            ts = float(self._ts)
            frame = self._frm
            clients = int(self._clients)

        if frame is None or ts <= 0:
            return None, 0.0

        # If nobody is connected, skip CPU work but still advance ts.
        if clients <= 0:
            return None, ts

        # Return cached JPEG if still valid.
        with self._lock:
            if self._jpeg is not None and float(self._jpeg_ts) == ts:
                return self._jpeg, ts

        # Encode once per frame.
        with self._encode_lock:
            with self._lock:
                if self._jpeg is not None and float(self._jpeg_ts) == ts:
                    return self._jpeg, ts
                frame_ref = self._frm
                ts_ref = float(self._ts)

            if frame_ref is None or ts_ref <= 0:
                return None, 0.0

            q = int(max(30, min(95, int(jpeg_quality))))
            ok, enc = cv2.imencode(".jpg", frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return None, ts_ref

            jpg = enc.tobytes()

            with self._lock:
                # Only cache if frame didn't change mid-encode.
                if float(self._ts) == ts_ref:
                    self._jpeg = jpg
                    self._jpeg_ts = ts_ref

            return jpg, ts_ref


# ----------------------------
# CSV SUMMARY REPORT (LIVE writer)
# ----------------------------
@dataclass
class _ActiveSession:
    start_ts: float
    last_seen_ts: float
    conf_max: float = 0.0
    conf_sum: float = 0.0
    conf_n: int = 0


class SummaryReport:
    """
    Builds summary CSV / Excel:
    Member | c1 | c2 | ... | Total time
    Each camera cell: L1 - HH:MM:SS to HH:MM:SS | conf 0.82
    """
    def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")

        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._disabled = False

        self._first_seen: Dict[str, float] = {}
        self._active: Dict[Tuple[str, int], _ActiveSession] = {}

        self._logs: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
        self._total_seconds: Dict[str, float] = defaultdict(float)

    def stop(self) -> None:
        with self._lock:
            self._disabled = True

    def update(
        self,
        cam_id: int,
        present_names: List[str],
        ts: float,
        name_to_conf: Optional[Dict[str, float]] = None,
    ) -> None:
        if ts <= 0:
            ts = time.time()
        cam_id = int(cam_id)

        with self._lock:
            if self._disabled:
                return

            self._close_expired_locked(now_ts=float(ts))

            for nm in present_names or []:
                name = str(nm).strip()
                if not name:
                    continue

                conf = 0.0
                if isinstance(name_to_conf, dict):
                    try:
                        conf = float(name_to_conf.get(name, 0.0) or 0.0)
                    except Exception:
                        conf = 0.0

                if name not in self._first_seen:
                    self._first_seen[name] = float(ts)

                key = (name, cam_id)
                sess = self._active.get(key)
                if sess is None:
                    self._active[key] = _ActiveSession(
                        start_ts=float(ts),
                        last_seen_ts=float(ts),
                        conf_max=float(conf),
                        conf_sum=float(conf),
                        conf_n=1 if conf > 0 else 0,
                    )
                else:
                    sess.last_seen_ts = float(ts)
                    if conf > 0:
                        sess.conf_max = max(float(sess.conf_max), float(conf))
                        sess.conf_sum += float(conf)
                        sess.conf_n += 1

    def _close_expired_locked(self, now_ts: float) -> None:
        if self.gap_seconds <= 0:
            return
        to_close: List[Tuple[str, int, _ActiveSession]] = []
        for (name, cam_id), sess in list(self._active.items()):
            if (float(now_ts) - float(sess.last_seen_ts)) > self.gap_seconds:
                to_close.append((name, cam_id, sess))

        for name, cam_id, sess in to_close:
            self._close_session_locked(name, cam_id, sess)

    def _close_session_locked(self, name: str, cam_id: int, sess: _ActiveSession) -> None:
        st = float(sess.start_ts)
        en = float(sess.last_seen_ts)
        if en < st:
            en = st

        conf_max = float(sess.conf_max or 0.0)

        self._logs[name][int(cam_id)].append((st, en, conf_max))
        self._total_seconds[name] += max(0.0, (en - st))
        self._active.pop((name, int(cam_id)), None)

    def close_all(self) -> None:
        with self._lock:
            for (name, cam_id), sess in list(self._active.items()):
                self._close_session_locked(name, cam_id, sess)

    def _fmt_time(self, ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
        except Exception:
            return ""

    def _fmt_total(self, total_seconds: float) -> str:
        try:
            sec_f = float(total_seconds)
        except Exception:
            sec_f = 0.0

        sec_i = int(round(max(0.0, sec_f)))
        if sec_i == 0 and sec_f > 0.0:
            sec_i = 1

        minutes = int(sec_i // 60)
        seconds = int(sec_i % 60)

        m_word = "minute" if minutes == 1 else "minutes"
        s_word = "second" if seconds == 1 else "seconds"
        return f"{minutes} {m_word} {seconds} {s_word}"

    def _snapshot_for_export(
        self,
        include_active: bool,
    ) -> Tuple[List[str], Dict[str, Dict[int, List[Tuple[float, float, float]]]], Dict[str, float]]:
        with self._lock:
            items = list(self._first_seen.items())
            logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
            totals = dict(self._total_seconds)
            active_items = list(self._active.items()) if include_active else []

        items.sort(key=lambda kv: kv[1])
        names_order = [nm for nm, _ in items]

        if include_active and active_items:
            for (name, cam_id), sess in active_items:
                st = float(sess.start_ts)
                en = float(sess.last_seen_ts)
                if en < st:
                    en = st
                conf_max = float(sess.conf_max or 0.0)

                logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en, conf_max))
                totals[name] = float(totals.get(name, 0.0)) + max(0.0, (en - st))
                if name not in names_order:
                    names_order.append(name)

        return names_order, logs, totals

    @staticmethod
    def _fallback_path(path: str, suffix: str) -> str:
        try:
            p = Path(path)
            return str(p.with_name(p.stem + suffix + p.suffix))
        except Exception:
            return str(path) + suffix

    def _write_snapshot_csv(
        self,
        path: str,
        names_order: List[str],
        logs: Dict[str, Dict[int, List[Tuple[float, float, float]]]],
        totals: Dict[str, float],
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]

        tmp_path = path + ".tmp"
        fallback_path = self._fallback_path(path, "_live")

        def _write_csv(pth: str) -> None:
            with open(pth, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for name in names_order:
                    row = [name]
                    cam_map = logs.get(name, {})
                    for cam_id in range(self.num_cams):
                        entries = cam_map.get(cam_id, [])
                        lines = []
                        for idx, (st, en, conf) in enumerate(entries, start=1):
                            lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)} | conf {float(conf):.2f}")
                        row.append("\n".join(lines))
                    row.append(self._fmt_total(totals.get(name, 0.0)))
                    w.writerow(row)

        with self._write_lock:
            try:
                _write_csv(tmp_path)
            except Exception as e:
                print("[CSV] write tmp failed:", e)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return

            try:
                os.replace(tmp_path, path)
                return
            except Exception as e:
                print("[CSV] replace failed (file may be locked). Writing fallback snapshot:", e)
                try:
                    os.replace(tmp_path, fallback_path)
                    return
                except Exception:
                    try:
                        _write_csv(fallback_path)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        return
                    except Exception as e2:
                        print("[CSV] fallback write failed:", e2)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass

    def write_csv_live(self, path: str) -> None:
        names_order, logs, totals = self._snapshot_for_export(include_active=True)
        self._write_snapshot_csv(path, names_order, logs, totals)

    def write_csv(self, path: str) -> None:
        self.close_all()
        names_order, logs, totals = self._snapshot_for_export(include_active=False)
        self._write_snapshot_csv(path, names_order, logs, totals)


def live_csv_writer_loop(report: SummaryReport, path: str, interval_s: float, stop_evt: threading.Event) -> None:
    base = float(interval_s)
    if base <= 0:
        return
    backoff = base
    while not stop_evt.is_set():
        try:
            report.write_csv_live(path)
            backoff = base
        except Exception as e:
            print("[WARN] Live CSV write failed:", e)
            backoff = min(60.0, max(base, backoff * 2.0))
        stop_evt.wait(backoff)


# ----------------------------
# Name de-duplication + global multi-camera ownership
# ----------------------------
@dataclass
class DrawItem:
    tid: int
    bbox: Tuple[int, int, int, int]
    name: str
    member_id: int
    face_sim: float
    stable_score: float
    det_conf: Optional[float]
    face_hit: bool


def _box_area_xyxy(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return int(max(0, x2 - x1) * max(0, y2 - y1))


def _priority_tuple(it: DrawItem) -> tuple:
    return (
        1 if it.face_hit else 0,
        float(it.face_sim),
        float(it.stable_score),
        float(it.det_conf or 0.0),
        float(_box_area_xyxy(it.bbox)),
    )


def deduplicate_draw_items(items: List[DrawItem], iou_thresh: float) -> List[DrawItem]:
    """Deduplicate ONLY known-name items. Unknown are kept."""
    if not items:
        return []
    known = [it for it in items if it.name]
    unknown = [it for it in items if not it.name]
    if not known:
        return unknown

    groups: Dict[str, List[DrawItem]] = defaultdict(list)
    for it in known:
        groups[it.name].append(it)

    kept: List[DrawItem] = []
    for _name, group in groups.items():
        group_sorted = sorted(group, key=_priority_tuple, reverse=True)
        if float(iou_thresh) <= 0.0:
            kept.append(group_sorted[0])
            continue

        selected: List[DrawItem] = []
        for it in group_sorted:
            ok = True
            for s in selected:
                if iou_xyxy(it.bbox, s.bbox) >= float(iou_thresh):
                    ok = False
                    break
            if ok:
                selected.append(it)
        kept.extend(selected)

    kept.extend(unknown)
    return kept


class GlobalNameOwner:
    def __init__(self, hold_seconds: float = 0.5, switch_margin: float = 0.02):
        self.hold_seconds = float(max(0.0, hold_seconds))
        self.switch_margin = float(max(0.0, switch_margin))
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, Any]] = {}

    def _cleanup(self, now: float) -> None:
        if self.hold_seconds <= 0:
            return
        dead = []
        for name, st in self._state.items():
            ts = float(st.get("ts", 0.0))
            if (now - ts) > self.hold_seconds:
                dead.append(name)
        for name in dead:
            self._state.pop(name, None)

    def allow(self, name: str, sid: int, score: float) -> bool:
        if not name:
            return False
        now = time.time()
        with self._lock:
            self._cleanup(now)

            st = self._state.get(name)
            if st is None:
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True

            owner_sid = int(st.get("sid", -1))
            owner_score = float(st.get("score", 0.0))
            owner_ts = float(st.get("ts", 0.0))

            if owner_sid == int(sid):
                st["score"] = max(owner_score, float(score))
                st["ts"] = now
                return True

            if self.hold_seconds > 0 and (now - owner_ts) > self.hold_seconds:
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True

            if float(score) > (owner_score + self.switch_margin):
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True

            return False


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        "YOLO -> DeepSORT (TorchReID) with member_embeddings DB gallery + 2-slot rolling embedding updates.",
        conflict_handler="resolve",
    )

    ap.add_argument("--src", nargs="+", required=True, help="Video sources (RTSP/RTMP/HTTP/file).")

    # Map streams to DB camera_id
    ap.add_argument("--camera-ids", nargs="+", type=int, default=[],
                    help="DB camera_ids aligned with --src order. If omitted, defaults to 1..N.")

    # DB
    ap.add_argument("--use-db", action="store_true", help="Enable DB gallery (required for this script).")
    ap.add_argument("--db-url", default="", help="SQLAlchemy DB URL (postgresql://...).")
    ap.add_argument("--db-refresh-seconds", type=float, default=30.0, help="Reload DB gallery every N seconds (0=off).")
    ap.add_argument("--db-max-bank", type=int, default=0, help="Max embeddings per entry to load from *_embeddings_raw (0=all).")
    ap.add_argument("--db-include-inactive", action="store_true", help="Include members where status != 'active'.")

    # ✅ Embedding update
    ap.add_argument("--update-db-embeddings", action="store_true",
                    help="Update member_embeddings.*_embeddings_raw banks when face match is strong.")
    ap.add_argument("--update-face-sim-thresh", type=float, default=0.75,
                    help="Minimum face similarity to sample embeddings for DB update. Default=0.75.")
    ap.add_argument("--embeddings-slot-mb", type=float, default=0.5,
                    help="Slot size in MB. Total bank = 2 * slot_mb. Default=0.5 => 1.0MB total.")
    ap.add_argument("--embeddings-reset-if-gap-days", type=int, default=2,
                    help="If last update is >= this many days ago, clear BOTH slots (default=2).")
    ap.add_argument("--embeddings-min-sample-seconds", type=float, default=0.5,
                    help="Min seconds between sampling embeddings per (member,camera).")
    ap.add_argument("--embeddings-flush-seconds", type=float, default=10.0,
                    help="Flush buffered embeddings to DB at least every N seconds (per member,camera).")
    ap.add_argument("--embeddings-log-csv", default="",
                    help="Append-only CSV log for embedding updates (auto if empty).")
    ap.add_argument("--embeddings-samples-log-csv", default="",
                    help="CSV log for each accepted sample (auto if empty).")
    ap.add_argument("--no-update-face-bank", action="store_true", help="Do not update face_embeddings_raw/face_embedding.")
    ap.add_argument("--no-update-body-bank", action="store_true", help="Do not update body_embeddings_raw/body_embedding.")

    # ✅ Gate embedding extraction on InsightFace det_score (your requirement)
    ap.add_argument("--embed-extract-face-sim-thresh", type=float, default=0.75,
                    help="Only extract/enqueue embeddings when face_sim >= this. Default=0.75.")
    ap.add_argument("--embed-min-face-det-score", type=float, default=0.75,
                    help="Only consider faces (recognition + DB update) when InsightFace det_score >= this. Default=0.75.")

    # YOLO
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO inference size (0=ultralytics default)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.20)
    ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")

    # Runtime/perf
    ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
    ap.add_argument("--reader", choices=["adaptive"], default="adaptive")
    ap.add_argument("--queue-size", type=int, default=128, help="Queue size for ordered readers")
    ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")

    # Stream reconnect
    ap.add_argument("--stream-freeze-seconds", type=float, default=2.0)
    ap.add_argument("--stream-open-timeout-ms", type=int, default=1000)
    ap.add_argument("--stream-read-timeout-ms", type=int, default=1000)
    ap.add_argument("--stream-reconnect-base-seconds", type=float, default=0.5)
    ap.add_argument("--stream-reconnect-max-seconds", type=float, default=3.0)
    ap.add_argument("--stream-reconnect-jitter", type=float, default=0.10)
    ap.add_argument("--stream-reconnect-log-interval", type=float, default=2.0)

    ap.add_argument("--resize", type=int, nargs=2, default=[0, 0], help="Force resize W H after reading (0 0 = keep)")

    # Adaptive skipping policy
    ap.add_argument("--max-queue-age-ms", type=int, default=1000, help="Drop frames older than this (0=off).")
    ap.add_argument("--max-drain-per-cycle", type=int, default=32, help="Max stale frames to drop per processing cycle.")

    # DeepSORT / TorchReID
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-deepsort", action="store_true", help="Disable DeepSORT even if installed")
    g.add_argument("--use-deepsort", action="store_true", help="Force enable DeepSORT")

    ap.add_argument("--reid-model", default="osnet_x0_25")
    ap.add_argument("--reid-weights", default="", help="Optional TorchReID weights path")
    ap.add_argument("--reid-batch-size", type=int, default=16, help="TorchReID batch size")

    ap.add_argument("--max-age", type=int, default=15)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--nn-budget", type=int, default=200)
    ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
    ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)

    # Body matching
    ap.add_argument("--gallery-thresh", type=float, default=0.99)
    ap.add_argument("--gallery-gap", type=float, default=0.08)
    ap.add_argument("--reid-topk", type=int, default=3)
    ap.add_argument("--min-box-wh", type=int, default=40)

    # Face
    ap.add_argument("--use-face", action="store_true")
    ap.add_argument("--face-model", default="buffalo_l")
    ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
    ap.add_argument("--face-thresh", type=float, default=0.50, help="Minimum face similarity")
    ap.add_argument("--face-gap", type=float, default=0.05, help="Top1-top2 gap required")
    ap.add_argument("--face-every-n", type=int, default=1, help="Run face detector every N frames")
    ap.add_argument("--face-hold-frames", type=int, default=2, help="How long to keep 'face visible' after last linked face match")
    ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--ort-log", action="store_true")

    ap.add_argument("--face-iou-link", type=float, default=0.05, help="Face->person link threshold")
    ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
    ap.add_argument("--face-center-in-person", action="store_true", help="Require face center inside person box")

    # Extra anti-FP heuristics
    ap.add_argument("--min-face-px", type=int, default=24, help="Min face bbox width/height (px) to consider")
    ap.add_argument("--min-face-area-ratio", type=float, default=0.006, help="Min face_area/person_area to accept face link (0=off)")
    ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70, help="Face center must be in top portion of person box (0..1)")
    ap.add_argument("--face-strong-thresh", type=float, default=0.50, help="If face_sim >= this, ignore body conflicts")

    # Identity smoothing knobs
    ap.add_argument("--name-decay", type=float, default=0.85)
    ap.add_argument("--name-min-score", type=float, default=0.60)
    ap.add_argument("--name-margin", type=float, default=0.30)
    ap.add_argument("--name-ttl", type=int, default=20)
    ap.add_argument("--name-face-weight", type=float, default=1.2)
    ap.add_argument("--name-body-weight", type=float, default=0.5)

    # Drawing/ghost control
    ap.add_argument("--draw-only-matched", action="store_true", help="Draw tracks only when matched with a detection this frame")
    ap.add_argument("--min-det-conf", type=float, default=0.45, help="Hide boxes below this detection confidence when drawing")
    ap.add_argument("--iou-max-miss", type=int, default=5, help="Max missed frames for IOU fallback before dropping the track")

    # Duplicate-name suppression / multi-camera ownership
    ap.add_argument("--allow-duplicate-names", action="store_true", help="Allow the same name multiple times in same camera frame")
    ap.add_argument("--dedup-iou", type=float, default=0.0, help="Duplicate-name suppression IoU threshold")
    ap.add_argument("--no-global-unique-names", action="store_true", help="Disable cross-camera name ownership gate")
    ap.add_argument("--global-hold-seconds", type=float, default=0.5)
    ap.add_argument("--global-switch-margin", type=float, default=0.02)
    ap.add_argument("--show-global-id", action="store_true", help="Append DB member_id to labels")

    # tracking display behavior
    ap.add_argument("--hide-unknown", action="store_true", help="Do not draw tracks that have no recognized identity yet.")
    ap.add_argument("--no-persist-names", action="store_true", help="Do not persist recognized name when face is not visible.")
    ap.add_argument("--no-show-track-id", action="store_true", help="Do not append tracker ID (T#) in the label.")

    ap.add_argument(
        "--report-use-drawn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), summary CSV driven only by boxes actually drawn.",
    )

    # Output & view
    ap.add_argument("--show", action="store_true")

    # Save summary CSV
    ap.add_argument("--save-csv", action="store_true", help="Save SUMMARY CSV report.")
    ap.add_argument("--csv", default="detections_summary.csv")
    ap.add_argument("--report-gap-seconds", type=float, default=2.0)
    ap.add_argument("--report-time-format", default="%H:%M:%S")
    ap.add_argument("--csv-live-interval", type=float, default=1.0)

    ap.add_argument("--overlay-fps", action="store_true", help="Draw FPS/lag/queue stats on each stream.")

    # DISPLAY GRID controls
    ap.add_argument("--grid-rows", type=int, default=0)
    ap.add_argument("--grid-cols", type=int, default=0)
    ap.add_argument("--grid-mode", choices=["cover", "contain"], default="cover")
    ap.add_argument("--fullscreen", action="store_true")

    # Save annotated output video (enabled by default)
    ap.add_argument("--no-save-video", action="store_true")
    ap.add_argument("--video-dir", default="saved_videos")
    ap.add_argument("--video-prefix", default="saved_video")
    ap.add_argument("--video-fps", type=float, default=20.0)
    ap.add_argument("--video-fourcc", default="mp4v")
    ap.add_argument("--video-ext", default=".mp4")

    # Reduce storage by resizing SAVED video (display unaffected)
    ap.add_argument("--video-save-height", type=int, default=480)

    # segment video
    ap.add_argument("--video-segment-seconds", type=float, default=3600.0)

    return ap.parse_args(argv)



# ----------------------------
# Per-frame processing
# ----------------------------
def process_one_frame(
    frame_idx: int,
    frame_bgr: np.ndarray,
    sid: int,                 # stream index
    camera_db_id: int,        # DB camera_id
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    people: list[PersonEntry],
    reid_extractor,
    face_app,
    face_gallery: FaceGallery,
    name_to_member_id: dict[str, int],
    identity_state: dict,
    device_is_cuda: bool,
    global_owner: Optional[GlobalNameOwner] = None,
    ts_cap: float = 0.0,
    embed_updater: Optional[EmbeddingDBUpdater] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Optional resize
    rw, rh = int(args.resize[0]), int(args.resize[1])
    if rw > 0 and rh > 0:
        frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

    H, W = frame_bgr.shape[:2]

    # YOLO detect
    tlwh_conf: list[list[float]] = []
    if yolo is not None:
        try:
            res = _yolo_forward_safe(yolo, frame_bgr, args)
            boxes = res[0].boxes if (res and len(res)) else None
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
                cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                keep = cls == 0
                xyxy, conf = xyxy[keep], conf[keep]
                for (x1, y1, x2, y2), c in zip(xyxy, conf):
                    x1f = float(max(0, min(W - 1, x1)))
                    y1f = float(max(0, min(H - 1, y1)))
                    x2f = float(max(0, min(W - 1, x2)))
                    y2f = float(max(0, min(H - 1, y2)))
                    ww = float(max(1.0, x2f - x1f))
                    hh = float(max(1.0, y2f - y1f))
                    if ww < args.min_box_wh or hh < args.min_box_wh:
                        continue
                    tlwh_conf.append([x1f, y1f, ww, hh, float(c)])
        except Exception as e:
            print(f"[SRC {sid}] YOLO error:", e)

    dets_np = np.asarray(tlwh_conf, dtype=np.float32)
    if dets_np.ndim != 2:
        dets_np = dets_np.reshape((0, 5)).astype(np.float32)

    dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

    # Tracker
    out_tracks = []
    if deep_tracker is not None:
        try:
            out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
        except Exception as e:
            print(f"[SRC {sid}] DeepSORT update_tracks error:", e)
            out_tracks = []
    else:
        out_tracks = iou_tracker.update(dets_np)

    # Face detect (every N frames)
    recognized_faces: List[Dict[str, Any]] = []
    do_face = (
        face_app is not None
        and face_gallery is not None
        and (not face_gallery.is_empty())
        and (frame_idx % max(1, int(args.face_every_n)) == 0)
    )
    if do_face:
        try:
            det_min = float(getattr(args, "embed_min_face_det_score", 0.75))
            with _face_lock:
                faces = face_app.get(np.ascontiguousarray(frame_bgr))
            for f in safe_iter_faces(faces):
                bbox = getattr(f, "bbox", None)
                if bbox is None:
                    continue
                b = np.asarray(bbox).reshape(-1)
                if b.size < 4:
                    continue
                fx1, fy1, fx2, fy2 = map(float, b[:4])
                fw = max(0.0, fx2 - fx1)
                fh = max(0.0, fy2 - fy1)
                if fw < float(args.min_face_px) or fh < float(args.min_face_px):
                    continue

                det_score = float(extract_face_det_score(f))
                if det_score < det_min:
                    continue

                emb = extract_face_embedding(f)
                if emb is None:
                    continue
                emb = l2_normalize(np.asarray(emb, dtype=np.float32))

                best_mid, flabel, fsim, fsecond = best_face_top2(emb, face_gallery)
                if flabel is None or best_mid is None:
                    continue

                gap = float(fsim - fsecond)
                if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
                    recognized_faces.append(
                        {
                            "bbox": (fx1, fy1, fx2, fy2),
                            "label": str(flabel),
                            "member_id": int(best_mid),
                            "sim": float(fsim),
                            "second": float(fsecond),
                            "gap": float(gap),
                            "det_score": float(det_score),
                            "emb": emb.astype(np.float32),
                        }
                    )
        except Exception as e:
            print(f"[SRC {sid}] FaceAnalysis error:", e)

    out = frame_bgr.copy()
    tracks_info: List[Dict[str, Any]] = []

    # Mark last seen for cleanup
    for tid in list(identity_state.keys()):
        st = identity_state.get(tid, {})
        if isinstance(st, dict) and "last_seen_frame" not in st:
            st["last_seen_frame"] = -1

    # Collect track boxes first
    raw_tracks: List[Dict[str, Any]] = []
    for t in out_tracks:
        time_since_update = getattr(t, "time_since_update", 0)
        had_match_this_frame = (time_since_update == 0) or (getattr(t, "last_detection", None) is not None)
        if args.draw_only_matched and not had_match_this_frame:
            continue

        try:
            if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
                continue
            if hasattr(t, "to_tlbr"):
                ltrb = t.to_tlbr()
            elif hasattr(t, "to_ltrb"):
                ltrb = t.to_ltrb()
            else:
                ltrb = t.to_tlbr()

            x1, y1, x2, y2 = map(int, ltrb)
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W, x2)))
            y2 = int(max(0, min(H, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
            if tid < 0:
                continue
        except Exception:
            continue

        det_conf = None
        try:
            if hasattr(t, "det_conf") and t.det_conf is not None:
                det_conf = float(t.det_conf)
            elif hasattr(t, "last_detection") and t.last_detection is not None:
                ld = t.last_detection
                if isinstance(ld, (list, tuple)) and len(ld) >= 2:
                    det_conf = float(ld[1])
                elif isinstance(ld, dict):
                    det_conf = float(ld.get("confidence", ld.get("det_conf", 0.0)))
        except Exception:
            det_conf = None

        if args.min_det_conf > 0 and det_conf is not None and det_conf < args.min_det_conf:
            if args.draw_only_matched:
                continue

        raw_tracks.append({"tid": tid, "bbox": (x1, y1, x2, y2), "det_conf": det_conf})

    # One-to-one face assignment
    face_for_tid: Dict[int, Dict[str, Any]] = assign_faces_to_tracks_one_to_one(
        recognized_faces=recognized_faces,
        raw_tracks=raw_tracks,
        args=args,
    )

    # Build tracks_info with face hit results
    for tr in raw_tracks:
        tid = int(tr["tid"])
        x1, y1, x2, y2 = tr["bbox"]
        det_conf = tr.get("det_conf", None)

        face_label, face_sim, face_gap = "", 0.0, 0.0
        face_det_score = 1.0
        face_member_id = -1
        face_emb = None
        face_hit = False

        fm = face_for_tid.get(tid)
        if fm is not None:
            face_hit = True
            face_label = str(fm.get("label", ""))
            face_sim = float(fm.get("sim", 0.0))
            face_gap = float(fm.get("gap", 0.0))
            face_member_id = int(fm.get("member_id", -1))
            face_emb = fm.get("emb", None)
            try:
                face_det_score = float(fm.get("det_score", 1.0))
            except Exception:
                face_det_score = 1.0

        entry = identity_state.setdefault(
            tid,
            {
                "scores": defaultdict(float),
                "last": "",
                "ttl": 0,
                "face_vis_ttl": 0,
                "last_face_label": "",
                "last_face_sim": 0.0,
                "assigned_name": "",
                "assigned_member_id": -1,
                "assigned_score": 0.0,
                "last_seen_frame": -1,
            },
        )
        entry["last_seen_frame"] = int(frame_idx)

        entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - 1)
        if face_hit and face_label:
            entry["face_vis_ttl"] = max(1, int(args.face_hold_frames))
            entry["last_face_label"] = face_label
            entry["last_face_sim"] = float(face_sim)

        tracks_info.append(
            {
                "tid": tid,
                "bbox": (x1, y1, x2, y2),
                "det_conf": det_conf,
                "face_hit": face_hit,
                "face_label": face_label,
                "face_sim": face_sim,
                "face_gap": face_gap,
                "face_det_score": float(face_det_score),
                "face_member_id": int(face_member_id),
                "face_emb": face_emb,
                "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
                "last_face_sim": float(entry.get("last_face_sim", 0.0)),
            }
        )

    # Face winners (snap-back)
    # label -> (winner_tid, sim, member_id)
    face_winners: Dict[str, Tuple[int, float, int]] = {}
    for r in tracks_info:
        if r["face_hit"] and r["face_label"]:
            lab = str(r["face_label"])
            sim = float(r["face_sim"])
            tid = int(r["tid"])
            mid = int(r.get("face_member_id", -1))
            prev = face_winners.get(lab)
            if prev is None or sim > prev[1]:
                face_winners[lab] = (tid, sim, mid)

    active_tids = {int(r["tid"]) for r in tracks_info}

    for lab, (winner_tid, sim, mid) in face_winners.items():
        w_ent = identity_state.get(winner_tid)
        if isinstance(w_ent, dict):
            w_ent["assigned_name"] = lab
            if mid > 0:
                w_ent["assigned_member_id"] = int(mid)
            w_ent["assigned_score"] = max(float(w_ent.get("assigned_score", 0.0)), float(sim))

        for tid in active_tids:
            if tid == winner_tid:
                continue
            ent = identity_state.get(tid)
            if isinstance(ent, dict) and str(ent.get("assigned_name", "")) == lab:
                ent["assigned_name"] = ""
                ent["assigned_member_id"] = -1
                ent["assigned_score"] = 0.0
                try:
                    ent["scores"].clear()
                except Exception:
                    pass
                ent["last"] = ""
                ent["ttl"] = 0

    # Body support: compute ONLY for tracks that have a recognized face hit this frame and pass embedding gates
    min_sim_for_emb = float(getattr(args, "embed_extract_face_sim_thresh", getattr(args, "update_face_sim_thresh", 0.75)))
    min_det_for_emb = float(getattr(args, "embed_min_face_det_score", 0.75))

    face_hit_tracks = [
        r
        for r in tracks_info
        if r["face_hit"]
        and r["face_label"]
        and float(r.get("face_sim", 0.0)) >= min_sim_for_emb
        and float(r.get("face_det_score", 1.0)) >= min_det_for_emb
    ]
    body_by_tid: Dict[int, Tuple[str, float, float]] = {}
    body_emb_by_tid: Dict[int, np.ndarray] = {}

    if face_hit_tracks and (reid_extractor is not None) and people:
        crops: List[np.ndarray] = []
        tids: List[int] = []
        for r in face_hit_tracks:
            x1, y1, x2, y2 = r["bbox"]
            crop = frame_bgr[y1:y2, x1:x2]
            crops.append(crop)
            tids.append(int(r["tid"]))

        bs = 16
        try:
            bs = max(1, int(getattr(args, "reid_batch_size", 16)))
        except Exception:
            bs = 16

        for start in range(0, len(crops), bs):
            chunk = crops[start: start + bs]
            chunk_tids = tids[start: start + bs]
            feats = extract_body_embeddings_batch(reid_extractor, chunk, device_is_cuda=device_is_cuda, use_half=bool(args.half))
            if feats is None:
                continue
            for tid, emb in zip(chunk_tids, feats):
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                body_emb_by_tid[int(tid)] = emb
                blabel, bsim, bsecond = best_body_label_from_emb(emb, people, topk=max(1, int(args.reid_topk)))
                if blabel is None:
                    continue
                body_by_tid[int(tid)] = (str(blabel), float(bsim), float(bsecond))

    # Sample embeddings into DB updater from face winners (strong sim + det_score gate)
    if embed_updater is not None and face_winners:
        ts_use = float(ts_cap) if float(ts_cap or 0.0) > 0 else time.time()
        sim_thresh = float(getattr(args, "update_face_sim_thresh", 0.75))
        det_thresh = float(getattr(args, "embed_min_face_det_score", 0.75))

        for lab, (winner_tid, sim, member_id) in face_winners.items():
            sim_f = float(sim or 0.0)
            if sim_f < sim_thresh:
                continue
            if int(member_id) <= 0:
                continue

            fm = face_for_tid.get(int(winner_tid), None)
            if not isinstance(fm, dict):
                continue
            try:
                det_score = float(fm.get("det_score", 1.0))
            except Exception:
                det_score = 1.0
            if det_score < det_thresh:
                continue

            try:
                if hasattr(embed_updater, "can_accept") and (not embed_updater.can_accept(int(member_id), int(camera_db_id))):
                    continue
            except Exception:
                pass

            face_emb = fm.get("emb", None)
            body_emb = body_emb_by_tid.get(int(winner_tid), None)

            embed_updater.enqueue(
                EmbeddingSample(
                    member_id=int(member_id),
                    name=str(lab),
                    camera_id=int(camera_db_id),
                    track_id=int(winner_tid),
                    ts=float(ts_use),
                    face_sim=float(sim_f),
                    face_det_score=float(det_score),
                    body_emb=body_emb,
                    face_emb=face_emb,
                )
            )

    # Build draw candidates for ALL tracks (known OR unknown)
    draw_candidates: List[DrawItem] = []

    for r in tracks_info:
        tid = int(r["tid"])
        x1, y1, x2, y2 = r["bbox"]

        face_hit = bool(r["face_hit"])
        face_label = str(r["face_label"])
        face_sim = float(r["face_sim"])
        last_face_sim = float(r.get("last_face_sim", 0.0))
        face_mid = int(r.get("face_member_id", -1))

        candidates: List[Tuple[str, float, str]] = []

        # Only face sets the name; body only supports if it agrees
        if face_hit and face_label:
            candidates.append((face_label, face_sim, "face"))

            if tid in body_by_tid:
                b_label, b_sim, b_second = body_by_tid[tid]
                b_gap = float(b_sim - b_second)
                if (b_label == face_label) and (b_sim >= float(args.gallery_thresh)) and (b_gap >= float(args.gallery_gap)):
                    candidates.append((face_label, b_sim, "body"))
                else:
                    if (b_label != face_label) and (b_sim >= max(float(args.gallery_thresh), 0.80)) and (b_gap >= max(float(args.gallery_gap), 0.10)):
                        if face_sim < float(args.face_strong_thresh):
                            candidates = []  # reject weak face if strong body conflict

        stable_name, stable_score, entry = update_track_identity(
            identity_state,
            tid,
            candidates,
            decay=args.name_decay,
            min_score=args.name_min_score,
            margin=args.name_margin,
            ttl_reset=args.name_ttl,
            w_face=args.name_face_weight,
            w_body=args.name_body_weight,
        )

        # Persist identity per track once it becomes stable
        if stable_name:
            prev = str(entry.get("assigned_name", "") or "")
            if (not prev) or (prev == stable_name) or (float(stable_score) > float(entry.get("assigned_score", 0.0))):
                entry["assigned_name"] = str(stable_name)
                # Prefer member_id from face hit; else lookup by name
                if face_hit and face_mid > 0:
                    entry["assigned_member_id"] = int(face_mid)
                else:
                    entry["assigned_member_id"] = int(name_to_member_id.get(str(stable_name), entry.get("assigned_member_id", -1)))
                entry["assigned_score"] = float(stable_score)

        persist_names = not bool(getattr(args, "no_persist_names", False))
        final_name = str(entry.get("assigned_name", "") or "") if persist_names else str(stable_name or "")
        final_mid = int(entry.get("assigned_member_id", -1)) if final_name else -1

        # If not persisting names, only show name when face is "recently visible"
        if (not persist_names) and int(entry.get("face_vis_ttl", 0)) <= 0:
            final_name = ""
            final_mid = -1

        if bool(getattr(args, "hide_unknown", False)) and not final_name:
            continue

        disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))

        draw_candidates.append(
            DrawItem(
                tid=tid,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                name=str(final_name),
                member_id=int(final_mid),
                face_sim=float(disp_face_sim),
                stable_score=float(stable_score),
                det_conf=r.get("det_conf", None),
                face_hit=bool(face_hit),
            )
        )

    # Per-camera duplicate suppression
    if not bool(getattr(args, "allow_duplicate_names", False)):
        draw_final = deduplicate_draw_items(draw_candidates, iou_thresh=float(getattr(args, "dedup_iou", 0.0)))
    else:
        draw_final = draw_candidates

    # Optional: cross-camera global unique names (known names only)
    if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
        gated: List[DrawItem] = []
        for it in draw_final:
            if not it.name:
                gated.append(it)
                continue
            score = float(it.face_sim)
            if global_owner.allow(it.name, sid=int(sid), score=score):
                gated.append(it)
        draw_final = gated

    # Presence for CSV/report must match what is actually DRAWN (known boxes)
    present_conf: Dict[str, float] = {}
    for it in draw_final:
        if it.name:
            try:
                present_conf[it.name] = max(float(present_conf.get(it.name, 0.0)), float(it.face_sim or 0.0))
            except Exception:
                present_conf[it.name] = float(present_conf.get(it.name, 0.0))
    present_names = sorted(present_conf.keys())

    shown = 0
    events: List[Tuple[int, int, int, int, int, str, float, int]] = []

    show_track_id = not bool(getattr(args, "no_show_track_id", False))

    for it in draw_final:
        x1, y1, x2, y2 = it.bbox
        is_known = bool(it.name)
        color = (0, 255, 0) if is_known else (0, 255, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if is_known:
            label_txt = f"{it.name}"
            if bool(getattr(args, "show_global_id", False)) and it.member_id > 0:
                label_txt += f" [{it.member_id}]"
        else:
            label_txt = "Unknown"

        if show_track_id:
            label_txt += f" (T{int(it.tid)})"

        if it.face_hit:
            label_txt += f" | F {float(it.face_sim):.2f}"
        elif is_known and float(it.face_sim) > 0:
            label_txt += f" | F* {float(it.face_sim):.2f}"

        cv2.putText(out, label_txt, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        shown += 1
        events.append((int(it.tid), x1, y1, x2, y2, str(it.name), float(it.face_sim), int(it.member_id)))

    # Cleanup identity_state
    cleanup_frames = max(30, int(getattr(args, "max_age", 15)) + int(getattr(args, "iou_max_miss", 5)) + 10)
    for tid in list(identity_state.keys()):
        st = identity_state.get(tid, {})
        lf = int(st.get("last_seen_frame", -1)) if isinstance(st, dict) else -1
        if lf >= 0 and (int(frame_idx) - lf) > cleanup_frames:
            identity_state.pop(tid, None)

    meta = {
        "tracks": int(len(tracks_info)),
        "shown": int(shown),
        "faces_recognized": int(len(recognized_faces)),
        "do_face": bool(do_face),
        "events": events,
        "present_names": present_names,
        "present_conf": present_conf,
    }
    return out, meta


def processor_thread(
    sid: int,
    camera_db_id: int,
    vs,
    render_store: RenderedFrame,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    gallery_mgr: GalleryManager,
    reid_extractor,
    face_app,
    global_owner: Optional[GlobalNameOwner],
    report: Optional[SummaryReport],
    embed_updater: Optional[EmbeddingDBUpdater],
    debug: bool = False,
):
    frame_idx = 0
    identity_state: dict[int, dict] = {}

    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10

    device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

    while True:
        ok, frame, ts_cap = vs.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        if int(args.max_queue_age_ms) > 0:
            now = time.time()
            age_ms = (now - float(ts_cap)) * 1000.0
            dropped_here = 0
            while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
                try:
                    vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
                except Exception:
                    pass
                ok2, frame2, ts2 = vs.read()
                if not ok2 or frame2 is None:
                    break
                frame, ts_cap = frame2, ts2
                age_ms = (time.time() - float(ts_cap)) * 1000.0
                dropped_here += 1

        try:
            gallery_mgr.maybe_reload(args)
            people_by_cam, face_gallery, name_to_mid = gallery_mgr.snapshot()
            people = people_by_cam.get(int(camera_db_id), [])

            out, meta = process_one_frame(
                frame_idx,
                frame,
                sid,
                int(camera_db_id),
                yolo,
                args,
                deep_tracker,
                iou_tracker,
                people,
                reid_extractor,
                face_app,
                face_gallery,
                name_to_mid,
                identity_state,
                device_is_cuda=device_is_cuda,
                global_owner=global_owner,
                ts_cap=float(ts_cap),
                embed_updater=embed_updater,
            )

            if report is not None:
                ts_use = float(ts_cap) if ts_cap else time.time()
                if bool(getattr(args, "report_use_drawn_only", True)):
                    events = meta.get("events", []) or []
                    conf_map: Dict[str, float] = {}
                    for ev in events:
                        try:
                            nm = str(ev[5] or "")
                            sim = float(ev[6] or 0.0)
                        except Exception:
                            continue
                        if not nm:
                            continue
                        prev = float(conf_map.get(nm, 0.0))
                        if sim > prev:
                            conf_map[nm] = sim
                    names = sorted(conf_map.keys())
                else:
                    names = meta.get("present_names", []) or []
                    conf_map = meta.get("present_conf", {}) or {}
                    if not isinstance(conf_map, dict):
                        conf_map = {}

                report.update(
                    cam_id=int(sid),
                    present_names=list(names),
                    ts=ts_use,
                    name_to_conf=dict(conf_map),
                )

            now = time.time()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
            last_t = now

            if args.overlay_fps:
                qsz = int(vs.qsize()) if hasattr(vs, "qsize") else 0
                dropped_cap = int(getattr(vs, "dropped", 0))
                dropped_read = int(getattr(vs, "read_dropped", 0))
                lag_ms = (now - float(ts_cap)) * 1000.0 if ts_cap else 0.0
                lines = [
                    f"SRC {sid} (cam_id={camera_db_id}) | FPS {fps_ema:.1f} | lag {lag_ms:.0f}ms",
                    f"q {qsz} | drop(cap) {dropped_cap} | drop(stale) {dropped_read}",
                    f"tracks {meta.get('tracks', 0)} | shown {meta.get('shown', 0)} | faces {meta.get('faces_recognized', 0)}",
                ]
                y = 22
                for ln in lines:
                    cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 22

            render_store.set(out, meta={"fps": float(fps_ema), "sid": int(sid), "camera_db_id": int(camera_db_id)})
            frame_idx += 1

        except Exception as e:
            if debug:
                print(f"[PROC {sid}] error:", e)
            time.sleep(0.001)


# ----------------------------
# Video writer helpers (simple segmented writer)
# ----------------------------
def _norm_ext(ext: str) -> str:
    ext = str(ext or "").strip()
    if not ext:
        return ".mp4"
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _open_writer_with_fallback(path: str, fps: float, size_wh: Tuple[int, int], preferred_fourcc: str) -> Optional[cv2.VideoWriter]:
    w, h = int(size_wh[0]), int(size_wh[1])
    fps = float(max(1.0, fps))

    codecs = []
    p = str(preferred_fourcc or "mp4v")[:4]
    codecs.append(p)
    for c in ["mp4v", "avc1", "XVID", "MJPG"]:
        if c not in codecs:
            codecs.append(c)

    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if vw is not None and vw.isOpened():
                print(f"[VIDEO] Writer opened: codec={c} fps={fps} size={w}x{h}")
                return vw
            try:
                if vw is not None:
                    vw.release()
            except Exception:
                pass
        except Exception:
            pass

    print(f"[WARN] Could not open VideoWriter at: {path}")
    return None


class SegmentedVideoWriter:
    def __init__(
        self,
        out_dir: str,
        basename: str = "annotated",
        fps: float = 20.0,
        fourcc: str = "mp4v",
        ext: str = ".mp4",
        segment_seconds: int = 600,
        save_height: int = 480,
    ):
        self.out_dir = out_dir
        self.basename = basename
        self.fps = float(fps)
        self.fourcc = str(fourcc)
        self.ext = str(ext) if str(ext).startswith(".") else "." + str(ext)
        self.segment_seconds = int(segment_seconds)
        self.save_height = int(max(0, int(save_height or 0)))

        os.makedirs(self.out_dir, exist_ok=True)
        self.run_dir = os.path.join(self.out_dir, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)

        self._vw: Optional[cv2.VideoWriter] = None
        self._size_wh: Optional[Tuple[int, int]] = None
        self._seg_start_mono = 0.0
        self._seg_index = 0
        self._current_path: Optional[str] = None

        self.segments_started = 0
        self.segments_closed = 0

    @staticmethod
    def _make_even(x: int) -> int:
        xi = int(x)
        return xi if (xi % 2) == 0 else max(2, xi - 1)

    def _prepare_frame_for_save(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr
        if self.save_height <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return frame_bgr
        if h <= self.save_height:
            return frame_bgr
        new_h = self._make_even(self.save_height)
        new_w = int(round(w * (new_h / float(h))))
        new_w = self._make_even(max(2, new_w))
        try:
            return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame_bgr

    def _close_current(self):
        if self._vw is not None:
            try:
                self._vw.release()
            except Exception:
                pass
            self._vw = None
            self.segments_closed += 1
        self._current_path = None
        self._size_wh = None
        self._seg_start_mono = 0.0

    def _open_new(self, frame_wh: Tuple[int, int]):
        self._close_current()
        self._size_wh = frame_wh
        self._seg_start_mono = time.monotonic()
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.run_dir, f"{self.basename}_{ts_tag}_p{self._seg_index:04d}{self.ext}")
        vw = _open_writer_with_fallback(path, fps=self.fps, size_wh=frame_wh, preferred_fourcc=self.fourcc)
        if vw is None:
            return
        self._vw = vw
        self._current_path = path
        self._seg_index += 1
        self.segments_started += 1

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        frame_bgr = self._prepare_frame_for_save(frame_bgr)
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return
        frame_wh = (int(w), int(h))

        now_mono = time.monotonic()
        if self._vw is None or self._size_wh != frame_wh:
            self._open_new(frame_wh)
            if self._vw is None:
                return

        if self.segment_seconds > 0 and (now_mono - float(self._seg_start_mono)) >= self.segment_seconds:
            self._open_new(frame_wh)
            if self._vw is None:
                return

        try:
            self._vw.write(frame_bgr)
        except Exception:
            self._close_current()

    def close(self):
        self._close_current()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    if not args.use_db:
        raise SystemExit("This script is DB-first. Use: --use-db --db-url ...")
    if not args.db_url:
        raise SystemExit("--use-db requires --db-url")

    # Map streams -> DB camera_ids
    if not getattr(args, "camera_ids", None):
        args.camera_ids = []
    if len(args.camera_ids) == 0:
        args.camera_ids = list(range(1, len(args.src) + 1))
    if len(args.camera_ids) != len(args.src):
        raise SystemExit("--camera-ids must have the same length as --src")

    args.save_video = not bool(getattr(args, "no_save_video", False))
    args.global_unique_names = (len(args.src) > 1) and (not bool(getattr(args, "no_global_unique_names", False)))

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        pass

    gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
    if gpu:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.half and not gpu:
        print("[WARN] --half requested but CUDA not available; disabling FP16.")
        args.half = False
    print(f"[INIT] device={args.device} cuda_available={torch.cuda.is_available()} half={args.half}")

    gallery_mgr = GalleryManager(args)

    global_owner: Optional[GlobalNameOwner] = None
    if bool(getattr(args, "global_unique_names", False)):
        global_owner = GlobalNameOwner(
            hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
            switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
        )
        print(f"[INIT] Global unique names: ON (hold={global_owner.hold_seconds}s, margin={global_owner.switch_margin})")
    else:
        print("[INIT] Global unique names: OFF")

    # YOLO
    yolo = None
    if YOLO is not None:
        try:
            weights = args.yolo_weights
            if not Path(weights).exists():
                print(f"[INIT] {weights} not found, falling back to yolov8n.pt")
                weights = "yolov8n.pt"
            yolo = YOLO(weights)
            if gpu:
                try:
                    yolo.to(args.device)
                except Exception as e:
                    print("[WARN] YOLO .to(device) failed:", e)
            print("[INIT] YOLO ready")
        except Exception as e:
            print("[ERROR] YOLO load failed:", e)
            yolo = None
    else:
        print("[WARN] ultralytics not installed; detection disabled")

    # TorchReID extractor
    reid_extractor = None
    if TorchreidExtractor is not None:
        try:
            dev = args.device if gpu else "cpu"
            if args.reid_weights and Path(args.reid_weights).exists():
                reid_extractor = TorchreidExtractor(model_name=args.reid_model, model_path=args.reid_weights, device=dev)
            else:
                reid_extractor = TorchreidExtractor(model_name=args.reid_model, device=dev)
            print(f"[INIT] TorchReID ready (model={args.reid_model}, device={dev})")
        except Exception as e:
            print("[WARN] TorchReID init failed; body support disabled:", e)
            reid_extractor = None
    else:
        print("[WARN] torchreid not installed; body support disabled")

    # Face
    face_app = init_face_engine(
        args.use_face,
        args.device,
        args.face_model,
        int(args.face_det_size[0]),
        int(args.face_det_size[1]),
        face_provider=getattr(args, "face_provider", "auto"),
        ort_log=getattr(args, "ort_log", False),
    )

    # DeepSORT usage
    enable_deepsort = False
    if args.no_deepsort:
        enable_deepsort = False
    else:
        enable_deepsort = DeepSort is not None
    if args.use_deepsort and DeepSort is None:
        print("[WARN] --use-deepsort requested but deep-sort-realtime not installed.")

    # Summary report (CSV)
    report: Optional[SummaryReport] = None
    csv_stop_evt: Optional[threading.Event] = None
    csv_thread: Optional[threading.Thread] = None

    if args.save_csv:
        out_dir = "csv_output"
        os.makedirs(out_dir, exist_ok=True)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = SummaryReport(
            num_cams=len(args.src),
            gap_seconds=float(args.report_gap_seconds),
            time_format=args.report_time_format,
        )

        # Respect user-provided --csv path.
        # If user left it at the default, write into csv_output/ with a timestamp.
        csv_arg = str(getattr(args, "csv", "") or "").strip()
        if (not csv_arg) or (csv_arg == "detections_summary.csv"):
            args.csv = os.path.join(out_dir, f"output_csv_new{ts_tag}.csv")
        else:
            args.csv = csv_arg
            try:
                os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            except Exception:
                pass

        print(f"[CSV] Live summary will be written to: {args.csv}")
        interval = float(args.csv_live_interval)
        if interval > 0:
            csv_stop_evt = threading.Event()
            csv_thread = threading.Thread(
                target=live_csv_writer_loop,
                args=(report, args.csv, interval, csv_stop_evt),
                daemon=True,
            )
            csv_thread.start()

    # ✅ Embedding updater
    embed_updater: Optional[EmbeddingDBUpdater] = None
    if bool(getattr(args, "update_db_embeddings", False)):
        if not bool(getattr(args, "use_face", False)):
            print("[WARN] --update-db-embeddings requires --use-face. Disabling embedding updates.")
        else:
            os.makedirs("csv_output", exist_ok=True)
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = str(getattr(args, "embeddings_log_csv", "") or "").strip()
            if not log_path:
                log_path = os.path.join("csv_output", f"embeddings_updates_{ts_tag}.csv")
            samples_log_path = str(getattr(args, "embeddings_samples_log_csv", "") or "").strip()
            if not samples_log_path:
                samples_log_path = os.path.join("csv_output", f"embeddings_samples_{ts_tag}.csv")

            embed_updater = EmbeddingDBUpdater(
                db_url=args.db_url,
                slot_mb=float(getattr(args, "embeddings_slot_mb", 0.5)),
                flush_seconds=float(getattr(args, "embeddings_flush_seconds", 10.0)),
                min_sample_seconds=float(getattr(args, "embeddings_min_sample_seconds", 0.5)),
                min_face_sim=float(getattr(args, "update_face_sim_thresh", 0.75)),
                min_face_det_score=float(getattr(args, "embed_min_face_det_score", 0.75)),
                log_csv_path=log_path,
                update_body=not bool(getattr(args, "no_update_body_bank", False)),
                update_face=not bool(getattr(args, "no_update_face_bank", False)),
                reset_if_gap_days_ge=int(getattr(args, "embeddings_reset_if_gap_days", 2)),
                reset_on_start=False,
                samples_log_csv_path=samples_log_path,
            )
            print(
                f"[INIT] DB embedding updater: ON "
                f"(slot_mb={float(getattr(args,'embeddings_slot_mb',0.5)):.3f} => "
                f"{2*float(getattr(args,'embeddings_slot_mb',0.5)):.3f}MB total) "
                f"(updates_log={log_path}, samples_log={samples_log_path})"
            )

    # Open sources
    streams = []
    for i, raw_src in enumerate(args.src):
        src = raw_src.strip() if isinstance(raw_src, str) else raw_src
        camera_db_id = int(args.camera_ids[i])

        vs = AdaptiveQueueStream(
            src,
            queue_size=args.queue_size,
            rtsp_transport=args.rtsp_transport,
            use_opencv=True,
            freeze_seconds=float(args.stream_freeze_seconds),
            open_timeout_ms=int(args.stream_open_timeout_ms),
            read_timeout_ms=int(args.stream_read_timeout_ms),
            reconnect_base_delay=float(args.stream_reconnect_base_seconds),
            reconnect_max_delay=float(args.stream_reconnect_max_seconds),
            reconnect_jitter=float(args.stream_reconnect_jitter),
            reconnect_log_interval=float(args.stream_reconnect_log_interval),
        )

        deep_tracker = None
        if enable_deepsort:
            try:
                deep_tracker = DeepSort(
                    max_age=int(args.max_age),
                    n_init=int(args.n_init),
                    nn_budget=int(args.nn_budget),
                    max_cosine_distance=float(args.tracker_max_cosine),
                    nms_max_overlap=float(args.tracker_nms_overlap),
                    embedder="torchreid",
                    embedder_gpu=gpu,
                    half=(gpu and args.half),
                    bgr=True,
                )
            except Exception as e:
                print(f"[WARN] DeepSORT init failed for SRC {i}, fallback to IoU tracker:", e)
                deep_tracker = None

        iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=0.3)

        streams.append({"sid": i, "camera_db_id": camera_db_id, "src": raw_src, "vs": vs, "deep": deep_tracker, "iou": iou_tracker})

    print(f"[INIT] sources requested: {len(args.src)}")
    for s in streams:
        print(f"[SRC {s['sid']}] cam_id={s['camera_db_id']} open={s['vs'].is_opened()} :: {s['src']}")

    if not any(s["vs"].is_opened() for s in streams):
        print("[ERROR] No sources opened. Check your --src URLs/paths and codecs.")
        return

    render_map: dict[int, RenderedFrame] = {s["sid"]: RenderedFrame() for s in streams}
    last_good: dict[int, np.ndarray] = {}

    # Start workers
    for s in streams:
        sid = int(s["sid"])
        t = threading.Thread(
            target=processor_thread,
            args=(
                sid,
                int(s["camera_db_id"]),
                s["vs"],
                render_map[sid],
                yolo,
                args,
                s["deep"],
                s["iou"],
                gallery_mgr,
                reid_extractor,
                face_app,
                global_owner,
                report,
                embed_updater,
                False,
            ),
            daemon=True,
        )
        t.start()

    print("[Main] Running. Press 'q' to quit (when --show).")

    win_name = "YOLO + DeepSORT + member_embeddings + 2-slot rolling banks"
    screen_w, screen_h = (0, 0)
    if args.show or args.save_video:
        screen_w, screen_h = _get_screen_resolution(default=(1920, 1080))

    if args.show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(win_name, int(screen_w), int(screen_h))
            cv2.moveWindow(win_name, 0, 0)
        except Exception:
            pass
        if bool(getattr(args, "fullscreen", False)):
            try:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass

    seg_writer: Optional[SegmentedVideoWriter] = None
    if args.save_video:
        out_dir = str(getattr(args, "video_dir", "saved_videos") or "saved_videos")
        os.makedirs(out_dir, exist_ok=True)

        seg_writer = SegmentedVideoWriter(
            out_dir=out_dir,
            basename=str(getattr(args, "video_prefix", "saved_video") or "saved_video"),
            ext=_norm_ext(getattr(args, "video_ext", ".mp4")),
            fps=float(getattr(args, "video_fps", 20.0) or 20.0),
            fourcc=str(getattr(args, "video_fourcc", "mp4v") or "mp4v"),
            segment_seconds=int(float(getattr(args, "video_segment_seconds", 3600.0) or 0.0)),
            save_height=int(getattr(args, "video_save_height", 480) or 0),
        )

        print(f"[INIT] Saving annotated videos to folder: {seg_writer.run_dir}")
        if seg_writer.segment_seconds > 0:
            print(f"[INIT] Video segmentation: ON ({seg_writer.segment_seconds:.0f}s per file)")
        else:
            print("[INIT] Video segmentation: OFF (single file)")

    disp_last = time.time()
    disp_fps_ema = 0.0
    disp_alpha = 0.10

    try:
        while True:
            display_frames: List[np.ndarray] = []
            any_open = False

            for s in streams:
                sid = int(s["sid"])
                vs = s["vs"]
                if not vs.is_opened():
                    display_frames.append(last_good.get(sid, np.zeros((720, 1280, 3), dtype=np.uint8)))
                    continue
                any_open = True
                frm, _ts, _meta = render_map[sid].get()
                if frm is not None:
                    last_good[sid] = frm
                    display_frames.append(frm)
                else:
                    display_frames.append(last_good.get(sid, np.zeros((720, 1280, 3), dtype=np.uint8)))

            if (args.show or args.save_video) and display_frames:
                if screen_w <= 0 or screen_h <= 0:
                    screen_w, screen_h = _get_screen_resolution(default=(1920, 1080))

                vis = make_grid_view(
                    display_frames,
                    screen_w=int(screen_w),
                    screen_h=int(screen_h),
                    mode=str(getattr(args, "grid_mode", "cover")),
                    grid_rows=int(getattr(args, "grid_rows", 0) or 0),
                    grid_cols=int(getattr(args, "grid_cols", 0) or 0),
                )

                now = time.time()
                dt = max(1e-6, now - disp_last)
                disp_fps = 1.0 / dt
                disp_fps_ema = (1 - disp_alpha) * disp_fps_ema + disp_alpha * disp_fps
                disp_last = now

                cv2.putText(
                    vis,
                    f"DISPLAY FPS {disp_fps_ema:.1f} | cams {len(display_frames)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                if args.save_video and seg_writer is not None:
                    seg_writer.write(vis)

                if args.show:
                    cv2.imshow(win_name, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if not any_open:
                break

            time.sleep(0.001)

    finally:
        for s in streams:
            try:
                s["vs"].release()
            except Exception:
                pass

        if seg_writer is not None:
            try:
                seg_writer.close()
            except Exception:
                pass
            print(f"[DONE] Saved videos folder: {seg_writer.run_dir} (segments_closed={seg_writer.segments_closed})")

        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Stop live writer threads and write final CSV
        try:
            if csv_stop_evt is not None:
                csv_stop_evt.set()
            if csv_thread is not None:
                csv_thread.join(timeout=2.0)
        except Exception:
            pass

        if report is not None:
            report.stop()
            try:
                if args.save_csv:
                    report.write_csv(args.csv)
                    print(f"[CSV] Final summary written to {args.csv}")
            except Exception as e:
                print("[CSV] Final write failed:", e)

        # Close updater (final flush)
        if embed_updater is not None:
            try:
                embed_updater.close()
                print("[DONE] Embedding updater closed (final flush done).")
            except Exception as e:
                print("[WARN] Embedding updater close failed:", e)

        print("Done.")


# ============================================================
# v1-style service wrappers (keep v2 logic unchanged)
# ============================================================

import shlex


def parse_pipeline_args(pipeline_args: str | None) -> argparse.Namespace:
    """
    Parse pipeline args from a single string (PIPELINE_ARGS style),
    without modifying the original v2 argparse logic.
    """
    s = str(pipeline_args or "").strip()
    argv = shlex.split(s) if s else []
    return parse_args(argv)



def processor_thread_with_stop(
    sid: int,
    camera_db_id: int,
    vs,
    render_store: RenderedFrame,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    gallery_mgr: GalleryManager,
    reid_extractor,
    face_app,
    global_owner: Optional[GlobalNameOwner],
    report: Optional[SummaryReport],
    embed_updater: Optional[EmbeddingDBUpdater],
    stop_evt: threading.Event,
    debug: bool = False,
):
    """
    Service-safe version of processor_thread() that exits when stop_evt is set.
    Tracking logic is identical to v2 processor_thread.
    """
    frame_idx = 0
    identity_state: dict[int, dict] = {}

    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10

    device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

    while not stop_evt.is_set():
        ok, frame, ts_cap = vs.read()
        if not ok or frame is None:
            stop_evt.wait(0.005)
            continue

        if int(args.max_queue_age_ms) > 0:
            now = time.time()
            age_ms = (now - float(ts_cap)) * 1000.0
            dropped_here = 0
            while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
                try:
                    vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
                except Exception:
                    pass
                ok2, frame2, ts2 = vs.read()
                if not ok2 or frame2 is None:
                    break
                frame, ts_cap = frame2, ts2
                age_ms = (time.time() - float(ts_cap)) * 1000.0
                dropped_here += 1

        try:
            gallery_mgr.maybe_reload(args)
            people_by_cam, face_gallery, name_to_mid = gallery_mgr.snapshot()
            people = people_by_cam.get(int(camera_db_id), [])

            out, meta = process_one_frame(
                frame_idx,
                frame,
                sid,
                int(camera_db_id),
                yolo,
                args,
                deep_tracker,
                iou_tracker,
                people,
                reid_extractor,
                face_app,
                face_gallery,
                name_to_mid,
                identity_state,
                device_is_cuda=device_is_cuda,
                global_owner=global_owner,
                ts_cap=float(ts_cap),
                embed_updater=embed_updater,
            )

            if report is not None:
                ts_use = float(ts_cap) if ts_cap else time.time()
                if bool(getattr(args, "report_use_drawn_only", True)):
                    events = meta.get("events", []) or []
                    conf_map: Dict[str, float] = {}
                    for ev in events:
                        try:
                            nm = str(ev[5] or "")
                            sim = float(ev[6] or 0.0)
                        except Exception:
                            continue
                        if not nm:
                            continue
                        prev = float(conf_map.get(nm, 0.0))
                        if sim > prev:
                            conf_map[nm] = sim
                    names = sorted(conf_map.keys())
                else:
                    names = meta.get("present_names", []) or []
                    conf_map = meta.get("present_conf", {}) or {}
                    if not isinstance(conf_map, dict):
                        conf_map = {}

                report.update(
                    cam_id=int(sid),
                    present_names=list(names),
                    ts=ts_use,
                    name_to_conf=dict(conf_map),
                )

            now = time.time()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
            last_t = now

            if args.overlay_fps:
                qsz = int(vs.qsize()) if hasattr(vs, "qsize") else 0
                dropped_cap = int(getattr(vs, "dropped", 0))
                dropped_read = int(getattr(vs, "read_dropped", 0))
                lag_ms = (now - float(ts_cap)) * 1000.0 if ts_cap else 0.0
                lines = [
                    f"SRC {sid} (cam_id={camera_db_id}) | FPS {fps_ema:.1f} | lag {lag_ms:.0f}ms",
                    f"q {qsz} | drop(cap) {dropped_cap} | drop(stale) {dropped_read}",
                    f"tracks {meta.get('tracks', 0)} | shown {meta.get('shown', 0)} | faces {meta.get('faces_recognized', 0)}",
                ]
                y = 22
                for ln in lines:
                    cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 22

            render_store.set(out, meta={"fps": float(fps_ema), "sid": int(sid), "camera_db_id": int(camera_db_id)})
            frame_idx += 1

        except Exception as e:
            if debug:
                print(f"[PROC {sid}] error:", e)
            stop_evt.wait(0.001)


class TrackingRunner:
    """
    FastAPI/service runner that starts the SAME v2 pipeline, but without any GUI loop.

    - Buffers are addressed by DB camera_id (the values you pass via --camera-ids).
    - MJPEG endpoints can call get_camera_buffer(camera_id).
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []

        self._streams: List[Dict[str, Any]] = []
        self._render_by_cam: Dict[int, RenderedFrame] = {}

        self._gallery_mgr: Optional[GalleryManager] = None
        self._global_owner: Optional[GlobalNameOwner] = None
        self._report: Optional[SummaryReport] = None
        self._csv_stop_evt: Optional[threading.Event] = None
        self._csv_thread: Optional[threading.Thread] = None
        self._embed_updater: Optional[EmbeddingDBUpdater] = None

        self._yolo = None
        self._reid_extractor = None
        self._face_app = None

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        args = self.args

        if not bool(getattr(args, "use_db", False)):
            raise RuntimeError("TrackingRunner requires --use-db and --db-url (same as v2).")
        if not str(getattr(args, "db_url", "") or "").strip():
            raise RuntimeError("TrackingRunner requires --db-url.")

        # Map streams -> DB camera_ids (same as v2 main)
        if not getattr(args, "camera_ids", None):
            args.camera_ids = []
        if len(args.camera_ids) == 0:
            args.camera_ids = list(range(1, len(args.src) + 1))
        if len(args.camera_ids) != len(args.src):
            raise RuntimeError("--camera-ids must have the same length as --src")

        args.save_video = not bool(getattr(args, "no_save_video", False))
        args.global_unique_names = (len(args.src) > 1) and (not bool(getattr(args, "no_global_unique_names", False)))

        if getattr(args, "cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

        gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
        if gpu:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if bool(getattr(args, "half", False)) and (not gpu):
            args.half = False

        # DB gallery manager
        self._gallery_mgr = GalleryManager(args)

        # Global name owner
        self._global_owner = None
        if bool(getattr(args, "global_unique_names", False)):
            self._global_owner = GlobalNameOwner(
                hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
                switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
            )

        # YOLO
        yolo = None
        if YOLO is not None:
            try:
                weights = args.yolo_weights
                if weights and (not Path(weights).exists()):
                    weights = "yolov8n.pt"
                yolo = YOLO(weights)
                if gpu:
                    try:
                        yolo.to(args.device)
                    except Exception:
                        pass
            except Exception:
                yolo = None
        self._yolo = yolo

        # TorchReID extractor
        reid_extractor = None
        if TorchreidExtractor is not None:
            try:
                dev = args.device if gpu else "cpu"
                if getattr(args, "reid_weights", "") and Path(args.reid_weights).exists():
                    reid_extractor = TorchreidExtractor(model_name=args.reid_model, model_path=args.reid_weights, device=dev)
                else:
                    reid_extractor = TorchreidExtractor(model_name=args.reid_model, device=dev)
            except Exception:
                reid_extractor = None
        self._reid_extractor = reid_extractor

        # Face
        self._face_app = init_face_engine(
            bool(getattr(args, "use_face", False)),
            args.device,
            args.face_model,
            int(args.face_det_size[0]),
            int(args.face_det_size[1]),
            face_provider=getattr(args, "face_provider", "auto"),
            ort_log=getattr(args, "ort_log", False),
        )

        # Summary report (CSV live writer) - same as v2 main
        self._report = None
        self._csv_stop_evt = None
        self._csv_thread = None
        if bool(getattr(args, "save_csv", False)):
            out_dir = "csv_output"
            os.makedirs(out_dir, exist_ok=True)
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

            self._report = SummaryReport(
                num_cams=len(args.src),
                gap_seconds=float(getattr(args, "report_gap_seconds", 2.0)),
                time_format=str(getattr(args, "report_time_format", "%H:%M:%S")),
            )

            csv_arg = str(getattr(args, "csv", "") or "").strip()
            if (not csv_arg) or (csv_arg == "detections_summary.csv"):
                args.csv = os.path.join(out_dir, f"output_csv_new{ts_tag}.csv")
            else:
                args.csv = csv_arg
                try:
                    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
                except Exception:
                    pass

            interval = float(getattr(args, "csv_live_interval", 1.0) or 0.0)
            if interval > 0:
                self._csv_stop_evt = threading.Event()
                self._csv_thread = threading.Thread(
                    target=live_csv_writer_loop,
                    args=(self._report, args.csv, interval, self._csv_stop_evt),
                    daemon=True,
                )
                self._csv_thread.start()

        # Embedding updater - same as v2 main
        self._embed_updater = None
        if bool(getattr(args, "update_db_embeddings", False)):
            if not bool(getattr(args, "use_face", False)):
                self._embed_updater = None
            else:
                os.makedirs("csv_output", exist_ok=True)
                ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = str(getattr(args, "embeddings_log_csv", "") or "").strip() or os.path.join("csv_output", f"embeddings_updates_{ts_tag}.csv")
                samples_log_path = str(getattr(args, "embeddings_samples_log_csv", "") or "").strip() or os.path.join("csv_output", f"embeddings_samples_{ts_tag}.csv")

                self._embed_updater = EmbeddingDBUpdater(
                    db_url=args.db_url,
                    slot_mb=float(getattr(args, "embeddings_slot_mb", 0.5)),
                    flush_seconds=float(getattr(args, "embeddings_flush_seconds", 10.0)),
                    min_sample_seconds=float(getattr(args, "embeddings_min_sample_seconds", 0.5)),
                    min_face_sim=float(getattr(args, "update_face_sim_thresh", 0.75)),
                    min_face_det_score=float(getattr(args, "embed_min_face_det_score", 0.75)),
                    log_csv_path=log_path,
                    update_body=not bool(getattr(args, "no_update_body_bank", False)),
                    update_face=not bool(getattr(args, "no_update_face_bank", False)),
                    reset_if_gap_days_ge=int(getattr(args, "embeddings_reset_if_gap_days", 2)),
                    reset_on_start=False,
                    samples_log_csv_path=samples_log_path,
                )

        # DeepSORT enable decision (same as v2 main)
        enable_deepsort = False
        if getattr(args, "no_deepsort", False):
            enable_deepsort = False
        else:
            enable_deepsort = DeepSort is not None

        if bool(getattr(args, "use_deepsort", False)) and DeepSort is None:
            enable_deepsort = False

        # Open sources + start worker threads
        self._streams = []
        self._render_by_cam = {}

        for sid, raw_src in enumerate(args.src):
            src = raw_src.strip() if isinstance(raw_src, str) else raw_src
            camera_db_id = int(args.camera_ids[sid])

            vs = AdaptiveQueueStream(
                src,
                queue_size=args.queue_size,
                rtsp_transport=args.rtsp_transport,
                use_opencv=True,
                freeze_seconds=float(args.stream_freeze_seconds),
                open_timeout_ms=int(args.stream_open_timeout_ms),
                read_timeout_ms=int(args.stream_read_timeout_ms),
                reconnect_base_delay=float(args.stream_reconnect_base_seconds),
                reconnect_max_delay=float(args.stream_reconnect_max_seconds),
                reconnect_jitter=float(args.stream_reconnect_jitter),
                reconnect_log_interval=float(args.stream_reconnect_log_interval),
            )

            deep_tracker = None
            if enable_deepsort:
                try:
                    deep_tracker = DeepSort(
                        max_age=int(args.max_age),
                        n_init=int(args.n_init),
                        nn_budget=int(args.nn_budget),
                        max_cosine_distance=float(args.tracker_max_cosine),
                        nms_max_overlap=float(args.tracker_nms_overlap),
                        embedder="torchreid",
                        embedder_gpu=gpu,
                        half=(gpu and bool(args.half)),
                        bgr=True,
                    )
                except Exception:
                    deep_tracker = None

            iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=0.3)

            buf = RenderedFrame()
            self._render_by_cam[int(camera_db_id)] = buf

            self._streams.append({"sid": sid, "camera_db_id": camera_db_id, "src": raw_src, "vs": vs, "deep": deep_tracker, "iou": iou_tracker, "buf": buf})

        if not any(s["vs"].is_opened() for s in self._streams):
            raise RuntimeError("No sources opened. Check --src URLs and codecs.")

        # threads
        for s in self._streams:
            t = threading.Thread(
                target=processor_thread_with_stop,
                args=(
                    int(s["sid"]),
                    int(s["camera_db_id"]),
                    s["vs"],
                    s["buf"],
                    self._yolo,
                    args,
                    s["deep"],
                    s["iou"],
                    self._gallery_mgr,
                    self._reid_extractor,
                    self._face_app,
                    self._global_owner,
                    self._report,
                    self._embed_updater,
                    self._stop_evt,
                    False,
                ),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()

        # stop streams
        for s in self._streams:
            try:
                s["vs"].release()
            except Exception:
                pass

        # stop csv writer
        try:
            if self._csv_stop_evt is not None:
                self._csv_stop_evt.set()
            if self._csv_thread is not None:
                self._csv_thread.join(timeout=2.0)
        except Exception:
            pass

        # final report write (optional)
        if self._report is not None:
            try:
                self._report.stop()
                if bool(getattr(self.args, "save_csv", False)) and str(getattr(self.args, "csv", "") or "").strip():
                    self._report.write_csv(self.args.csv)
            except Exception:
                pass

        # close embedding updater
        if self._embed_updater is not None:
            try:
                self._embed_updater.close()
            except Exception:
                pass

        # join worker threads briefly
        for t in self._threads:
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        self._started = False

    def get_camera_buffer(self, cam_id: int) -> Optional[RenderedFrame]:
        return self._render_by_cam.get(int(cam_id))

    def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Compatibility helper for the existing FastAPI router.
        In this v2 pipeline, cameras come from (--src, --camera-ids).
        """
        out: List[Dict[str, Any]] = []
        for s in self._streams:
            cam_id = int(s.get("camera_db_id", -1))
            out.append({
                "id": cam_id,
                "camera_id": cam_id,
                "src": str(s.get("src", "")),
                "running": bool(getattr(s.get("vs", None), "is_opened", lambda: False)()),
            })
        out.sort(key=lambda x: int(x.get("id", 0)))
        return out

    def status(self) -> Dict[str, Any]:
        cams = sorted(list(self._render_by_cam.keys()))
        return {
            "running": bool(self._started),
            "camera_ids": cams,
            "num_cameras": len(cams),
            "save_csv": bool(getattr(self.args, "save_csv", False)),
            "csv_path": str(getattr(self.args, "csv", "") or ""),
            "update_db_embeddings": bool(getattr(self.args, "update_db_embeddings", False)),
        }

    def write_report_snapshot(self, path: str | None = None) -> str:
        if self._report is None:
            return ""
        out_path = str(path or getattr(self.args, "csv", "") or "").strip()
        if not out_path:
            out_path = "detections_summary_snapshot.csv"
        try:
            self._report.write_csv_live(out_path)
        except Exception:
            pass
        return out_path

if __name__ == "__main__":
    try:
        print("[BOOT] updated_tracking_member_embeddings_rollingslots.py starting...")
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        print("[FATAL] Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)
