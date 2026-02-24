# # # app/services/tracking/pipeline.py
# # from __future__ import annotations

# # import argparse
# # import csv
# # import ctypes
# # import gzip
# # import math
# # import os
# # import sys
# # import time
# # import threading
# # import queue
# # import subprocess
# # from dataclasses import dataclass
# # from datetime import datetime
# # from pathlib import Path
# # from collections import defaultdict
# # from typing import Any, Dict, List, Optional, Tuple
# # from urllib.parse import urlsplit, urlunsplit, quote, unquote
# # from io import BytesIO

# # import cv2
# # import numpy as np
# # import torch

# # # ✅ NEW DB repo (members + member_embeddings)
# # from app.repositories.gallery_repo import (
# #     GalleryRepository,
# #     PersonEntry,
# #     FaceGallery,
# #     EXPECTED_DIM,
# # )

# # from app.utils.embeddings import l2_normalize, l2_normalize_rows

# # # --- YOLO (Ultralytics) ---
# # try:
# #     from ultralytics import YOLO
# # except Exception:
# #     YOLO = None

# # # --- DeepSORT (deep-sort-realtime) ---
# # try:
# #     from deep_sort_realtime.deepsort_tracker import DeepSort
# # except Exception:
# #     DeepSort = None

# # # --- TorchReID gallery extractor (body support only) ---
# # try:
# #     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# # except Exception:
# #     TorchreidExtractor = None

# # # --- InsightFace ---
# # try:
# #     from insightface.app import FaceAnalysis
# #     INSIGHT_OK = True
# # except Exception:
# #     FaceAnalysis = None
# #     INSIGHT_OK = False

# # # --- ONNX Runtime ---
# # try:
# #     import onnxruntime as ort
# # except Exception:
# #     ort = None


# # # ----------------------------
# # # Globals (guards)
# # # ----------------------------
# # _yolo_lock = threading.Lock()
# # _reid_lock = threading.Lock()
# # _face_lock = threading.Lock()


# # # ----------------------------
# # # Utils
# # # ----------------------------
# # def safe_iter_faces(obj):
# #     if obj is None:
# #         return []
# #     try:
# #         return list(obj)
# #     except TypeError:
# #         return [obj]


# # def extract_face_embedding(face):
# #     emb = getattr(face, "normed_embedding", None)
# #     if emb is None:
# #         emb = getattr(face, "embedding", None)
# #     return emb


# # def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
# #     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# # def _sanitize_rtsp_url(url: str) -> str:
# #     parts = urlsplit(url)
# #     if parts.username or parts.password:
# #         user = quote(unquote(parts.username or ""), safe="")
# #         pwd = quote(unquote(parts.password or ""), safe="")
# #         host = parts.hostname or ""
# #         netloc = f"{user}:{pwd}@{host}"
# #         if parts.port:
# #             netloc += f":{parts.port}"
# #         return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
# #     return url


# # # ----------------------------
# # # ORT CUDA provider probe
# # # ----------------------------
# # def _cuda_ep_loadable() -> bool:
# #     if ort is None:
# #         return False
# #     try:
# #         if sys.platform.startswith("darwin"):
# #             return False
# #         capi_dir = Path(ort.__file__).parent / "capi"
# #         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
# #         lib_path = capi_dir / name
# #         if not lib_path.exists():
# #             return False
# #         ctypes.CDLL(str(lib_path))
# #         return True
# #     except Exception:
# #         return False


# # # ----------------------------
# # # DB Gallery manager
# # # ----------------------------
# # class GalleryManager:
# #     """Thread-safe DB gallery with optional periodic refresh."""

# #     def __init__(self, args):
# #         if not getattr(args, "db_url", ""):
# #             raise RuntimeError("db_url is required")
# #         self._repo = GalleryRepository(args.db_url)

# #         self._lock = threading.Lock()
# #         self._reload_lock = threading.Lock()
# #         self.people: list[PersonEntry] = []
# #         self.face_gallery: FaceGallery = FaceGallery([], np.zeros((0, EXPECTED_DIM), dtype=np.float32))
# #         self.last_load_ts: float = 0.0
# #         self.load(args)

# #     def load(self, args) -> None:
# #         active_only = not bool(getattr(args, "db_include_inactive", False))
# #         max_bank = int(getattr(args, "db_max_bank", 0) or 0)

# #         # ✅ this loads from members + member_embeddings (repo handles schema)
# #         people, fg = self._repo.load(active_only=active_only, max_bank_per_member=max_bank)

# #         with self._lock:
# #             self.people = people
# #             self.face_gallery = fg
# #             self.last_load_ts = time.time()

# #         body_count = sum(1 for p in people if p.body_bank is not None and p.body_bank.size > 0)
# #         print(f"[DB] Loaded identities: {len(people)} (body={body_count}, face={len(fg.names)})")

# #     def maybe_reload(self, args) -> None:
# #         period = float(getattr(args, "db_refresh_seconds", 0.0) or 0.0)
# #         if period <= 0:
# #             return
# #         now = time.time()
# #         if (now - self.last_load_ts) < period:
# #             return
# #         if not self._reload_lock.acquire(blocking=False):
# #             return
# #         try:
# #             if (time.time() - self.last_load_ts) < period:
# #                 return
# #             try:
# #                 self.load(args)
# #                 print("[DB] Gallery reloaded")
# #             except Exception as e:
# #                 print("[DB] reload failed:", e)
# #         finally:
# #             self._reload_lock.release()

# #     def snapshot(self) -> tuple[list[PersonEntry], FaceGallery]:
# #         with self._lock:
# #             return self.people, self.face_gallery


# # # ----------------------------
# # # Tracking helpers (IoU)
# # # ----------------------------
# # def iou_xyxy(a, b) -> float:
# #     ax1, ay1, ax2, ay2 = a
# #     bx1, by1, bx2, by2 = b
# #     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
# #     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
# #     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
# #     inter = iw * ih
# #     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
# #     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
# #     denom = a_area + b_area - inter
# #     return float(inter / denom) if denom > 0 else 0.0


# # def ioa_xyxy(inner, outer) -> float:
# #     """Intersection over AREA(inner). Good for linking small face box to big person box."""
# #     ix1, iy1, ix2, iy2 = inner
# #     ox1, oy1, ox2, oy2 = outer
# #     inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
# #     inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
# #     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
# #     inter = iw * ih
# #     inner_area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
# #     return float(inter / inner_area) if inner_area > 0 else 0.0


# # def _point_in_xyxy(px: float, py: float, box) -> bool:
# #     x1, y1, x2, y2 = box
# #     return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


# # class IOUTrack:
# #     def __init__(self, tlwh, tid):
# #         self.tlwh = np.array(tlwh, dtype=np.float32)
# #         self.tid = int(tid)
# #         self.miss = 0


# # class IOUTracker:
# #     """Minimal IoU tracker fallback (stable IDs)."""

# #     def __init__(self, max_miss=5, iou_thresh=0.3):
# #         self.tracks: list[IOUTrack] = []
# #         self.next_id = 1
# #         self.max_miss = int(max_miss)
# #         self.iou_thresh = float(iou_thresh)

# #     def update(self, dets_tlwh_conf: np.ndarray):
# #         dets = np.asarray(dets_tlwh_conf, dtype=np.float32)
# #         if dets.ndim != 2 or dets.shape[1] < 4:
# #             dets = dets.reshape((0, 5)).astype(np.float32)

# #         assigned = set()
# #         for tr in self.tracks:
# #             tr.miss += 1
# #             t_x, t_y, t_w, t_h = tr.tlwh
# #             t_xyxy = np.array([t_x, t_y, t_x + t_w, t_y + t_h], dtype=np.float32)
# #             best_j, best_iou = -1, 0.0
# #             for j, d in enumerate(dets):
# #                 if j in assigned:
# #                     continue
# #                 x, y, w, h = d[:4]
# #                 d_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
# #                 s = iou_xyxy(t_xyxy, d_xyxy)
# #                 if s > best_iou:
# #                     best_iou, best_j = s, j
# #             if best_j >= 0 and best_iou >= self.iou_thresh:
# #                 tr.tlwh = dets[best_j][:4]
# #                 tr.miss = 0
# #                 assigned.add(best_j)

# #         for j, d in enumerate(dets):
# #             if j in assigned:
# #                 continue
# #             self.tracks.append(IOUTrack(d[:4], self.next_id))
# #             self.next_id += 1

# #         self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

# #         outs = []
# #         for t in self.tracks:
# #             x, y, w, h = t.tlwh

# #             class Dummy:
# #                 pass

# #             o = Dummy()
# #             o.track_id = t.tid
# #             o.is_confirmed = lambda: True
# #             o.to_tlbr = lambda: (x, y, x + w, y + h)
# #             o.det_conf = None
# #             o.last_detection = None
# #             o.time_since_update = 0
# #             outs.append(o)

# #         return outs


# # # ----------------------------
# # # Identity smoothing (anti-flicker)
# # # ----------------------------
# # def update_track_identity(
# #     state: dict,
# #     tid: int,
# #     candidates: list[tuple[str, float, str]],
# #     decay: float,
# #     min_score: float,
# #     margin: float,
# #     ttl_reset: int,
# #     w_face: float,
# #     w_body: float,
# # ) -> tuple[str, float, dict]:
# #     """
# #     Exponentially decayed per-name score accumulator.

# #     NOTE: Body votes are only accepted when they agree with the face label (see caller).
# #     """
# #     entry = state.setdefault(
# #         tid,
# #         {
# #             "scores": defaultdict(float),
# #             "last": "",
# #             "ttl": 0,
# #             "face_vis_ttl": 0,
# #             "last_face_label": "",
# #             "last_face_sim": 0.0,
# #         },
# #     )
# #     scores = entry["scores"]

# #     for k in list(scores.keys()):
# #         scores[k] *= float(decay)
# #         if scores[k] < 1e-6:
# #             del scores[k]

# #     for label, sim, src in candidates:
# #         if not label:
# #             continue
# #         w = w_face if src == "face" else w_body
# #         scores[label] += max(0.0, float(sim)) * float(w)

# #     if scores:
# #         ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
# #         top_label, top_score = ranked[0]
# #         second_score = ranked[1][1] if len(ranked) > 1 else 0.0
# #     else:
# #         top_label, top_score, second_score = "", 0.0, 0.0

# #     if top_label and (top_score >= min_score) and (entry["last"] == top_label or (top_score - second_score) >= margin):
# #         entry["last"] = top_label
# #         entry["ttl"] = int(ttl_reset)
# #     else:
# #         if entry["ttl"] > 0:
# #             entry["ttl"] -= 1
# #         else:
# #             entry["last"] = ""

# #     return entry["last"], float(scores.get(entry["last"], 0.0)), entry


# # # ----------------------------
# # # Models init
# # # ----------------------------
# # def init_face_engine(
# #     use_face: bool,
# #     device: str,
# #     face_model: str,
# #     det_w: int,
# #     det_h: int,
# #     face_provider: str,
# #     ort_log: bool,
# # ):
# #     if not use_face:
# #         return None
# #     if not INSIGHT_OK:
# #         print("[WARN] insightface not installed; face recognition disabled.")
# #         return None
# #     try:
# #         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
# #         cuda_ok = _cuda_ep_loadable()

# #         if ort is not None and ort_log:
# #             try:
# #                 print(f"[INFO] ORT available providers: {ort.get_available_providers()}")
# #             except Exception:
# #                 pass

# #         providers = ["CPUExecutionProvider"]
# #         if face_provider == "cuda":
# #             if cuda_ok:
# #                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# #             else:
# #                 print("[INFO] Requested CUDA EP, but not loadable. Using CPU.")
# #         elif face_provider == "auto":
# #             if is_cuda and cuda_ok:
# #                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# #         app = FaceAnalysis(name=face_model, providers=providers)
# #         ctx_id = 0 if providers[0].startswith("CUDA") else -1
# #         try:
# #             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
# #         except TypeError:
# #             app.prepare(ctx_id=ctx_id)
# #         print(f"[INIT] InsightFace ready (model={face_model}, providers={providers}).")
# #         return app
# #     except Exception as e:
# #         print("[WARN] InsightFace init failed:", e)
# #         return None


# # def _yolo_forward_safe(yolo, frame, args):
# #     """
# #     Guarded YOLO call with a per-model lock and FP16->FP32 fallback.
# #     """
# #     with _yolo_lock, torch.inference_mode():
# #         try:
# #             return yolo(
# #                 frame,
# #                 conf=args.conf,
# #                 iou=args.iou,
# #                 verbose=False,
# #                 device=args.device,
# #                 half=args.half,
# #                 imgsz=int(args.yolo_imgsz) if int(args.yolo_imgsz) > 0 else None,
# #             )
# #         except TypeError:
# #             return yolo(
# #                 frame,
# #                 conf=args.conf,
# #                 iou=args.iou,
# #                 verbose=False,
# #                 device=args.device,
# #                 half=args.half,
# #             )
# #         except Exception as e:
# #             if args.half:
# #                 print("[YOLO] FP16 failed, retrying in FP32 once:", e)
# #                 args.half = False
# #                 return yolo(
# #                     frame,
# #                     conf=args.conf,
# #                     iou=args.iou,
# #                     verbose=False,
# #                     device=args.device,
# #                     half=False,
# #                 )
# #             raise


# # def extract_body_embeddings_batch(
# #     extractor,
# #     crops_bgr: List[np.ndarray],
# #     device_is_cuda: bool,
# #     use_half: bool,
# # ) -> Optional[np.ndarray]:
# #     """
# #     Batched TorchReID embedding extraction for multiple crops in one call.
# #     Returns: np.ndarray [N,512] normalized, or None.
# #     """
# #     if extractor is None or not crops_bgr:
# #         return None

# #     crops_rgb: List[np.ndarray] = []
# #     for c in crops_bgr:
# #         if c is None or c.size == 0:
# #             crops_rgb.append(np.zeros((1, 1, 3), dtype=np.uint8))
# #             continue
# #         crops_rgb.append(_to_rgb(c))

# #     with _reid_lock, torch.inference_mode():
# #         if device_is_cuda and use_half:
# #             try:
# #                 with torch.autocast(device_type="cuda", dtype=torch.float16):
# #                     feats = extractor(crops_rgb)
# #             except Exception:
# #                 feats = extractor(crops_rgb)
# #         else:
# #             feats = extractor(crops_rgb)

# #     try:
# #         if isinstance(feats, (list, tuple)):
# #             feats_arr = []
# #             for f in feats:
# #                 f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
# #                 feats_arr.append(np.asarray(f, dtype=np.float32).reshape(-1))
# #             mat = np.stack(feats_arr, axis=0)
# #         else:
# #             f = feats.detach().cpu().numpy() if hasattr(feats, "detach") else np.asarray(feats)
# #             mat = np.asarray(f, dtype=np.float32)
# #             if mat.ndim == 1:
# #                 mat = mat.reshape(1, -1)
# #         if mat.ndim != 2 or mat.shape[1] != EXPECTED_DIM:
# #             return None
# #         if not np.isfinite(mat).all():
# #             return None
# #         return l2_normalize_rows(mat)
# #     except Exception:
# #         return None


# # def best_body_label_from_emb(
# #     emb: np.ndarray | None,
# #     people: list[PersonEntry],
# #     topk: int = 3,
# # ) -> tuple[str | None, float, float]:
# #     """TopK-mean cosine match across each person's body_bank."""
# #     if emb is None:
# #         return None, 0.0, 0.0
# #     q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
# #     if q.size != EXPECTED_DIM or not np.isfinite(q).all():
# #         return None, 0.0, 0.0

# #     k_req = max(1, int(topk))
# #     scored: list[tuple[str, float]] = []

# #     for p in people:
# #         bank = p.body_bank
# #         if bank is None or bank.size == 0:
# #             continue
# #         try:
# #             sims = bank @ q
# #         except Exception:
# #             continue
# #         if sims.ndim != 1 or sims.size == 0:
# #             continue
# #         k = min(k_req, sims.size)
# #         if k <= 1:
# #             score = float(np.max(sims))
# #         else:
# #             top_vals = np.partition(sims, -k)[-k:]
# #             score = float(np.mean(top_vals))
# #         scored.append((p.name, score))

# #     if not scored:
# #         return None, 0.0, 0.0

# #     scored.sort(key=lambda x: x[1], reverse=True)
# #     best_label, best_score = scored[0]
# #     second_score = scored[1][1] if len(scored) > 1 else 0.0
# #     return best_label, float(best_score), float(second_score)


# # def best_face_label_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> tuple[str | None, float, float]:
# #     """
# #     Returns: (best_label, best_sim, second_sim)
# #     """
# #     if emb is None or face_gallery is None or face_gallery.is_empty():
# #         return None, 0.0, 0.0
# #     q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
# #     if q.size != EXPECTED_DIM or not np.isfinite(q).all():
# #         return None, 0.0, 0.0

# #     sims = face_gallery.mat @ q
# #     if sims.size == 0:
# #         return None, 0.0, 0.0

# #     if sims.size == 1:
# #         return face_gallery.names[0], float(sims[0]), 0.0

# #     idxs = np.argpartition(sims, -2)[-2:]
# #     i1, i2 = int(idxs[0]), int(idxs[1])
# #     if sims[i2] > sims[i1]:
# #         i1, i2 = i2, i1
# #     best_idx, second_idx = i1, i2
# #     return face_gallery.names[best_idx], float(sims[best_idx]), float(sims[second_idx])


# # # ----------------------------
# # # Threaded readers
# # # ----------------------------
# # class AdaptiveQueueStream:
# #     """
# #     Ordered reader with a bounded queue.
# #     - When queue is full, drops the oldest frame to make room (keeps latency bounded).
# #     """

# #     def __init__(self, src: str, queue_size: int, rtsp_transport: str, use_opencv: bool = True):
# #         self.src = src
# #         src_use = src
# #         if isinstance(src, str) and src.lower().startswith("rtsp"):
# #             sep = "&" if "?" in src_use else "?"
# #             src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"

# #         self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG) if use_opencv else cv2.VideoCapture(src_use)
# #         self.ok = self.cap.isOpened()
# #         if not self.ok:
# #             print(f"[WARN] cannot open source {src}")
# #         try:
# #             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(queue_size)))
# #         except Exception:
# #             pass

# #         self.q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
# #         self.stop_flag = threading.Event()
# #         self.dropped = 0
# #         self.read_dropped = 0
# #         self.thread = threading.Thread(target=self._loop, daemon=True)
# #         self.thread.start()

# #     def _drop_oldest(self) -> None:
# #         try:
# #             _ = self.q.get_nowait()
# #             self.dropped += 1
# #         except queue.Empty:
# #             return

# #     def _loop(self):
# #         while not self.stop_flag.is_set() and self.ok:
# #             ok, frame = self.cap.read()
# #             if not ok or frame is None:
# #                 time.sleep(0.005)
# #                 continue
# #             item = (frame, time.time())
# #             try:
# #                 self.q.put_nowait(item)
# #             except queue.Full:
# #                 self._drop_oldest()
# #                 try:
# #                     self.q.put_nowait(item)
# #                 except queue.Full:
# #                     self._drop_oldest()

# #     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
# #         try:
# #             frame, ts = self.q.get(timeout=0.1)
# #             return True, frame, float(ts)
# #         except queue.Empty:
# #             return False, None, 0.0

# #     def qsize(self) -> int:
# #         try:
# #             return int(self.q.qsize())
# #         except Exception:
# #             return 0

# #     def is_opened(self) -> bool:
# #         return bool(self.ok)

# #     def release(self):
# #         self.stop_flag.set()
# #         try:
# #             self.thread.join(timeout=1.0)
# #         except Exception:
# #             pass
# #         try:
# #             self.cap.release()
# #         except Exception:
# #             pass


# # class LatestStream:
# #     """
# #     Latest-frame reader (keeps only 1 frame).
# #     """

# #     def __init__(self, src: str, rtsp_buffer: int, decode_skip: int, rtsp_transport: str):
# #         self.src = src
# #         src_use = src
# #         if isinstance(src, str) and src.lower().startswith("rtsp"):
# #             sep = "&" if "?" in src_use else "?"
# #             src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"
# #         self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG)
# #         self.ok = self.cap.isOpened()
# #         if not self.ok:
# #             print(f"[WARN] cannot open source {src}")
# #         try:
# #             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(rtsp_buffer)))
# #         except Exception:
# #             pass
# #         self.decode_skip = max(0, int(decode_skip))
# #         self.q = queue.Queue(maxsize=1)
# #         self.stop_flag = threading.Event()
# #         self.dropped = 0
# #         self.read_dropped = 0
# #         self.thread = threading.Thread(target=self._loop, daemon=True)
# #         self.thread.start()

# #     def _loop(self):
# #         idx = 0
# #         while not self.stop_flag.is_set() and self.ok:
# #             ok, frame = self.cap.read()
# #             if not ok or frame is None:
# #                 time.sleep(0.01)
# #                 continue
# #             if self.decode_skip > 0:
# #                 if (idx % (self.decode_skip + 1)) != 0:
# #                     idx += 1
# #                     continue
# #                 idx += 1
# #             item = (frame, time.time())
# #             if not self.q.empty():
# #                 try:
# #                     _ = self.q.get_nowait()
# #                     self.dropped += 1
# #                 except queue.Empty:
# #                     pass
# #             try:
# #                 self.q.put_nowait(item)
# #             except queue.Full:
# #                 self.dropped += 1

# #     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
# #         try:
# #             frame, ts = self.q.get(timeout=0.02)
# #             return True, frame, float(ts)
# #         except queue.Empty:
# #             return False, None, 0.0

# #     def qsize(self) -> int:
# #         return 1 if not self.q.empty() else 0

# #     def is_opened(self) -> bool:
# #         return bool(self.ok)

# #     def release(self):
# #         self.stop_flag.set()
# #         try:
# #             self.thread.join(timeout=1.0)
# #         except Exception:
# #             pass
# #         try:
# #             self.cap.release()
# #         except Exception:
# #             pass


# # class FFmpegStream:
# #     """FFmpeg pipe reader (ordered). NVDEC optional."""

# #     def __init__(
# #         self,
# #         src: str,
# #         queue_size: int,
# #         rtsp_transport: str,
# #         use_cuda: bool,
# #         force_w: int = 0,
# #         force_h: int = 0,
# #     ):
# #         self.src = _sanitize_rtsp_url(src) if isinstance(src, str) and src.lower().startswith("rtsp") else src
# #         self.q = queue.Queue(maxsize=max(1, int(queue_size)))
# #         self.stop_flag = threading.Event()
# #         self.proc = None
# #         self.ok = False
# #         self.dropped = 0
# #         self.read_dropped = 0

# #         probed_w = probed_h = 0
# #         try:
# #             cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
# #             if cap.isOpened():
# #                 probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
# #                 probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
# #             cap.release()
# #         except Exception:
# #             pass

# #         scheme = ""
# #         try:
# #             scheme = urlsplit(self.src).scheme.lower()
# #         except Exception:
# #             pass

# #         cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", self.src, "-an", "-dn", "-sn", "-vsync", "0"]
# #         if scheme == "rtsp":
# #             cmd = [
# #                 "ffmpeg",
# #                 "-rtsp_transport",
# #                 rtsp_transport,
# #                 "-flags",
# #                 "+genpts",
# #                 "-fflags",
# #                 "+genpts",
# #                 "-use_wallclock_as_timestamps",
# #                 "1",
# #                 "-i",
# #                 self.src,
# #                 "-an",
# #                 "-dn",
# #                 "-sn",
# #                 "-vsync",
# #                 "0",
# #             ]
# #         cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]

# #         out_w = int(force_w) if force_w and force_w > 0 else int(probed_w)
# #         out_h = int(force_h) if force_h and force_h > 0 else int(probed_h)
# #         if out_w > 0 and out_h > 0:
# #             cmd += ["-vf", f"scale={out_w}:{out_h}"]
# #         else:
# #             out_w, out_h = 1280, 720
# #             cmd += ["-vf", "scale=1280:720"]

# #         if use_cuda:
# #             hw = [
# #                 "-hwaccel",
# #                 "cuda",
# #                 "-hwaccel_output_format",
# #                 "cuda",
# #                 "-vf",
# #                 f"hwdownload,format=bgr24,scale={out_w}:{out_h}",
# #             ]
# #             if "-vf" in cmd:
# #                 i = cmd.index("-vf")
# #                 cmd.pop(i)
# #                 prev = cmd.pop(i)
# #                 hw[-1] = f"hwdownload,format=bgr24,{prev},scale={out_w}:{out_h}" if "scale=" in prev else hw[-1]
# #             cmd += hw

# #         cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

# #         self.cmd = cmd
# #         self.width, self.height = int(out_w), int(out_h)
# #         self.frame_bytes = self.width * self.height * 3

# #         try:
# #             self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
# #             self.ok = True
# #         except Exception as e:
# #             print(f"[FFMPEG] start failed: {e}")
# #             self.ok = False

# #         self.thread = threading.Thread(target=self._loop, daemon=True)
# #         self.thread.start()
# #         self.log_thread = threading.Thread(target=self._log_stderr, daemon=True)
# #         self.log_thread.start()

# #     def _log_stderr(self):
# #         if not self.proc or not self.proc.stderr:
# #             return
# #         try:
# #             for _line in iter(self.proc.stderr.readline, b""):
# #                 if not _line:
# #                     break
# #         except Exception:
# #             pass

# #     def _drop_oldest(self) -> None:
# #         try:
# #             _ = self.q.get_nowait()
# #             self.dropped += 1
# #         except queue.Empty:
# #             return

# #     def _loop(self):
# #         if not self.proc or not self.proc.stdout or not self.ok:
# #             return
# #         fb = int(self.frame_bytes)
# #         w, h = int(self.width), int(self.height)
# #         while not self.stop_flag.is_set():
# #             buf = self.proc.stdout.read(fb)
# #             if not buf or len(buf) < fb:
# #                 time.sleep(0.001)
# #                 continue
# #             frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
# #             item = (frame, time.time())
# #             try:
# #                 self.q.put_nowait(item)
# #             except queue.Full:
# #                 self._drop_oldest()
# #                 try:
# #                     self.q.put_nowait(item)
# #                 except queue.Full:
# #                     self._drop_oldest()

# #     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
# #         try:
# #             frame, ts = self.q.get(timeout=0.1)
# #             return True, frame, float(ts)
# #         except queue.Empty:
# #             return False, None, 0.0

# #     def qsize(self) -> int:
# #         try:
# #             return int(self.q.qsize())
# #         except Exception:
# #             return 0

# #     def is_opened(self) -> bool:
# #         return bool(self.ok)

# #     def release(self):
# #         self.stop_flag.set()
# #         try:
# #             self.thread.join(timeout=1.0)
# #         except Exception:
# #             pass
# #         try:
# #             if self.proc:
# #                 self.proc.terminate()
# #                 self.proc.kill()
# #         except Exception:
# #             pass


# # # ----------------------------
# # # Frame buffer for MJPEG (single latest frame + cached JPEG)
# # # ----------------------------
# # class FrameBuffer:
# #     """
# #     Stores latest annotated frame.
# #     Single-cam MJPEG endpoint waits on condition and gets cached JPEG.
# #     Multi-cam MJPEG reads latest frames via get_frame().
# #     """

# #     def __init__(self):
# #         self._lock = threading.Lock()
# #         self._cond = threading.Condition(self._lock)

# #         self._frame_bgr: Optional[np.ndarray] = None
# #         self._ts: float = 0.0
# #         self._meta: Dict[str, Any] = {}

# #         self._jpeg: Optional[bytes] = None
# #         self._jpeg_ts: float = 0.0

# #         self._encode_lock = threading.Lock()
# #         self._clients = 0

# #     def add_client(self) -> None:
# #         with self._lock:
# #             self._clients += 1

# #     def remove_client(self) -> None:
# #         with self._lock:
# #             self._clients = max(0, self._clients - 1)

# #     def set(self, frame_bgr: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
# #         with self._cond:
# #             self._frame_bgr = frame_bgr
# #             self._ts = time.time()
# #             self._meta = dict(meta or {})
# #             # invalidate jpeg cache
# #             self._jpeg = None
# #             self._jpeg_ts = 0.0
# #             self._cond.notify_all()

# #     def get_meta(self) -> Dict[str, Any]:
# #         with self._lock:
# #             return dict(self._meta)

# #     def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
# #         """Used by multi-cam grid MJPEG."""
# #         with self._lock:
# #             return self._frame_bgr, float(self._ts)

# #     def wait_jpeg(self, last_ts: float, timeout: float, jpeg_quality: int = 80) -> Tuple[Optional[bytes], float]:
# #         """
# #         Wait until we have a frame newer than last_ts, then return cached JPEG.
# #         JPEG encoding is only useful when MJPEG clients exist.
# #         """
# #         with self._cond:
# #             if self._ts <= last_ts:
# #                 self._cond.wait(timeout=timeout)

# #             ts = self._ts
# #             frame = self._frame_bgr
# #             clients = self._clients

# #         if frame is None or ts <= 0:
# #             return None, 0.0

# #         if clients <= 0:
# #             return None, ts

# #         with self._lock:
# #             if self._jpeg is not None and self._jpeg_ts == ts:
# #                 return self._jpeg, ts

# #         with self._encode_lock:
# #             with self._lock:
# #                 if self._jpeg is not None and self._jpeg_ts == ts:
# #                     return self._jpeg, ts
# #                 frame_ref = self._frame_bgr
# #                 ts_copy = self._ts

# #             if frame_ref is None or ts_copy <= 0:
# #                 return None, 0.0

# #             ok, enc = cv2.imencode(
# #                 ".jpg",
# #                 frame_ref,
# #                 [int(cv2.IMWRITE_JPEG_QUALITY), int(max(30, min(95, jpeg_quality)))],
# #             )
# #             if not ok:
# #                 return None, ts_copy

# #             jpg = enc.tobytes()

# #             with self._lock:
# #                 if self._ts == ts_copy:
# #                     self._jpeg = jpg
# #                     self._jpeg_ts = ts_copy

# #             return jpg, ts_copy


# # # ----------------------------
# # # CSV SUMMARY REPORT (per person per camera logs)
# # # ----------------------------
# # @dataclass
# # class _ActiveSession:
# #     start_ts: float
# #     last_seen_ts: float


# # class SummaryReport:
# #     """
# #     Builds a summary CSV:
# #       Member | c1 | c2 | ... | Total time

# #     Each camera cell contains multiple lines:
# #       L1 - HH:MM:SS to HH:MM:SS
# #       L2 - HH:MM:SS to HH:MM:SS

# #     IMPORTANT: One row per person name (never creates duplicates later).
# #     """

# #     def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
# #         self.num_cams = int(max(1, num_cams))
# #         self.gap_seconds = float(max(0.0, gap_seconds))
# #         self.time_format = str(time_format or "%H:%M:%S")
# #         self._lock = threading.Lock()
# #         self._disabled = False

# #         self._first_seen: Dict[str, float] = {}
# #         self._active: Dict[Tuple[str, int], _ActiveSession] = {}
# #         self._logs: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
# #         self._total_seconds: Dict[str, float] = defaultdict(float)

# #     def stop(self) -> None:
# #         with self._lock:
# #             self._disabled = True

# #     def update(self, cam_id: int, present_names: List[str], ts: float) -> None:
# #         if ts <= 0:
# #             ts = time.time()
# #         cam_id = int(cam_id)

# #         with self._lock:
# #             if self._disabled:
# #                 return

# #             # close expired first
# #             self._close_expired_locked(now_ts=float(ts))

# #             for nm in present_names or []:
# #                 name = str(nm).strip()
# #                 if not name:
# #                     continue

# #                 if name not in self._first_seen:
# #                     self._first_seen[name] = float(ts)

# #                 key = (name, cam_id)
# #                 sess = self._active.get(key)
# #                 if sess is None:
# #                     self._active[key] = _ActiveSession(start_ts=float(ts), last_seen_ts=float(ts))
# #                 else:
# #                     sess.last_seen_ts = float(ts)

# #     def _close_expired_locked(self, now_ts: float) -> None:
# #         if self.gap_seconds <= 0:
# #             return
# #         to_close: List[Tuple[str, int, _ActiveSession]] = []
# #         for (name, cam_id), sess in list(self._active.items()):
# #             if (float(now_ts) - float(sess.last_seen_ts)) > self.gap_seconds:
# #                 to_close.append((name, cam_id, sess))

# #         for name, cam_id, sess in to_close:
# #             self._close_session_locked(name, cam_id, sess)

# #     def _close_session_locked(self, name: str, cam_id: int, sess: _ActiveSession) -> None:
# #         st = float(sess.start_ts)
# #         en = float(sess.last_seen_ts)
# #         if en < st:
# #             en = st

# #         self._logs[name][int(cam_id)].append((st, en))
# #         self._total_seconds[name] += max(0.0, (en - st))
# #         self._active.pop((name, int(cam_id)), None)

# #     def close_all(self) -> None:
# #         with self._lock:
# #             for (name, cam_id), sess in list(self._active.items()):
# #                 self._close_session_locked(name, cam_id, sess)

# #     def _fmt_time(self, ts: float) -> str:
# #         try:
# #             return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
# #         except Exception:
# #             return ""

# #     def _fmt_total(self, total_seconds: float) -> str:
# #         try:
# #             sec_f = float(total_seconds)
# #         except Exception:
# #             sec_f = 0.0

# #         sec_i = int(round(max(0.0, sec_f)))
# #         if sec_i == 0 and sec_f > 0.0:
# #             sec_i = 1

# #         minutes = int(sec_i // 60)
# #         seconds = int(sec_i % 60)

# #         m_word = "minute" if minutes == 1 else "minutes"
# #         s_word = "second" if seconds == 1 else "seconds"
# #         return f"{minutes} {m_word} {seconds} {s_word}"

# #     def write_csv(self, path: str) -> None:
# #         """Final write: closes all active sessions first."""
# #         self.close_all()

# #         with self._lock:
# #             items = list(self._first_seen.items())
# #             logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
# #             totals = dict(self._total_seconds)

# #         items.sort(key=lambda kv: kv[1])
# #         names_order = [nm for nm, _ in items]

# #         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# #         header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
# #         with open(path, "w", newline="", encoding="utf-8") as f:
# #             w = csv.writer(f)
# #             w.writerow(header)

# #             for name in names_order:
# #                 row = [name]
# #                 cam_map = logs.get(name, {})

# #                 for cam_id in range(self.num_cams):
# #                     entries = cam_map.get(cam_id, [])
# #                     lines = []
# #                     for idx, (st, en) in enumerate(entries, start=1):
# #                         lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
# #                     row.append("\n".join(lines))

# #                 row.append(self._fmt_total(totals.get(name, 0.0)))
# #                 w.writerow(row)

# #     def write_csv_snapshot(self, path: str) -> None:
# #         """
# #         Snapshot write: does NOT close active sessions permanently.
# #         Adds active segments as temporary logs for the snapshot.
# #         """
# #         now_ts = time.time()

# #         with self._lock:
# #             if self._disabled:
# #                 return

# #             self._close_expired_locked(now_ts=now_ts)

# #             items = list(self._first_seen.items())
# #             logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
# #             totals = dict(self._total_seconds)

# #             # include active segments temporarily
# #             for (name, cam_id), sess in self._active.items():
# #                 st = float(sess.start_ts)
# #                 en = float(sess.last_seen_ts)
# #                 logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en))
# #                 totals[name] = totals.get(name, 0.0) + max(0.0, (en - st))

# #         items.sort(key=lambda kv: kv[1])
# #         names_order = [nm for nm, _ in items]

# #         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# #         header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
# #         with open(path, "w", newline="", encoding="utf-8") as f:
# #             w = csv.writer(f)
# #             w.writerow(header)

# #             for name in names_order:
# #                 row = [name]
# #                 cam_map = logs.get(name, {})

# #                 for cam_id in range(self.num_cams):
# #                     entries = cam_map.get(cam_id, [])
# #                     lines = []
# #                     for idx, (st, en) in enumerate(entries, start=1):
# #                         lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
# #                     row.append("\n".join(lines))

# #                 row.append(self._fmt_total(totals.get(name, 0.0)))
# #                 w.writerow(row)


# # # ----------------------------
# # # Name de-duplication + global multi-camera ownership
# # # ----------------------------
# # @dataclass
# # class DrawItem:
# #     tid: int
# #     bbox: Tuple[int, int, int, int]
# #     name: str
# #     user_id: int
# #     face_sim: float
# #     stable_score: float
# #     det_conf: Optional[float]
# #     face_hit: bool


# # def _box_area_xyxy(b: Tuple[int, int, int, int]) -> int:
# #     x1, y1, x2, y2 = b
# #     return int(max(0, x2 - x1) * max(0, y2 - y1))


# # def _priority_tuple(it: DrawItem) -> tuple:
# #     return (
# #         1 if it.face_hit else 0,
# #         float(it.face_sim),
# #         float(it.stable_score),
# #         float(it.det_conf or 0.0),
# #         float(_box_area_xyxy(it.bbox)),
# #     )


# # def deduplicate_draw_items(items: List[DrawItem], iou_thresh: float) -> List[DrawItem]:
# #     if not items:
# #         return []

# #     groups: Dict[str, List[DrawItem]] = defaultdict(list)
# #     for it in items:
# #         if it.name:
# #             groups[it.name].append(it)

# #     kept: List[DrawItem] = []
# #     for _name, group in groups.items():
# #         group_sorted = sorted(group, key=_priority_tuple, reverse=True)

# #         if float(iou_thresh) <= 0.0:
# #             kept.append(group_sorted[0])
# #             continue

# #         selected: List[DrawItem] = []
# #         for it in group_sorted:
# #             ok = True
# #             for s in selected:
# #                 if iou_xyxy(it.bbox, s.bbox) >= float(iou_thresh):
# #                     ok = False
# #                     break
# #             if ok:
# #                 selected.append(it)
# #         kept.extend(selected)

# #     return kept


# # class GlobalNameOwner:
# #     def __init__(self, hold_seconds: float = 0.5, switch_margin: float = 0.02):
# #         self.hold_seconds = float(max(0.0, hold_seconds))
# #         self.switch_margin = float(max(0.0, switch_margin))
# #         self._lock = threading.Lock()
# #         self._state: Dict[str, Dict[str, Any]] = {}

# #     def _cleanup(self, now: float) -> None:
# #         if self.hold_seconds <= 0:
# #             return
# #         dead = []
# #         for name, st in self._state.items():
# #             ts = float(st.get("ts", 0.0))
# #             if (now - ts) > self.hold_seconds:
# #                 dead.append(name)
# #         for name in dead:
# #             self._state.pop(name, None)

# #     def allow(self, name: str, sid: int, score: float) -> bool:
# #         if not name:
# #             return False
# #         now = time.time()
# #         with self._lock:
# #             self._cleanup(now)

# #             st = self._state.get(name)
# #             if st is None:
# #                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
# #                 return True

# #             owner_sid = int(st.get("sid", -1))
# #             owner_score = float(st.get("score", 0.0))
# #             owner_ts = float(st.get("ts", 0.0))

# #             if owner_sid == int(sid):
# #                 st["score"] = max(owner_score, float(score))
# #                 st["ts"] = now
# #                 return True

# #             if self.hold_seconds > 0 and (now - owner_ts) > self.hold_seconds:
# #                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
# #                 return True

# #             if float(score) > (owner_score + self.switch_margin):
# #                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
# #                 return True

# #             return False


# # # ----------------------------
# # # CLI
# # # ----------------------------
# # def parse_args(argv: Optional[List[str]] = None):
# #     ap = argparse.ArgumentParser(
# #         "YOLO -> DeepSORT (TorchReID) with DB gallery tagging + Face-only labeling (body supports only).",
# #         conflict_handler="resolve",
# #     )

# #     # Sources
# #     ap.add_argument("--src", nargs="+", required=True, help="Video sources (RTSP/RTMP/HTTP/file).")

# #     # DB
# #     ap.add_argument("--use-db", action="store_true", help="Enable DB gallery.")
# #     ap.add_argument("--db-url", default="", help="SQLAlchemy DB URL.")
# #     ap.add_argument("--db-refresh-seconds", type=float, default=60.0, help="Reload DB gallery every N seconds (0=off).")
# #     ap.add_argument("--db-max-bank", type=int, default=256, help="Max embeddings per member to load (0=all).")
# #     ap.add_argument("--db-include-inactive", action="store_true", help="Include inactive members.")

# #     # YOLO
# #     ap.add_argument("--yolo-weights", default="yolov8n.pt")
# #     ap.add_argument("--yolo-imgsz", type=int, default=1280)
# #     ap.add_argument("--device", default="cuda:0")
# #     ap.add_argument("--conf", type=float, default=0.30)
# #     ap.add_argument("--iou", type=float, default=0.40)
# #     ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")

# #     # Runtime/perf
# #     ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
# #     ap.add_argument("--rtsp-buffer", type=int, default=2)
# #     ap.add_argument("--decode-skip", type=int, default=0)
# #     ap.add_argument("--reader", choices=["latest", "adaptive", "ffmpeg"], default="adaptive")
# #     ap.add_argument("--queue-size", type=int, default=128)
# #     ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
# #     ap.add_argument("--ffmpeg-cuda", action="store_true")
# #     ap.add_argument("--ffmpeg-width", type=int, default=0)
# #     ap.add_argument("--ffmpeg-height", type=int, default=0)
# #     ap.add_argument("--resize", type=int, nargs=2, default=[0, 0])

# #     # Adaptive skipping
# #     ap.add_argument("--max-queue-age-ms", type=int, default=1000)
# #     ap.add_argument("--max-drain-per-cycle", type=int, default=32)

# #     # DeepSORT / TorchReID
# #     g = ap.add_mutually_exclusive_group()
# #     g.add_argument("--no-deepsort", action="store_true")
# #     g.add_argument("--use-deepsort", action="store_true")

# #     ap.add_argument("--reid-model", default="osnet_x0_25")
# #     ap.add_argument("--reid-weights", default="")
# #     ap.add_argument("--reid-batch-size", type=int, default=16)

# #     ap.add_argument("--max-age", type=int, default=15)
# #     ap.add_argument("--n-init", type=int, default=3)
# #     ap.add_argument("--nn-budget", type=int, default=200)
# #     ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
# #     ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)

# #     # Body matching (support only)
# #     ap.add_argument("--gallery-thresh", type=float, default=0.70)
# #     ap.add_argument("--gallery-gap", type=float, default=0.08)
# #     ap.add_argument("--reid-topk", type=int, default=3)
# #     ap.add_argument("--min-box-wh", type=int, default=40)

# #     # Face (main identity)
# #     ap.add_argument("--use-face", action="store_true")
# #     ap.add_argument("--face-model", default="buffalo_l")
# #     ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
# #     ap.add_argument("--face-thresh", type=float, default=0.30)
# #     ap.add_argument("--face-gap", type=float, default=0.05)
# #     ap.add_argument("--face-every-n", type=int, default=1)
# #     ap.add_argument("--face-hold-frames", type=int, default=2)
# #     ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
# #     ap.add_argument("--ort-log", action="store_true")

# #     ap.add_argument("--face-iou-link", type=float, default=0.35)
# #     ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
# #     ap.add_argument("--face-center-in-person", action="store_true")

# #     # Extra heuristics
# #     ap.add_argument("--min-face-px", type=int, default=24)
# #     ap.add_argument("--min-face-area-ratio", type=float, default=0.006)
# #     ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70)
# #     ap.add_argument("--face-strong-thresh", type=float, default=0.50)

# #     # Identity smoothing
# #     ap.add_argument("--name-decay", type=float, default=0.85)
# #     ap.add_argument("--name-min-score", type=float, default=0.60)
# #     ap.add_argument("--name-margin", type=float, default=0.30)
# #     ap.add_argument("--name-ttl", type=int, default=20)
# #     ap.add_argument("--name-face-weight", type=float, default=1.2)
# #     ap.add_argument("--name-body-weight", type=float, default=0.5)

# #     # Drawing/ghost control
# #     ap.add_argument("--draw-only-matched", action="store_true")
# #     ap.add_argument("--min-det-conf", type=float, default=0.45)
# #     ap.add_argument("--iou-max-miss", type=int, default=5)

# #     # Duplicate-name suppression / multi-camera ownership
# #     ap.add_argument("--allow-duplicate-names", action="store_true")
# #     ap.add_argument("--dedup-iou", type=float, default=0.0)
# #     ap.add_argument("--no-global-unique-names", action="store_true")
# #     ap.add_argument("--global-hold-seconds", type=float, default=0.5)
# #     ap.add_argument("--global-switch-margin", type=float, default=0.02)
# #     ap.add_argument("--show-global-id", action="store_true")

# #     # Output & view (accepted for compatibility; service ignores)
# #     ap.add_argument("--show", action="store_true")

# #     # CSV report
# #     ap.add_argument("--save-csv", action="store_true")
# #     ap.add_argument("--csv", default="detections_summary.csv")
# #     ap.add_argument("--report-gap-seconds", type=float, default=2.0)
# #     ap.add_argument("--report-time-format", default="%H:%M:%S")

# #     # FPS overlays (not used in service output, but accepted)
# #     ap.add_argument("--overlay-fps", action="store_true")

# #     args = ap.parse_args(argv)
# #     return args


# # # ----------------------------
# # # Per-frame processing (FACE-ONLY labeling)
# # # ----------------------------
# # def process_one_frame(
# #     frame_idx: int,
# #     frame_bgr: np.ndarray,
# #     sid: int,
# #     yolo,
# #     args,
# #     deep_tracker,
# #     iou_tracker: IOUTracker,
# #     people: list[PersonEntry],
# #     reid_extractor,
# #     face_app,
# #     face_gallery: FaceGallery,
# #     identity_state: dict,
# #     device_is_cuda: bool,
# #     global_owner: Optional[GlobalNameOwner] = None,
# # ) -> Tuple[np.ndarray, Dict[str, Any]]:
# #     """
# #     Rules implemented:
# #       - Only label when a face is visible AND recognized.
# #       - Body never labels alone. It can only support the face label when it agrees.
# #       - No back-only labeling: if no face, no label is shown.
# #       - Known identities only.
# #     """

# #     # Optional resize
# #     rw, rh = int(args.resize[0]), int(args.resize[1])
# #     if rw > 0 and rh > 0:
# #         frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

# #     H, W = frame_bgr.shape[:2]

# #     # Map name -> member_id
# #     name_to_uid: Dict[str, int] = {}
# #     try:
# #         for p in people:
# #             name_to_uid[str(p.name)] = int(p.user_id)
# #     except Exception:
# #         name_to_uid = {}

# #     # YOLO detect
# #     tlwh_conf: list[list[float]] = []
# #     if yolo is not None:
# #         try:
# #             res = _yolo_forward_safe(yolo, frame_bgr, args)
# #             boxes = res[0].boxes if (res and len(res)) else None
# #             if boxes is not None:
# #                 xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
# #                 conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
# #                 cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
# #                 keep = cls == 0  # person
# #                 xyxy, conf = xyxy[keep], conf[keep]
# #                 for (x1, y1, x2, y2), c in zip(xyxy, conf):
# #                     x1f = float(max(0, min(W - 1, x1)))
# #                     y1f = float(max(0, min(H - 1, y1)))
# #                     x2f = float(max(0, min(W - 1, x2)))
# #                     y2f = float(max(0, min(H - 1, y2)))
# #                     ww = float(max(1.0, x2f - x1f))
# #                     hh = float(max(1.0, y2f - y1f))
# #                     if ww < args.min_box_wh or hh < args.min_box_wh:
# #                         continue
# #                     tlwh_conf.append([x1f, y1f, ww, hh, float(c)])
# #         except Exception as e:
# #             print(f"[SRC {sid}] YOLO error:", e)

# #     dets_np = np.asarray(tlwh_conf, dtype=np.float32)
# #     if dets_np.ndim != 2:
# #         dets_np = dets_np.reshape((0, 5)).astype(np.float32)

# #     dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

# #     # Tracker
# #     out_tracks = []
# #     if deep_tracker is not None:
# #         try:
# #             out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
# #         except Exception as e:
# #             print(f"[SRC {sid}] DeepSORT update_tracks error:", e)
# #             out_tracks = []
# #     else:
# #         out_tracks = iou_tracker.update(dets_np)

# #     # Face detect (every N frames)
# #     recognized_faces: List[Dict[str, Any]] = []
# #     do_face = (
# #         face_app is not None
# #         and face_gallery is not None
# #         and (not face_gallery.is_empty())
# #         and (frame_idx % max(1, int(args.face_every_n)) == 0)
# #     )
# #     if do_face:
# #         try:
# #             with _face_lock:
# #                 faces = face_app.get(np.ascontiguousarray(frame_bgr))
# #             for f in safe_iter_faces(faces):
# #                 bbox = getattr(f, "bbox", None)
# #                 if bbox is None:
# #                     continue
# #                 b = np.asarray(bbox).reshape(-1)
# #                 if b.size < 4:
# #                     continue
# #                 fx1, fy1, fx2, fy2 = map(float, b[:4])
# #                 fw = max(0.0, fx2 - fx1)
# #                 fh = max(0.0, fy2 - fy1)
# #                 if fw < float(args.min_face_px) or fh < float(args.min_face_px):
# #                     continue

# #                 emb = extract_face_embedding(f)
# #                 if emb is None:
# #                     continue
# #                 emb = l2_normalize(np.asarray(emb, dtype=np.float32))
# #                 flabel, fsim, fsecond = best_face_label_top2(emb, face_gallery)
# #                 if flabel is None:
# #                     continue

# #                 gap = float(fsim - fsecond)
# #                 if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
# #                     recognized_faces.append(
# #                         {
# #                             "bbox": (fx1, fy1, fx2, fy2),
# #                             "label": str(flabel),
# #                             "sim": float(fsim),
# #                             "second": float(fsecond),
# #                             "gap": float(gap),
# #                         }
# #                     )
# #         except Exception as e:
# #             print(f"[SRC {sid}] FaceAnalysis error:", e)

# #     out = frame_bgr.copy()
# #     tracks_info: List[Dict[str, Any]] = []

# #     for t in out_tracks:
# #         time_since_update = getattr(t, "time_since_update", 0)
# #         had_match_this_frame = (time_since_update == 0) or (getattr(t, "last_detection", None) is not None)
# #         if args.draw_only_matched and not had_match_this_frame:
# #             continue

# #         try:
# #             if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
# #                 continue
# #             if hasattr(t, "to_tlbr"):
# #                 ltrb = t.to_tlbr()
# #             elif hasattr(t, "to_ltrb"):
# #                 ltrb = t.to_ltrb()
# #             else:
# #                 ltrb = t.to_tlbr()

# #             x1, y1, x2, y2 = map(int, ltrb)
# #             x1 = int(max(0, min(W - 1, x1)))
# #             y1 = int(max(0, min(H - 1, y1)))
# #             x2 = int(max(0, min(W, x2)))
# #             y2 = int(max(0, min(H, y2)))
# #             if x2 <= x1 or y2 <= y1:
# #                 continue

# #             tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
# #         except Exception:
# #             continue

# #         det_conf = None
# #         try:
# #             if hasattr(t, "det_conf") and t.det_conf is not None:
# #                 det_conf = float(t.det_conf)
# #             elif hasattr(t, "last_detection") and t.last_detection is not None:
# #                 ld = t.last_detection
# #                 if isinstance(ld, (list, tuple)) and len(ld) >= 2:
# #                     det_conf = float(ld[1])
# #                 elif isinstance(ld, dict):
# #                     det_conf = float(ld.get("confidence", ld.get("det_conf", 0.0)))
# #         except Exception:
# #             det_conf = None

# #         if args.min_det_conf > 0 and det_conf is not None and det_conf < args.min_det_conf:
# #             if args.draw_only_matched:
# #                 continue

# #         # Face link to person track (recognized faces only)
# #         face_label, face_sim, face_gap = "", 0.0, 0.0
# #         face_hit = False

# #         if recognized_faces:
# #             t_xyxy = (float(x1), float(y1), float(x2), float(y2))
# #             p_area = float(max(1.0, (x2 - x1) * (y2 - y1)))

# #             best_idx, best_score = -1, -1e9
# #             for idx, fm in enumerate(recognized_faces):
# #                 fbox = fm["bbox"]

# #                 cx = 0.5 * (float(fbox[0]) + float(fbox[2]))
# #                 cy = 0.5 * (float(fbox[1]) + float(fbox[3]))
# #                 if args.face_center_in_person and not _point_in_xyxy(cx, cy, t_xyxy):
# #                     continue

# #                 top_limit = float(y1) + float(args.face_center_y_max_ratio) * float(max(1, (y2 - y1)))
# #                 if cy > top_limit:
# #                     continue

# #                 link = ioa_xyxy(fbox, t_xyxy) if args.face_link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)
# #                 if link < float(args.face_iou_link):
# #                     continue

# #                 if float(args.min_face_area_ratio) > 0:
# #                     f_area = float(max(1.0, (float(fbox[2]) - float(fbox[0])) * (float(fbox[3]) - float(fbox[1]))))
# #                     if (f_area / p_area) < float(args.min_face_area_ratio):
# #                         continue

# #                 score = float(link) + 0.25 * float(fm["sim"])
# #                 if score > best_score:
# #                     best_score = score
# #                     best_idx = idx

# #             if best_idx >= 0:
# #                 fm = recognized_faces[best_idx]
# #                 face_label = str(fm["label"])
# #                 face_sim = float(fm["sim"])
# #                 face_gap = float(fm["gap"])
# #                 face_hit = True

# #         # Update face visibility TTL for this track
# #         entry = identity_state.setdefault(
# #             tid,
# #             {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
# #         )
# #         dec = 1 if do_face else 0
# #         entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - dec)
# #         if face_hit and face_label:
# #             entry["face_vis_ttl"] = max(1, int(args.face_hold_frames))
# #             entry["last_face_label"] = face_label
# #             entry["last_face_sim"] = float(face_sim)

# #         tracks_info.append(
# #             {
# #                 "tid": tid,
# #                 "bbox": (x1, y1, x2, y2),
# #                 "det_conf": det_conf,
# #                 "face_hit": face_hit,
# #                 "face_label": face_label,
# #                 "face_sim": face_sim,
# #                 "face_gap": face_gap,
# #                 "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
# #                 "last_face_sim": float(entry.get("last_face_sim", 0.0)),
# #             }
# #         )

# #     # Body support: compute ONLY for tracks that have a recognized face hit this frame.
# #     face_hit_tracks = [r for r in tracks_info if r["face_hit"] and r["face_label"]]
# #     body_by_tid: Dict[int, Tuple[str, float, float]] = {}

# #     if face_hit_tracks and (reid_extractor is not None) and people:
# #         crops: List[np.ndarray] = []
# #         tids: List[int] = []
# #         for r in face_hit_tracks:
# #             x1, y1, x2, y2 = r["bbox"]
# #             crop = frame_bgr[y1:y2, x1:x2]
# #             crops.append(crop)
# #             tids.append(int(r["tid"]))

# #         bs = 16
# #         if hasattr(args, "reid_batch_size"):
# #             try:
# #                 bs = max(1, int(getattr(args, "reid_batch_size")))
# #             except Exception:
# #                 bs = 16

# #         for start in range(0, len(crops), bs):
# #             chunk = crops[start : start + bs]
# #             chunk_tids = tids[start : start + bs]
# #             feats = extract_body_embeddings_batch(reid_extractor, chunk, device_is_cuda=device_is_cuda, use_half=bool(args.half))
# #             if feats is None:
# #                 continue
# #             for tid, emb in zip(chunk_tids, feats):
# #                 blabel, bsim, bsecond = best_body_label_from_emb(emb, people, topk=max(1, int(args.reid_topk)))
# #                 if blabel is None:
# #                     continue
# #                 body_by_tid[int(tid)] = (str(blabel), float(bsim), float(bsecond))

# #     draw_candidates: List[DrawItem] = []

# #     for r in tracks_info:
# #         tid = int(r["tid"])
# #         x1, y1, x2, y2 = r["bbox"]

# #         face_hit = bool(r["face_hit"])
# #         face_label = str(r["face_label"])
# #         face_sim = float(r["face_sim"])
# #         last_face_sim = float(r.get("last_face_sim", 0.0))

# #         candidates: List[Tuple[str, float, str]] = []

# #         if face_hit and face_label:
# #             candidates.append((face_label, face_sim, "face"))

# #             if tid in body_by_tid:
# #                 b_label, b_sim, b_second = body_by_tid[tid]
# #                 b_gap = float(b_sim - b_second)
# #                 if (b_label == face_label) and (b_sim >= float(args.gallery_thresh)) and (b_gap >= float(args.gallery_gap)):
# #                     candidates.append((face_label, b_sim, "body"))
# #                 else:
# #                     if (b_label != face_label) and (b_sim >= max(float(args.gallery_thresh), 0.80)) and (b_gap >= max(float(args.gallery_gap), 0.10)):
# #                         if face_sim < float(args.face_strong_thresh):
# #                             candidates = []
# #         W
# #         # If face didn't run this frame AND we have no new evidence,
# #         # keep the last identity without decaying (prevents flicker)
# #         entry = identity_state.setdefault(
# #             tid,
# #             {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
# #         )

# #         if (not do_face) and (not candidates):
# #             stable_name = str(entry.get("last", "") or "")
# #             stable_score = float(entry.get("scores", {}).get(stable_name, 0.0)) if stable_name else 0.0
# #         else:
# #             stable_name, stable_score, entry = update_track_identity(
# #                 identity_state,
# #                 tid,
# #                 candidates,
# #                 decay=args.name_decay,
# #                 min_score=args.name_min_score,
# #                 margin=args.name_margin,
# #                 ttl_reset=args.name_ttl,
# #                 w_face=args.name_face_weight,
# #                 w_body=args.name_body_weight,
# #             )


# #         show_label = bool(stable_name) and (int(entry.get("face_vis_ttl", 0)) > 0)
# #         if not show_label:
# #             continue

# #         disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))
# #         uid = int(name_to_uid.get(stable_name, -1))

# #         draw_candidates.append(
# #             DrawItem(
# #                 tid=tid,
# #                 bbox=(int(x1), int(y1), int(x2), int(y2)),
# #                 name=str(stable_name),
# #                 user_id=uid,
# #                 face_sim=float(disp_face_sim),
# #                 stable_score=float(stable_score),
# #                 det_conf=r.get("det_conf", None),
# #                 face_hit=bool(face_hit),
# #             )
# #         )

# #     # For report: names present in this camera frame (before dedup/global gating)
# #     present_names = sorted({it.name for it in draw_candidates if it.name})

# #     # Per-camera duplicate suppression
# #     if not bool(getattr(args, "allow_duplicate_names", False)):
# #         draw_final = deduplicate_draw_items(draw_candidates, iou_thresh=float(getattr(args, "dedup_iou", 0.0)))
# #     else:
# #         draw_final = draw_candidates

# #     # Optional: cross-camera global unique names (display only)
# #     if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
# #         gated: List[DrawItem] = []
# #         for it in draw_final:
# #             score = float(it.face_sim)
# #             if global_owner.allow(it.name, sid=int(sid), score=score):
# #                 gated.append(it)
# #         draw_final = gated

# #     # Draw
# #     shown = 0
# #     for it in draw_final:
# #         x1, y1, x2, y2 = it.bbox
# #         color = (0, 255, 0)
# #         cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

# #         if bool(getattr(args, "show_global_id", False)) and it.user_id >= 0:
# #             label_txt = f"{it.name} [{it.user_id}]"
# #         else:
# #             label_txt = f"{it.name}"

# #         if it.face_hit:
# #             label_txt = f"{label_txt} | F {float(it.face_sim):.2f}"

# #         cv2.putText(out, label_txt, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
# #         shown += 1

# #     meta = {
# #         "tracks": int(len(tracks_info)),
# #         "shown": int(shown),
# #         "faces_recognized": int(len(recognized_faces)),
# #         "do_face": bool(do_face),
# #         "present_names": present_names,
# #     }
# #     return out, meta


# # def processor_thread(
# #     sid: int,
# #     vs,
# #     out_buf: FrameBuffer,
# #     yolo,
# #     args,
# #     deep_tracker,
# #     iou_tracker: IOUTracker,
# #     gallery_mgr: GalleryManager,
# #     reid_extractor,
# #     face_app,
# #     global_owner: Optional[GlobalNameOwner],
# #     report: Optional[SummaryReport],
# #     stop_event: threading.Event,
# # ):
# #     frame_idx = 0
# #     identity_state: dict[int, dict] = {}

# #     device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

# #     while not stop_event.is_set():
# #         ok, frame, ts_cap = vs.read()
# #         if not ok or frame is None:
# #             time.sleep(0.005)
# #             continue

# #         # Drop stale frames policy
# #         if int(args.max_queue_age_ms) > 0:
# #             now = time.time()
# #             age_ms = (now - float(ts_cap)) * 1000.0
# #             dropped_here = 0
# #             while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
# #                 try:
# #                     vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
# #                 except Exception:
# #                     pass
# #                 ok2, frame2, ts2 = vs.read()
# #                 if not ok2 or frame2 is None:
# #                     break
# #                 frame, ts_cap = frame2, ts2
# #                 age_ms = (time.time() - float(ts_cap)) * 1000.0
# #                 dropped_here += 1

# #         try:
# #             gallery_mgr.maybe_reload(args)
# #             people, face_gallery = gallery_mgr.snapshot()

# #             out, meta = process_one_frame(
# #                 frame_idx=frame_idx,
# #                 frame_bgr=frame,
# #                 sid=sid,
# #                 yolo=yolo,
# #                 args=args,
# #                 deep_tracker=deep_tracker,
# #                 iou_tracker=iou_tracker,
# #                 people=people,
# #                 reid_extractor=reid_extractor,
# #                 face_app=face_app,
# #                 face_gallery=face_gallery,
# #                 identity_state=identity_state,
# #                 device_is_cuda=device_is_cuda,
# #                 global_owner=global_owner,
# #             )

# #             # -----------------------------
# #             # 🔥 DEBUG OVERLAY (VERY IMPORTANT)
# #             # -----------------------------
# #             tracks_count = meta.get("tracks", 0)
# #             shown_count = meta.get("shown", 0)
# #             faces_count = meta.get("faces_recognized", 0)

# #             cv2.putText(out, f"Tracks: {tracks_count}", (20, 40),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# #             cv2.putText(out, f"Shown: {shown_count}", (20, 80),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# #             cv2.putText(out, f"Faces: {faces_count}", (20, 120),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# #             # Also print to terminal every 30 frames
# #             if frame_idx % 30 == 0:
# #                 print(f"[SRC {sid}] Tracks={tracks_count} Shown={shown_count} Faces={faces_count}")

# #             # -----------------------------

# #             # Report update
# #             if report is not None:
# #                 names = meta.get("present_names", []) or []
# #                 ts_use = float(ts_cap) if ts_cap else time.time()
# #                 report.update(cam_id=int(sid), present_names=list(names), ts=ts_use)

# #             out_buf.set(out, meta=meta)
# #             frame_idx += 1

# #         except Exception as e:
# #             print(f"[PROC {sid}] error:", e)
# #             time.sleep(0.001)


# # # ----------------------------
# # # Runner
# # # ----------------------------
# # class TrackingRunner:
# #     def __init__(self, args):
# #         self.args = args
# #         self._lock = threading.Lock()
# #         self._started = False

# #         self.gallery_mgr: Optional[GalleryManager] = None
# #         self.global_owner: Optional[GlobalNameOwner] = None

# #         self.yolo = None
# #         self.reid_extractor = None
# #         self.face_app = None

# #         self.report: Optional[SummaryReport] = None

# #         self.streams: list[dict[str, Any]] = []
# #         self.buffers: list[FrameBuffer] = []
# #         self.threads: list[threading.Thread] = []
# #         self.stop_event = threading.Event()

# #     def start(self) -> None:
# #         with self._lock:
# #             if self._started:
# #                 return
# #             self._started = True

# #         args = self.args
# #         if not args.use_db:
# #             raise RuntimeError("DB-first: pass --use-db")
# #         if not args.db_url:
# #             raise RuntimeError("--db-url is required")

# #         # Global unique-name behavior (display only)
# #         args.global_unique_names = (len(args.src) > 1) and (not bool(getattr(args, "no_global_unique_names", False)))

# #         if args.cudnn_benchmark:
# #             torch.backends.cudnn.benchmark = True

# #         try:
# #             torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
# #         except Exception:
# #             pass

# #         gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
# #         if args.half and not gpu:
# #             print("[WARN] --half requested but CUDA not available; disabling FP16.")
# #             args.half = False

# #         print(f"[INIT] device={args.device} cuda_available={torch.cuda.is_available()} half={args.half}")

# #         # DB gallery
# #         self.gallery_mgr = GalleryManager(args)

# #         # Global unique names gate
# #         if bool(getattr(args, "global_unique_names", False)):
# #             self.global_owner = GlobalNameOwner(
# #                 hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
# #                 switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
# #             )
# #             print(f"[INIT] Global unique names: ON (hold={self.global_owner.hold_seconds}s, margin={self.global_owner.switch_margin})")
# #         else:
# #             print("[INIT] Global unique names: OFF")

# #         # YOLO
# #         if YOLO is not None:
# #             try:
# #                 weights = args.yolo_weights
# #                 if not Path(weights).exists():
# #                     print(f"[INIT] {weights} not found, falling back to yolov8n.pt")
# #                     weights = "yolov8n.pt"
# #                 self.yolo = YOLO(weights)
# #                 if gpu:
# #                     try:
# #                         self.yolo.to(args.device)
# #                     except Exception as e:
# #                         print("[WARN] YOLO .to(device) failed:", e)
# #                 print("[INIT] YOLO ready")
# #             except Exception as e:
# #                 print("[ERROR] YOLO load failed:", e)
# #                 self.yolo = None
# #         else:
# #             print("[WARN] ultralytics not installed; detection disabled")

# #         # TorchReID extractor (body support only)
# #         if TorchreidExtractor is not None:
# #             try:
# #                 dev = args.device if gpu else "cpu"
# #                 if args.reid_weights and Path(args.reid_weights).exists():
# #                     self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, model_path=args.reid_weights, device=dev)
# #                 else:
# #                     self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, device=dev)
# #                 print(f"[INIT] TorchReID ready (model={args.reid_model}, device={dev})")
# #             except Exception as e:
# #                 print("[WARN] TorchReID init failed; body support disabled:", e)
# #                 self.reid_extractor = None
# #         else:
# #             print("[WARN] torchreid not installed; body support disabled")

# #         # Face
# #         self.face_app = init_face_engine(
# #             args.use_face,
# #             args.device,
# #             args.face_model,
# #             int(args.face_det_size[0]),
# #             int(args.face_det_size[1]),
# #             face_provider=getattr(args, "face_provider", "auto"),
# #             ort_log=getattr(args, "ort_log", False),
# #         )

# #         # DeepSORT usage
# #         enable_deepsort = False
# #         if args.no_deepsort:
# #             enable_deepsort = False
# #         else:
# #             enable_deepsort = DeepSort is not None
# #         if args.use_deepsort and DeepSort is None:
# #             print("[WARN] --use-deepsort requested but deep-sort-realtime not installed.")

# #         # Summary report
# #         if args.save_csv:
# #             self.report = SummaryReport(
# #                 num_cams=len(args.src),
# #                 gap_seconds=float(getattr(args, "report_gap_seconds", 2.0)),
# #                 time_format=str(getattr(args, "report_time_format", "%H:%M:%S")),
# #             )
# #             print(f"[INIT] Summary CSV report: ON -> {args.csv}")
# #         else:
# #             self.report = None

# #         # Open sources + start workers
# #         self.streams = []
# #         self.buffers = []
# #         self.threads = []

# #         for i, raw_src in enumerate(args.src):
# #             src = raw_src.strip() if isinstance(raw_src, str) else raw_src

# #             if args.reader == "latest":
# #                 vs = LatestStream(src, rtsp_buffer=args.rtsp_buffer, decode_skip=args.decode_skip, rtsp_transport=args.rtsp_transport)
# #             elif args.reader == "ffmpeg":
# #                 vs = FFmpegStream(
# #                     src,
# #                     queue_size=args.queue_size,
# #                     rtsp_transport=args.rtsp_transport,
# #                     use_cuda=bool(args.ffmpeg_cuda),
# #                     force_w=int(args.ffmpeg_width),
# #                     force_h=int(args.ffmpeg_height),
# #                 )
# #             else:
# #                 vs = AdaptiveQueueStream(src, queue_size=args.queue_size, rtsp_transport=args.rtsp_transport, use_opencv=True)

# #             deep_tracker = None
# #             if enable_deepsort:
# #                 try:
# #                     deep_tracker = DeepSort(
# #                         max_age=int(args.max_age),
# #                         n_init=int(args.n_init),
# #                         nn_budget=int(args.nn_budget),
# #                         max_cosine_distance=float(args.tracker_max_cosine),
# #                         nms_max_overlap=float(args.tracker_nms_overlap),
# #                         embedder="torchreid",
# #                         embedder_gpu=gpu,
# #                         half=(gpu and args.half),
# #                         bgr=True,
# #                     )
# #                 except Exception as e:
# #                     print(f"[WARN] DeepSORT init failed for SRC {i}, fallback to IoU tracker:", e)
# #                     deep_tracker = None

# #             iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=0.3)
# #             buf = FrameBuffer()
# #             self.buffers.append(buf)

# #             self.streams.append({"sid": i, "src": raw_src, "vs": vs, "deep": deep_tracker, "iou": iou_tracker})

# #         print(f"[INIT] sources requested: {len(args.src)}")
# #         for s in self.streams:
# #             print(f"[SRC {s['sid']}] open={s['vs'].is_opened()} :: {s['src']}")

# #         if not any(s["vs"].is_opened() for s in self.streams):
# #             raise RuntimeError("No sources opened. Check your --src URLs/paths and codecs.")

# #         for s in self.streams:
# #             sid = int(s["sid"])
# #             t = threading.Thread(
# #                 target=processor_thread,
# #                 args=(
# #                     sid,
# #                     s["vs"],
# #                     self.buffers[sid],
# #                     self.yolo,
# #                     args,
# #                     s["deep"],
# #                     s["iou"],
# #                     self.gallery_mgr,
# #                     self.reid_extractor,
# #                     self.face_app,
# #                     self.global_owner,
# #                     self.report,
# #                     self.stop_event,
# #                 ),
# #                 daemon=True,
# #             )
# #             t.start()
# #             self.threads.append(t)

# #         print("[Runner] Background detection started.")

# #     def stop(self) -> None:
# #         with self._lock:
# #             if not self._started:
# #                 return
# #             self._started = False

# #         print("[Runner] Stopping...")
# #         self.stop_event.set()

# #         for s in self.streams:
# #             try:
# #                 s["vs"].release()
# #             except Exception:
# #                 pass

# #         for t in self.threads:
# #             try:
# #                 t.join(timeout=2.0)
# #             except Exception:
# #                 pass

# #         if self.report is not None:
# #             try:
# #                 self.report.stop()
# #                 # final write on shutdown
# #                 self.report.write_csv(self.args.csv)
# #                 print(f"[DONE] Summary CSV saved: {self.args.csv}")
# #             except Exception as e:
# #                 print("[WARN] Failed to write summary CSV:", e)

# #         print("[Runner] Stopped.")

# #     def num_cameras(self) -> int:
# #         return len(self.buffers)

# #     def cameras_brief(self) -> List[Dict[str, Any]]:
# #         out = []
# #         for s in self.streams:
# #             out.append(
# #                 {
# #                     "id": int(s["sid"]),
# #                     "opened": bool(s["vs"].is_opened()),
# #                     "src": str(s["src"]),
# #                 }
# #             )
# #         return out

# #     def get_buffer(self, cam_id: int) -> Optional[FrameBuffer]:
# #         if cam_id < 0 or cam_id >= len(self.buffers):
# #             return None
# #         return self.buffers[cam_id]

# #     def db_last_reload_ts(self) -> float:
# #         if self.gallery_mgr is None:
# #             return 0.0
# #         return float(self.gallery_mgr.last_load_ts)

# #     def write_report_snapshot(self, path: Optional[str] = None) -> str:
# #         if self.report is None:
# #             raise RuntimeError("CSV reporting not enabled. Add --save-csv in PIPELINE_ARGS.")
# #         out_path = str(path or self.args.csv)
# #         self.report.write_csv_snapshot(out_path)
# #         return out_path


# # app/services/tracking/pipeline.py
# """
# Person tracking pipeline with YOLO + DeepSORT.

# ✅ IMPORTANT (your requirement):
# - DO NOT draw bounding boxes for UNKNOWN people
# - Draw GREEN boxes only for KNOWN / identified members (face-recognized + stable)

# ✅ DB changes:
# - members table instead of users
# - embeddings are in member_embeddings table (member_id + camera_id)

# ✅ Camera source:
# - If you pass --src RTSP URLs => works like old manual mode (cam_id = 0..N-1)
# - If you pass --src as camera IDs (numbers) OR omit --src => DB camera mode (cam_id = DB camera.id)
#   -> RTSP URL is built from cameras.ip_address + RTSP_* env template

# This file is designed for FastAPI service mode (MJPEG output).
# """

# from __future__ import annotations

# import argparse
# import ctypes
# import csv
# import gzip
# import os
# import queue
# import shlex
# import subprocess
# import sys
# import threading
# import time
# from collections import defaultdict
# from dataclasses import dataclass
# from datetime import datetime
# from io import BytesIO
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union
# from urllib.parse import quote, unquote, urlsplit, urlunsplit

# import cv2
# import numpy as np
# import torch

# # --- YOLO (Ultralytics) ---
# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# # --- DeepSORT (deep-sort-realtime) ---
# try:
#     from deep_sort_realtime.deepsort_tracker import DeepSort
# except Exception:
#     DeepSort = None

# # --- TorchReID gallery extractor (body support only) ---
# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# except Exception:
#     TorchreidExtractor = None

# # --- InsightFace ---
# try:
#     from insightface.app import FaceAnalysis

#     INSIGHT_OK = True
# except Exception:
#     FaceAnalysis = None
#     INSIGHT_OK = False

# # --- ONNX Runtime ---
# try:
#     import onnxruntime as ort
# except Exception:
#     ort = None

# # --- SQLAlchemy / pgvector ---
# try:
#     from sqlalchemy import (
#         ARRAY,
#         BigInteger,
#         Boolean,
#         Column,
#         DateTime,
#         Float,
#         ForeignKey,
#         Integer,
#         LargeBinary,
#         String,
#         create_engine,
#         func,
#         select,
#     )
#     from sqlalchemy.orm import declarative_base, sessionmaker
# except Exception as e:
#     raise RuntimeError("SQLAlchemy is required for --use-db mode") from e

# try:
#     from pgvector.sqlalchemy import Vector
# except Exception:
#     Vector = None

# # ----------------------------
# # Globals (guards)
# # ----------------------------
# EXPECTED_DIM = 512
# _yolo_lock = threading.Lock()  # avoid concurrent YOLO kernels on same instance
# _reid_lock = threading.Lock()  # avoid concurrent TorchReID kernels on same extractor
# _face_lock = threading.Lock()  # InsightFace (ORT/CUDA) guard

# # ----------------------------
# # Utils
# # ----------------------------


# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n


# def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
#     m = np.asarray(m, dtype=np.float32)
#     if m.ndim != 2:
#         return m
#     norms = np.linalg.norm(m, axis=1, keepdims=True)
#     norms = np.where((norms == 0) | (~np.isfinite(norms)), 1.0, norms)
#     return m / norms


# def safe_iter_faces(obj):
#     if obj is None:
#         return []
#     try:
#         return list(obj)
#     except TypeError:
#         return [obj]


# def extract_face_embedding(face):
#     emb = getattr(face, "normed_embedding", None)
#     if emb is None:
#         emb = getattr(face, "embedding", None)
#     return emb


# def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# def _sanitize_rtsp_url(url: str) -> str:
#     parts = urlsplit(url)
#     if parts.username or parts.password:
#         user = quote(unquote(parts.username or ""), safe="")
#         pwd = quote(unquote(parts.password or ""), safe="")
#         host = parts.hostname or ""
#         netloc = f"{user}:{pwd}@{host}"
#         if parts.port:
#             netloc += f":{parts.port}"
#         return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
#     return url


# def _looks_like_url(s: str) -> bool:
#     s = str(s or "").strip().lower()
#     return "://" in s or s.startswith("rtsp:") or s.startswith("http:") or s.startswith("https:")


# def _is_int_str(s: str) -> bool:
#     s = str(s or "").strip()
#     if not s:
#         return False
#     if s.startswith("-"):
#         return s[1:].isdigit()
#     return s.isdigit()


# # ----------------------------
# # ORT CUDA provider probe
# # ----------------------------


# def _cuda_ep_loadable() -> bool:
#     if ort is None:
#         return False
#     try:
#         if sys.platform.startswith("darwin"):
#             return False
#         capi_dir = Path(ort.__file__).parent / "capi"
#         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
#         lib_path = capi_dir / name
#         if not lib_path.exists():
#             return False
#         ctypes.CDLL(str(lib_path))
#         return True
#     except Exception:
#         return False


# # ----------------------------
# # DB gallery + cameras (members + member_embeddings)
# # ----------------------------


# def decode_bank_blob(raw: bytes | None) -> np.ndarray | None:
#     """
#     Supports:
#       - gzip(np.load(npy))
#       - raw npy bytes

#     Returns normalized float32 [N,512] or None.
#     """
#     if not raw:
#         return None

#     # Try gzip first
#     for mode in ("gzip", "plain"):
#         try:
#             if mode == "gzip":
#                 data = gzip.decompress(raw)
#             else:
#                 data = raw

#             arr = np.load(BytesIO(data), allow_pickle=False)
#             arr = np.asarray(arr, dtype=np.float32)

#             if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
#                 continue
#             if not np.isfinite(arr).all():
#                 continue
#             return l2_normalize_rows(arr)
#         except Exception:
#             continue

#     return None


# def _as_vec512(x) -> np.ndarray | None:
#     if x is None:
#         return None
#     try:
#         a = np.asarray(x, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_DIM:
#             return None
#         if not np.isfinite(a).all():
#             return None
#         return l2_normalize(a)
#     except Exception:
#         return None


# @dataclass
# class PersonEntry:
#     user_id: int  # (kept name for compatibility) -> member_id
#     name: str
#     body_bank: np.ndarray | None
#     body_centroid: np.ndarray | None
#     face_centroid: np.ndarray | None


# @dataclass
# class FaceGallery:
#     names: list[str]
#     mat: np.ndarray

#     def is_empty(self) -> bool:
#         return (not self.names) or (self.mat is None) or (self.mat.size == 0)


# @dataclass
# class _Agg:
#     body_vecs: list[np.ndarray]
#     face_vecs: list[np.ndarray]
#     body_banks: list[np.ndarray]
#     face_banks: list[np.ndarray]


# class GalleryData:
#     def __init__(self):
#         self.people_global: list[PersonEntry] = []
#         self.face_global: FaceGallery = FaceGallery([], np.zeros((0, EXPECTED_DIM), dtype=np.float32))
#         self.people_by_cam: dict[int, list[PersonEntry]] = {}
#         self.face_by_cam: dict[int, FaceGallery] = {}


# def _member_label(member_number: str | None, first_name: str | None, last_name: str | None, member_id: int) -> str:
#     fn = (first_name or "").strip()
#     ln = (last_name or "").strip()
#     mn = (member_number or "").strip()
#     if fn and ln:
#         return f"{fn} {ln}".strip()
#     if fn:
#         return fn
#     if mn:
#         return mn
#     return f"member_{int(member_id)}"


# def build_galleries_from_db(
#     db_url: str,
#     active_only: bool = True,
#     max_bank_per_member: int = 0,
# ) -> GalleryData:
#     """
#     Loads:
#       - members
#       - member_embeddings (per camera)

#     Builds:
#       - global gallery (all cameras)
#       - per-camera galleries (camera_id-specific), with fallback possible.
#     """
#     Base = declarative_base()

#     class MemberRow(Base):
#         __tablename__ = "members"
#         id = Column(Integer, primary_key=True)
#         member_number = Column(String(16))
#         first_name = Column(String(64))
#         last_name = Column(String(64))
#         is_active = Column(Boolean)

#     class MemberEmbeddingRow(Base):
#         __tablename__ = "member_embeddings"
#         id = Column(BigInteger, primary_key=True)

#         member_id = Column(Integer, ForeignKey("members.id"))
#         camera_id = Column(Integer, ForeignKey("cameras.id"))

#         if Vector is not None:
#             body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
#             face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
#             back_body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
#         else:
#             body_embedding = Column(ARRAY(Float), nullable=True)
#             face_embedding = Column(ARRAY(Float), nullable=True)
#             back_body_embedding = Column(ARRAY(Float), nullable=True)

#         body_embeddings_raw = Column(LargeBinary, nullable=True)
#         face_embeddings_raw = Column(LargeBinary, nullable=True)
#         back_body_embeddings_raw = Column(LargeBinary, nullable=True)

#         last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True, server_default=func.now())

#     engine = create_engine(db_url, pool_pre_ping=True)
#     Session = sessionmaker(bind=engine)

#     # member_id -> metadata
#     member_meta: dict[int, dict[str, Any]] = {}

#     # global agg: member_id -> _Agg
#     agg_global: dict[int, _Agg] = defaultdict(lambda: _Agg([], [], [], []))

#     # camera-specific agg: camera_id -> member_id -> _Agg
#     agg_by_cam: dict[int, dict[int, _Agg]] = defaultdict(lambda: defaultdict(lambda: _Agg([], [], [], [])))

#     # Load all member+embedding rows
#     with Session() as session:
#         stmt = (
#             select(
#                 MemberRow.id,
#                 MemberRow.member_number,
#                 MemberRow.first_name,
#                 MemberRow.last_name,
#                 MemberRow.is_active,
#                 MemberEmbeddingRow.camera_id,
#                 MemberEmbeddingRow.body_embedding,
#                 MemberEmbeddingRow.face_embedding,
#                 MemberEmbeddingRow.back_body_embedding,
#                 MemberEmbeddingRow.body_embeddings_raw,
#                 MemberEmbeddingRow.face_embeddings_raw,
#                 MemberEmbeddingRow.back_body_embeddings_raw,
#             )
#             .select_from(MemberRow)
#             .join(MemberEmbeddingRow, MemberEmbeddingRow.member_id == MemberRow.id, isouter=False)
#         )

#         rows = session.execute(stmt).all()

#         # A per-member cap on how many raw embeddings we load (global cap)
#         raw_loaded_body: dict[int, int] = defaultdict(int)
#         raw_loaded_face: dict[int, int] = defaultdict(int)

#         # Also cap per-member-per-camera (so one camera doesn't dominate)
#         raw_loaded_body_cam: dict[tuple[int, int], int] = defaultdict(int)
#         raw_loaded_face_cam: dict[tuple[int, int], int] = defaultdict(int)

#         for r in rows:
#             member_id = int(r.id)
#             is_active = bool(r.is_active) if r.is_active is not None else True
#             if active_only and (not is_active):
#                 continue

#             cam_id = int(r.camera_id)

#             if member_id not in member_meta:
#                 member_meta[member_id] = {
#                     "member_number": r.member_number,
#                     "first_name": r.first_name,
#                     "last_name": r.last_name,
#                     "is_active": is_active,
#                 }

#             # ---- aggregated vectors
#             b = _as_vec512(r.body_embedding)
#             f = _as_vec512(r.face_embedding)
#             bb = _as_vec512(r.back_body_embedding)

#             if b is not None:
#                 agg_global[member_id].body_vecs.append(b)
#                 agg_by_cam[cam_id][member_id].body_vecs.append(b)

#             if bb is not None:
#                 # Optional: treat back-body as additional body evidence
#                 agg_global[member_id].body_vecs.append(bb)
#                 agg_by_cam[cam_id][member_id].body_vecs.append(bb)

#             if f is not None:
#                 agg_global[member_id].face_vecs.append(f)
#                 agg_by_cam[cam_id][member_id].face_vecs.append(f)

#             # ---- raw banks (optional)
#             max_bank = int(max_bank_per_member or 0)
#             # body raw
#             if r.body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[member_id] < max_bank):
#                 bank = decode_bank_blob(r.body_embeddings_raw)
#                 if bank is not None and bank.size > 0:
#                     remain = max_bank - raw_loaded_body[member_id] if max_bank > 0 else bank.shape[0]
#                     take = bank[: max(0, remain)] if max_bank > 0 else bank
#                     if take.size > 0:
#                         agg_global[member_id].body_banks.append(take)
#                         raw_loaded_body[member_id] += int(take.shape[0])

#                     # per-camera cap too
#                     key = (member_id, cam_id)
#                     if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
#                         remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
#                         take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
#                         if take2.size > 0:
#                             agg_by_cam[cam_id][member_id].body_banks.append(take2)
#                             raw_loaded_body_cam[key] += int(take2.shape[0])

#             # face raw
#             if r.face_embeddings_raw and (max_bank <= 0 or raw_loaded_face[member_id] < max_bank):
#                 bank = decode_bank_blob(r.face_embeddings_raw)
#                 if bank is not None and bank.size > 0:
#                     remain = max_bank - raw_loaded_face[member_id] if max_bank > 0 else bank.shape[0]
#                     take = bank[: max(0, remain)] if max_bank > 0 else bank
#                     if take.size > 0:
#                         agg_global[member_id].face_banks.append(take)
#                         raw_loaded_face[member_id] += int(take.shape[0])

#                     key = (member_id, cam_id)
#                     if max_bank <= 0 or raw_loaded_face_cam[key] < max_bank:
#                         remain2 = max_bank - raw_loaded_face_cam[key] if max_bank > 0 else bank.shape[0]
#                         take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
#                         if take2.size > 0:
#                             agg_by_cam[cam_id][member_id].face_banks.append(take2)
#                             raw_loaded_face_cam[key] += int(take2.shape[0])

#             # optional back-body raw
#             if r.back_body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[member_id] < max_bank):
#                 bank = decode_bank_blob(r.back_body_embeddings_raw)
#                 if bank is not None and bank.size > 0:
#                     remain = max_bank - raw_loaded_body[member_id] if max_bank > 0 else bank.shape[0]
#                     take = bank[: max(0, remain)] if max_bank > 0 else bank
#                     if take.size > 0:
#                         agg_global[member_id].body_banks.append(take)
#                         raw_loaded_body[member_id] += int(take.shape[0])

#                     key = (member_id, cam_id)
#                     if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
#                         remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
#                         take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
#                         if take2.size > 0:
#                             agg_by_cam[cam_id][member_id].body_banks.append(take2)
#                             raw_loaded_body_cam[key] += int(take2.shape[0])

#     def _build_people_from_agg(agg: dict[int, _Agg]) -> tuple[list[PersonEntry], FaceGallery]:
#         people: list[PersonEntry] = []
#         face_names: list[str] = []
#         face_vecs: list[np.ndarray] = []

#         for member_id, a in agg.items():
#             meta = member_meta.get(member_id, {})
#             name = _member_label(
#                 meta.get("member_number"),
#                 meta.get("first_name"),
#                 meta.get("last_name"),
#                 member_id,
#             )

#             # body bank
#             bank_parts = []
#             for bnk in a.body_banks:
#                 if bnk is not None and bnk.size > 0:
#                     bank_parts.append(np.asarray(bnk, dtype=np.float32))
#             body_bank = np.concatenate(bank_parts, axis=0) if bank_parts else None
#             if body_bank is not None:
#                 body_bank = l2_normalize_rows(body_bank)

#             # body centroid
#             body_cent = None
#             if a.body_vecs:
#                 body_cent = l2_normalize(np.mean(np.stack(a.body_vecs, axis=0), axis=0))
#             elif body_bank is not None and body_bank.size > 0:
#                 body_cent = l2_normalize(np.mean(body_bank, axis=0))

#             # face centroid
#             face_cent = None
#             if a.face_vecs:
#                 face_cent = l2_normalize(np.mean(np.stack(a.face_vecs, axis=0), axis=0))
#             else:
#                 # try face bank(s)
#                 fbank_parts = []
#                 for bnk in a.face_banks:
#                     if bnk is not None and bnk.size > 0:
#                         fbank_parts.append(np.asarray(bnk, dtype=np.float32))
#                 if fbank_parts:
#                     fb = l2_normalize_rows(np.concatenate(fbank_parts, axis=0))
#                     face_cent = l2_normalize(np.mean(fb, axis=0))

#             # if body_bank missing but centroid exists, create 1-sample bank for matching
#             if body_bank is None and body_cent is not None:
#                 body_bank = body_cent.reshape(1, -1).astype(np.float32)

#             people.append(
#                 PersonEntry(
#                     user_id=int(member_id),
#                     name=str(name),
#                     body_bank=body_bank,
#                     body_centroid=body_cent,
#                     face_centroid=face_cent,
#                 )
#             )

#             if face_cent is not None:
#                 face_names.append(str(name))
#                 face_vecs.append(face_cent.astype(np.float32))

#         face_mat = (
#             l2_normalize_rows(np.stack(face_vecs, axis=0))
#             if face_vecs
#             else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
#         )
#         return people, FaceGallery(face_names, face_mat)

#     gd = GalleryData()

#     # Build global
#     gd.people_global, gd.face_global = _build_people_from_agg(agg_global)

#     # Build per-camera
#     for cam_id, member_map in agg_by_cam.items():
#         ppl, fg = _build_people_from_agg(member_map)
#         gd.people_by_cam[int(cam_id)] = ppl
#         gd.face_by_cam[int(cam_id)] = fg

#     body_count = sum(1 for p in gd.people_global if p.body_bank is not None and p.body_bank.size > 0)
#     print(f"[DB] Loaded identities (global): {len(gd.people_global)} (body={body_count}, face={len(gd.face_global.names)})")

#     return gd


# class GalleryManager:
#     """Thread-safe DB gallery with optional periodic refresh."""

#     def __init__(self, args):
#         if not getattr(args, "db_url", ""):
#             raise RuntimeError("db_url is required")
#         self._lock = threading.Lock()
#         self._reload_lock = threading.Lock()
#         self.data: GalleryData = GalleryData()
#         self.last_load_ts: float = 0.0
#         self.load(args)

#     def load(self, args) -> None:
#         active_only = not bool(getattr(args, "db_include_inactive", False))
#         max_bank = int(getattr(args, "db_max_bank", 0) or 0)

#         data = build_galleries_from_db(args.db_url, active_only=active_only, max_bank_per_member=max_bank)

#         with self._lock:
#             self.data = data
#             self.last_load_ts = time.time()

#     def maybe_reload(self, args) -> None:
#         period = float(getattr(args, "db_refresh_seconds", 0.0) or 0.0)
#         if period <= 0:
#             return
#         now = time.time()
#         if (now - self.last_load_ts) < period:
#             return
#         if not self._reload_lock.acquire(blocking=False):
#             return
#         try:
#             if (time.time() - self.last_load_ts) < period:
#                 return
#             try:
#                 self.load(args)
#                 print("[DB] Gallery reloaded")
#             except Exception as e:
#                 print("[DB] reload failed:", e)
#         finally:
#             self._reload_lock.release()

#     def snapshot(self, camera_id: Optional[int] = None) -> tuple[list[PersonEntry], FaceGallery]:
#         """
#         Returns per-camera gallery when available; otherwise falls back to global.
#         """
#         with self._lock:
#             data = self.data
#             if camera_id is None:
#                 return data.people_global, data.face_global

#             cam_id = int(camera_id)
#             ppl = data.people_by_cam.get(cam_id)
#             fg = data.face_by_cam.get(cam_id)
#             if ppl is not None and fg is not None and (not fg.is_empty()):
#                 return ppl, fg

#             # fallback
#             return data.people_global, data.face_global


# # ----------------------------
# # Tracking helpers (IoU)
# # ----------------------------


# def iou_xyxy(a, b) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
#     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     denom = a_area + b_area - inter
#     return float(inter / denom) if denom > 0 else 0.0


# def ioa_xyxy(inner, outer) -> float:
#     """Intersection over AREA(inner). Good for linking small face box to big person box."""
#     ix1, iy1, ix2, iy2 = inner
#     ox1, oy1, ox2, oy2 = outer
#     inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
#     inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     inner_area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
#     return float(inter / inner_area) if inner_area > 0 else 0.0


# def _point_in_xyxy(px: float, py: float, box) -> bool:
#     x1, y1, x2, y2 = box
#     return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


# class IOUTrack:
#     def __init__(self, tlwh, tid):
#         self.tlwh = np.array(tlwh, dtype=np.float32)
#         self.tid = int(tid)
#         self.miss = 0


# class IOUTracker:
#     """Minimal IoU tracker fallback (stable IDs)."""

#     def __init__(self, max_miss=5, iou_thresh=0.3):
#         self.tracks: list[IOUTrack] = []
#         self.next_id = 1
#         self.max_miss = int(max_miss)
#         self.iou_thresh = float(iou_thresh)

#     def update(self, dets_tlwh_conf: np.ndarray):
#         dets = np.asarray(dets_tlwh_conf, dtype=np.float32)
#         if dets.ndim != 2 or dets.shape[1] < 4:
#             dets = dets.reshape((0, 5)).astype(np.float32)

#         assigned = set()
#         for tr in self.tracks:
#             tr.miss += 1
#             t_x, t_y, t_w, t_h = tr.tlwh
#             t_xyxy = np.array([t_x, t_y, t_x + t_w, t_y + t_h], dtype=np.float32)
#             best_j, best_iou = -1, 0.0
#             for j, d in enumerate(dets):
#                 if j in assigned:
#                     continue
#                 x, y, w, h = d[:4]
#                 d_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
#                 s = iou_xyxy(t_xyxy, d_xyxy)
#                 if s > best_iou:
#                     best_iou, best_j = s, j
#             if best_j >= 0 and best_iou >= self.iou_thresh:
#                 tr.tlwh = dets[best_j][:4]
#                 tr.miss = 0
#                 assigned.add(best_j)

#         for j, d in enumerate(dets):
#             if j in assigned:
#                 continue
#             self.tracks.append(IOUTrack(d[:4], self.next_id))
#             self.next_id += 1

#         self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

#         outs = []
#         for t in self.tracks:
#             x, y, w, h = t.tlwh

#             class Dummy:
#                 pass

#             o = Dummy()
#             o.track_id = t.tid
#             o.is_confirmed = lambda: True
#             o.to_tlbr = lambda: (x, y, x + w, y + h)
#             o.det_conf = None
#             o.last_detection = None
#             o.time_since_update = 0
#             outs.append(o)

#         return outs


# # ----------------------------
# # Identity smoothing (anti-flicker)
# # ----------------------------


# def update_track_identity(
#     state: dict,
#     tid: int,
#     candidates: list[tuple[str, float, str]],
#     decay: float,
#     min_score: float,
#     margin: float,
#     ttl_reset: int,
#     w_face: float,
#     w_body: float,
# ) -> tuple[str, float, dict]:
#     """
#     Exponentially decayed per-name score accumulator.

#     NOTE: Body votes are only accepted when they agree with the face label (see caller).
#     """
#     entry = state.setdefault(
#         tid,
#         {
#             "scores": defaultdict(float),
#             "last": "",
#             "ttl": 0,
#             "face_vis_ttl": 0,
#             "last_face_label": "",
#             "last_face_sim": 0.0,
#         },
#     )
#     scores = entry["scores"]

#     for k in list(scores.keys()):
#         scores[k] *= float(decay)
#         if scores[k] < 1e-6:
#             del scores[k]

#     for label, sim, src in candidates:
#         if not label:
#             continue
#         w = w_face if src == "face" else w_body
#         scores[label] += max(0.0, float(sim)) * float(w)

#     if scores:
#         ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#         top_label, top_score = ranked[0]
#         second_score = ranked[1][1] if len(ranked) > 1 else 0.0
#     else:
#         top_label, top_score, second_score = "", 0.0, 0.0

#     if top_label and (top_score >= min_score) and (entry["last"] == top_label or (top_score - second_score) >= margin):
#         entry["last"] = top_label
#         entry["ttl"] = int(ttl_reset)
#     else:
#         if entry["ttl"] > 0:
#             entry["ttl"] -= 1
#         else:
#             entry["last"] = ""

#     return entry["last"], float(scores.get(entry["last"], 0.0)), entry


# # ----------------------------
# # Models init
# # ----------------------------


# def init_face_engine(
#     use_face: bool,
#     device: str,
#     face_model: str,
#     det_w: int,
#     det_h: int,
#     face_provider: str,
#     ort_log: bool,
# ):
#     if not use_face:
#         return None
#     if not INSIGHT_OK:
#         print("[WARN] insightface not installed; face recognition disabled.")
#         return None
#     try:
#         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
#         cuda_ok = _cuda_ep_loadable()

#         if ort is not None and ort_log:
#             try:
#                 print(f"[INFO] ORT available providers: {ort.get_available_providers()}")
#             except Exception:
#                 pass

#         providers = ["CPUExecutionProvider"]
#         if face_provider == "cuda":
#             if cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#             else:
#                 print("[INFO] Requested CUDA EP, but not loadable. Using CPU.")
#         elif face_provider == "auto":
#             if is_cuda and cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

#         app = FaceAnalysis(name=face_model, providers=providers)
#         ctx_id = 0 if providers[0].startswith("CUDA") else -1
#         try:
#             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
#         except TypeError:
#             app.prepare(ctx_id=ctx_id)
#         print(f"[INIT] InsightFace ready (model={face_model}, providers={providers}).")
#         return app
#     except Exception as e:
#         print("[WARN] InsightFace init failed:", e)
#         return None


# def _yolo_forward_safe(yolo, frame, args):
#     """
#     Guarded YOLO call with a per-model lock and FP16->FP32 fallback.
#     """
#     with _yolo_lock, torch.inference_mode():
#         try:
#             return yolo(
#                 frame,
#                 conf=args.conf,
#                 iou=args.iou,
#                 verbose=False,
#                 device=args.device,
#                 half=args.half,
#                 imgsz=int(args.yolo_imgsz) if int(args.yolo_imgsz) > 0 else None,
#             )
#         except TypeError:
#             return yolo(
#                 frame,
#                 conf=args.conf,
#                 iou=args.iou,
#                 verbose=False,
#                 device=args.device,
#                 half=args.half,
#             )
#         except Exception as e:
#             if args.half:
#                 print("[YOLO] FP16 failed, retrying in FP32 once:", e)
#                 args.half = False
#                 return yolo(
#                     frame,
#                     conf=args.conf,
#                     iou=args.iou,
#                     verbose=False,
#                     device=args.device,
#                     half=False,
#                 )
#             raise


# def extract_body_embeddings_batch(
#     extractor,
#     crops_bgr: List[np.ndarray],
#     device_is_cuda: bool,
#     use_half: bool,
# ) -> Optional[np.ndarray]:
#     """
#     Batched TorchReID embedding extraction for multiple crops in one call.
#     Returns: np.ndarray [N,512] normalized, or None.
#     """
#     if extractor is None or not crops_bgr:
#         return None

#     crops_rgb: List[np.ndarray] = []
#     for c in crops_bgr:
#         if c is None or c.size == 0:
#             crops_rgb.append(np.zeros((1, 1, 3), dtype=np.uint8))
#             continue
#         crops_rgb.append(_to_rgb(c))

#     with _reid_lock, torch.inference_mode():
#         if device_is_cuda and use_half:
#             try:
#                 with torch.autocast(device_type="cuda", dtype=torch.float16):
#                     feats = extractor(crops_rgb)
#             except Exception:
#                 feats = extractor(crops_rgb)
#         else:
#             feats = extractor(crops_rgb)

#     try:
#         if isinstance(feats, (list, tuple)):
#             feats_arr = []
#             for f in feats:
#                 f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                 feats_arr.append(np.asarray(f, dtype=np.float32).reshape(-1))
#             mat = np.stack(feats_arr, axis=0)
#         else:
#             f = feats.detach().cpu().numpy() if hasattr(feats, "detach") else np.asarray(feats)
#             mat = np.asarray(f, dtype=np.float32)
#             if mat.ndim == 1:
#                 mat = mat.reshape(1, -1)
#         if mat.ndim != 2 or mat.shape[1] != EXPECTED_DIM:
#             return None
#         if not np.isfinite(mat).all():
#             return None
#         return l2_normalize_rows(mat)
#     except Exception:
#         return None


# def best_body_label_from_emb(
#     emb: np.ndarray | None,
#     people: list[PersonEntry],
#     topk: int = 3,
# ) -> tuple[str | None, float, float]:
#     """TopK-mean cosine match across each person's body_bank."""
#     if emb is None:
#         return None, 0.0, 0.0
#     q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
#     if q.size != EXPECTED_DIM or not np.isfinite(q).all():
#         return None, 0.0, 0.0

#     k_req = max(1, int(topk))
#     scored: list[tuple[str, float]] = []

#     for p in people:
#         bank = p.body_bank
#         if bank is None or bank.size == 0:
#             continue
#         try:
#             sims = bank @ q
#         except Exception:
#             continue
#         if sims.ndim != 1 or sims.size == 0:
#             continue
#         k = min(k_req, sims.size)
#         if k <= 1:
#             score = float(np.max(sims))
#         else:
#             top_vals = np.partition(sims, -k)[-k:]
#             score = float(np.mean(top_vals))
#         scored.append((p.name, score))

#     if not scored:
#         return None, 0.0, 0.0

#     scored.sort(key=lambda x: x[1], reverse=True)
#     best_label, best_score = scored[0]
#     second_score = scored[1][1] if len(scored) > 1 else 0.0
#     return best_label, float(best_score), float(second_score)


# def best_face_label_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> tuple[str | None, float, float]:
#     """
#     Returns: (best_label, best_sim, second_sim)
#     """
#     if emb is None or face_gallery is None or face_gallery.is_empty():
#         return None, 0.0, 0.0
#     q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
#     if q.size != EXPECTED_DIM or not np.isfinite(q).all():
#         return None, 0.0, 0.0

#     sims = face_gallery.mat @ q
#     if sims.size == 0:
#         return None, 0.0, 0.0

#     if sims.size == 1:
#         return face_gallery.names[0], float(sims[0]), 0.0

#     idxs = np.argpartition(sims, -2)[-2:]
#     i1, i2 = int(idxs[0]), int(idxs[1])
#     if sims[i2] > sims[i1]:
#         i1, i2 = i2, i1
#     best_idx, second_idx = i1, i2
#     return face_gallery.names[best_idx], float(sims[best_idx]), float(sims[second_idx])


# # ----------------------------
# # Threaded readers
# # ----------------------------


# class AdaptiveQueueStream:
#     """
#     Ordered reader with a bounded queue.
#     - When queue is full, drops the oldest frame to make room (keeps latency bounded).
#     """

#     def __init__(self, src: str, queue_size: int, rtsp_transport: str, use_opencv: bool = True):
#         self.src = src
#         src_use = src
#         if isinstance(src, str) and src.lower().startswith("rtsp"):
#             sep = "&" if "?" in src_use else "?"
#             src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"

#         self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG) if use_opencv else cv2.VideoCapture(src_use)
#         self.ok = self.cap.isOpened()
#         if not self.ok:
#             print(f"[WARN] cannot open source {src}")
#         try:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(queue_size)))
#         except Exception:
#             pass

#         self.q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
#         self.stop_flag = threading.Event()
#         self.dropped = 0
#         self.read_dropped = 0
#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()

#     def _drop_oldest(self) -> None:
#         try:
#             _ = self.q.get_nowait()
#             self.dropped += 1
#         except queue.Empty:
#             return

#     def _loop(self):
#         while not self.stop_flag.is_set() and self.ok:
#             ok, frame = self.cap.read()
#             if not ok or frame is None:
#                 time.sleep(0.005)
#                 continue
#             item = (frame, time.time())
#             try:
#                 self.q.put_nowait(item)
#             except queue.Full:
#                 self._drop_oldest()
#                 try:
#                     self.q.put_nowait(item)
#                 except queue.Full:
#                     self._drop_oldest()

#     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
#         try:
#             frame, ts = self.q.get(timeout=0.1)
#             return True, frame, float(ts)
#         except queue.Empty:
#             return False, None, 0.0

#     def qsize(self) -> int:
#         try:
#             return int(self.q.qsize())
#         except Exception:
#             return 0

#     def is_opened(self) -> bool:
#         return bool(self.ok)

#     def release(self):
#         self.stop_flag.set()
#         try:
#             self.thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.cap.release()
#         except Exception:
#             pass


# class LatestStream:
#     """
#     Latest-frame reader (keeps only 1 frame).
#     """

#     def __init__(self, src: str, rtsp_buffer: int, decode_skip: int, rtsp_transport: str):
#         self.src = src
#         src_use = src
#         if isinstance(src, str) and src.lower().startswith("rtsp"):
#             sep = "&" if "?" in src_use else "?"
#             src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"
#         self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG)
#         self.ok = self.cap.isOpened()
#         if not self.ok:
#             print(f"[WARN] cannot open source {src}")
#         try:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(rtsp_buffer)))
#         except Exception:
#             pass
#         self.decode_skip = max(0, int(decode_skip))
#         self.q = queue.Queue(maxsize=1)
#         self.stop_flag = threading.Event()
#         self.dropped = 0
#         self.read_dropped = 0
#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()

#     def _loop(self):
#         idx = 0
#         while not self.stop_flag.is_set() and self.ok:
#             ok, frame = self.cap.read()
#             if not ok or frame is None:
#                 time.sleep(0.01)
#                 continue
#             if self.decode_skip > 0:
#                 if (idx % (self.decode_skip + 1)) != 0:
#                     idx += 1
#                     continue
#                 idx += 1
#             item = (frame, time.time())
#             if not self.q.empty():
#                 try:
#                     _ = self.q.get_nowait()
#                     self.dropped += 1
#                 except queue.Empty:
#                     pass
#             try:
#                 self.q.put_nowait(item)
#             except queue.Full:
#                 self.dropped += 1

#     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
#         try:
#             frame, ts = self.q.get(timeout=0.02)
#             return True, frame, float(ts)
#         except queue.Empty:
#             return False, None, 0.0

#     def qsize(self) -> int:
#         return 1 if not self.q.empty() else 0

#     def is_opened(self) -> bool:
#         return bool(self.ok)

#     def release(self):
#         self.stop_flag.set()
#         try:
#             self.thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.cap.release()
#         except Exception:
#             pass


# class FFmpegStream:
#     """FFmpeg pipe reader (ordered). NVDEC optional."""

#     def __init__(
#         self,
#         src: str,
#         queue_size: int,
#         rtsp_transport: str,
#         use_cuda: bool,
#         force_w: int = 0,
#         force_h: int = 0,
#     ):
#         self.src = _sanitize_rtsp_url(src) if isinstance(src, str) and src.lower().startswith("rtsp") else src
#         self.q = queue.Queue(maxsize=max(1, int(queue_size)))
#         self.stop_flag = threading.Event()
#         self.proc = None
#         self.ok = False
#         self.dropped = 0
#         self.read_dropped = 0

#         probed_w = probed_h = 0
#         try:
#             cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
#             if cap.isOpened():
#                 probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
#                 probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
#             cap.release()
#         except Exception:
#             pass

#         scheme = ""
#         try:
#             scheme = urlsplit(self.src).scheme.lower()
#         except Exception:
#             pass

#         cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", self.src, "-an", "-dn", "-sn", "-vsync", "0"]
#         if scheme == "rtsp":
#             cmd = [
#                 "ffmpeg",
#                 "-rtsp_transport",
#                 rtsp_transport,
#                 "-flags",
#                 "+genpts",
#                 "-fflags",
#                 "+genpts",
#                 "-use_wallclock_as_timestamps",
#                 "1",
#                 "-i",
#                 self.src,
#                 "-an",
#                 "-dn",
#                 "-sn",
#                 "-vsync",
#                 "0",
#             ]
#         cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]

#         out_w = int(force_w) if force_w and force_w > 0 else int(probed_w)
#         out_h = int(force_h) if force_h and force_h > 0 else int(probed_h)
#         if out_w > 0 and out_h > 0:
#             cmd += ["-vf", f"scale={out_w}:{out_h}"]
#         else:
#             out_w, out_h = 1280, 720
#             cmd += ["-vf", "scale=1280:720"]

#         if use_cuda:
#             hw = [
#                 "-hwaccel",
#                 "cuda",
#                 "-hwaccel_output_format",
#                 "cuda",
#                 "-vf",
#                 f"hwdownload,format=bgr24,scale={out_w}:{out_h}",
#             ]
#             if "-vf" in cmd:
#                 i = cmd.index("-vf")
#                 cmd.pop(i)
#                 prev = cmd.pop(i)
#                 hw[-1] = f"hwdownload,format=bgr24,{prev},scale={out_w}:{out_h}" if "scale=" in prev else hw[-1]
#             cmd += hw

#         cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

#         self.cmd = cmd
#         self.width, self.height = int(out_w), int(out_h)
#         self.frame_bytes = self.width * self.height * 3

#         try:
#             self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
#             self.ok = True
#         except Exception as e:
#             print(f"[FFMPEG] start failed: {e}")
#             self.ok = False

#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()
#         self.log_thread = threading.Thread(target=self._log_stderr, daemon=True)
#         self.log_thread.start()

#     def _log_stderr(self):
#         if not self.proc or not self.proc.stderr:
#             return
#         try:
#             for _line in iter(self.proc.stderr.readline, b""):
#                 if not _line:
#                     break
#         except Exception:
#             pass

#     def _drop_oldest(self) -> None:
#         try:
#             _ = self.q.get_nowait()
#             self.dropped += 1
#         except queue.Empty:
#             return

#     def _loop(self):
#         if not self.proc or not self.proc.stdout or not self.ok:
#             return
#         fb = int(self.frame_bytes)
#         w, h = int(self.width), int(self.height)
#         while not self.stop_flag.is_set():
#             buf = self.proc.stdout.read(fb)
#             if not buf or len(buf) < fb:
#                 time.sleep(0.001)
#                 continue
#             frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
#             item = (frame, time.time())
#             try:
#                 self.q.put_nowait(item)
#             except queue.Full:
#                 self._drop_oldest()
#                 try:
#                     self.q.put_nowait(item)
#                 except queue.Full:
#                     self._drop_oldest()

#     def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
#         try:
#             frame, ts = self.q.get(timeout=0.1)
#             return True, frame, float(ts)
#         except queue.Empty:
#             return False, None, 0.0

#     def qsize(self) -> int:
#         try:
#             return int(self.q.qsize())
#         except Exception:
#             return 0

#     def is_opened(self) -> bool:
#         return bool(self.ok)

#     def release(self):
#         self.stop_flag.set()
#         try:
#             self.thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             if self.proc:
#                 self.proc.terminate()
#                 self.proc.kill()
#         except Exception:
#             pass


# # ----------------------------
# # Frame buffer for MJPEG (single latest frame + cached JPEG)
# # ----------------------------


# class FrameBuffer:
#     """
#     Stores latest annotated frame + metadata.

#     ✅ Compatibility with your mjpeg.py:
#       - wait_for_seq(last_seq, timeout) -> (frame, ts, meta, seq)

#     Also supports cached JPEG for MJPEG if you want to use wait_jpeg().
#     """

#     def __init__(self):
#         self._lock = threading.Lock()
#         self._cond = threading.Condition(self._lock)

#         self._frame_bgr: Optional[np.ndarray] = None
#         self._ts: float = 0.0
#         self._meta: Dict[str, Any] = {}

#         # ✅ NEW: sequence counter (increments per set)
#         self._seq: int = 0

#         # Cached JPEG
#         self._jpeg: Optional[bytes] = None
#         self._jpeg_ts: float = 0.0
#         self._encode_lock = threading.Lock()

#         # Optional (for load control)
#         self._clients = 0

#     def add_client(self) -> None:
#         with self._lock:
#             self._clients += 1

#     def remove_client(self) -> None:
#         with self._lock:
#             self._clients = max(0, self._clients - 1)

#     def set(self, frame_bgr: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
#         """
#         Called by processor thread. Updates frame + meta and bumps sequence.
#         """
#         with self._cond:
#             self._frame_bgr = frame_bgr
#             self._ts = time.time()
#             self._meta = dict(meta or {})

#             # ✅ bump sequence so mjpeg can wait for new frames
#             self._seq += 1

#             # invalidate jpeg cache
#             self._jpeg = None
#             self._jpeg_ts = 0.0

#             self._cond.notify_all()

#     def get(self) -> Optional[np.ndarray]:
#         """Compatibility helper: returns only the frame."""
#         with self._lock:
#             return self._frame_bgr

#     def get_meta(self) -> Dict[str, Any]:
#         with self._lock:
#             return dict(self._meta)

#     def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
#         """Used by multi-cam views sometimes."""
#         with self._lock:
#             return self._frame_bgr, float(self._ts)

#     def get_seq(self) -> int:
#         with self._lock:
#             return int(self._seq)

#     # THIS IS WHAT YOUR mjpeg.py NEEDS
#     def wait_for_seq(
#         self,
#         last_seq: int,
#         timeout: float = 1.0,
#     ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
#         """
#         Wait until seq changes from last_seq OR timeout.

#         Returns: (frame, ts, meta, seq)
#         """
#         try:
#             last_seq_i = int(last_seq)
#         except Exception:
#             last_seq_i = -1

#         with self._cond:
#             if self._seq == last_seq_i:
#                 self._cond.wait(timeout=float(timeout))

#             # Return latest snapshot (even if timeout)
#             frame = self._frame_bgr
#             ts = float(self._ts)
#             meta = dict(self._meta)
#             seq = int(self._seq)

#         return frame, ts, meta, seq

#     def wait_jpeg(
#         self,
#         last_ts: float,
#         timeout: float,
#         jpeg_quality: int = 80,
#     ) -> Tuple[Optional[bytes], float]:
#         """
#         Optional: wait for a newer frame than last_ts and return JPEG bytes.
#         """
#         with self._cond:
#             if self._ts <= float(last_ts):
#                 self._cond.wait(timeout=float(timeout))

#             ts = float(self._ts)
#             frame = self._frame_bgr
#             clients = int(self._clients)

#         if frame is None or ts <= 0:
#             return None, 0.0

#         # If nobody is watching MJPEG, skip encoding
#         if clients <= 0:
#             return None, ts

#         # Reuse cached JPEG if still valid
#         with self._lock:
#             if self._jpeg is not None and self._jpeg_ts == ts:
#                 return self._jpeg, ts

#         with self._encode_lock:
#             with self._lock:
#                 if self._jpeg is not None and self._jpeg_ts == ts:
#                     return self._jpeg, ts
#                 frame_ref = self._frame_bgr
#                 ts_copy = float(self._ts)

#             if frame_ref is None or ts_copy <= 0:
#                 return None, 0.0

#             q = int(max(30, min(95, int(jpeg_quality))))
#             ok, enc = cv2.imencode(".jpg", frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#             if not ok:
#                 return None, ts_copy

#             jpg = enc.tobytes()

#             with self._lock:
#                 if float(self._ts) == ts_copy:
#                     self._jpeg = jpg
#                     self._jpeg_ts = ts_copy

#             return jpg, ts_copy


# # Compatibility alias (some code imports RenderedFrame)
# RenderedFrame = FrameBuffer


# # ----------------------------
# # CSV SUMMARY REPORT (per person per camera logs)
# # ----------------------------


# @dataclass
# class _ActiveSession:
#     start_ts: float
#     last_seen_ts: float


# class SummaryReport:
#     """
#     Builds a summary CSV:
#       Member | c1 | c2 | ... | Total time

#     Each camera cell contains multiple lines:
#       L1 - HH:MM:SS to HH:MM:SS
#       L2 - HH:MM:SS to HH:MM:SS

#     IMPORTANT: One row per person name (never creates duplicates later).
#     """

#     def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
#         self.num_cams = int(max(1, num_cams))
#         self.gap_seconds = float(max(0.0, gap_seconds))
#         self.time_format = str(time_format or "%H:%M:%S")
#         self._lock = threading.Lock()
#         self._disabled = False

#         self._first_seen: Dict[str, float] = {}
#         self._active: Dict[Tuple[str, int], _ActiveSession] = {}
#         self._logs: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
#         self._total_seconds: Dict[str, float] = defaultdict(float)

#     def stop(self) -> None:
#         with self._lock:
#             self._disabled = True

#     def update(self, cam_id: int, present_names: List[str], ts: float) -> None:
#         if ts <= 0:
#             ts = time.time()
#         cam_id = int(cam_id)

#         with self._lock:
#             if self._disabled:
#                 return

#             # close expired first
#             self._close_expired_locked(now_ts=float(ts))

#             for nm in present_names or []:
#                 name = str(nm).strip()
#                 if not name:
#                     continue

#                 if name not in self._first_seen:
#                     self._first_seen[name] = float(ts)

#                 key = (name, cam_id)
#                 sess = self._active.get(key)
#                 if sess is None:
#                     self._active[key] = _ActiveSession(start_ts=float(ts), last_seen_ts=float(ts))
#                 else:
#                     sess.last_seen_ts = float(ts)

#     def _close_expired_locked(self, now_ts: float) -> None:
#         if self.gap_seconds <= 0:
#             return
#         to_close: List[Tuple[str, int, _ActiveSession]] = []
#         for (name, cam_id), sess in list(self._active.items()):
#             if (float(now_ts) - float(sess.last_seen_ts)) > self.gap_seconds:
#                 to_close.append((name, cam_id, sess))

#         for name, cam_id, sess in to_close:
#             self._close_session_locked(name, cam_id, sess)

#     def _close_session_locked(self, name: str, cam_id: int, sess: _ActiveSession) -> None:
#         st = float(sess.start_ts)
#         en = float(sess.last_seen_ts)
#         if en < st:
#             en = st

#         self._logs[name][int(cam_id)].append((st, en))
#         self._total_seconds[name] += max(0.0, (en - st))
#         self._active.pop((name, int(cam_id)), None)

#     def close_all(self) -> None:
#         with self._lock:
#             for (name, cam_id), sess in list(self._active.items()):
#                 self._close_session_locked(name, cam_id, sess)

#     def _fmt_time(self, ts: float) -> str:
#         try:
#             return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
#         except Exception:
#             return ""

#     def _fmt_total(self, total_seconds: float) -> str:
#         try:
#             sec_f = float(total_seconds)
#         except Exception:
#             sec_f = 0.0

#         sec_i = int(round(max(0.0, sec_f)))
#         if sec_i == 0 and sec_f > 0.0:
#             sec_i = 1

#         minutes = int(sec_i // 60)
#         seconds = int(sec_i % 60)

#         m_word = "minute" if minutes == 1 else "minutes"
#         s_word = "second" if seconds == 1 else "seconds"
#         return f"{minutes} {m_word} {seconds} {s_word}"

#     def write_csv(self, path: str) -> None:
#         """Final write: closes all active sessions first."""
#         self.close_all()

#         with self._lock:
#             items = list(self._first_seen.items())
#             logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
#             totals = dict(self._total_seconds)

#         items.sort(key=lambda kv: kv[1])
#         names_order = [nm for nm, _ in items]

#         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

#         header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
#         with open(path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(header)

#             for name in names_order:
#                 row = [name]
#                 cam_map = logs.get(name, {})

#                 for cam_id in range(self.num_cams):
#                     entries = cam_map.get(cam_id, [])
#                     lines = []
#                     for idx, (st, en) in enumerate(entries, start=1):
#                         lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
#                     row.append("\n".join(lines))

#                 row.append(self._fmt_total(totals.get(name, 0.0)))
#                 w.writerow(row)

#     def write_csv_snapshot(self, path: str) -> None:
#         """
#         Snapshot write: does NOT close active sessions permanently.
#         Adds active segments as temporary logs for the snapshot.
#         """
#         now_ts = time.time()

#         with self._lock:
#             if self._disabled:
#                 return

#             self._close_expired_locked(now_ts=now_ts)

#             items = list(self._first_seen.items())
#             logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
#             totals = dict(self._total_seconds)

#             # include active segments temporarily
#             for (name, cam_id), sess in self._active.items():
#                 st = float(sess.start_ts)
#                 en = float(sess.last_seen_ts)
#                 logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en))
#                 totals[name] = totals.get(name, 0.0) + max(0.0, (en - st))

#         items.sort(key=lambda kv: kv[1])
#         names_order = [nm for nm, _ in items]

#         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

#         header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
#         with open(path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(header)

#             for name in names_order:
#                 row = [name]
#                 cam_map = logs.get(name, {})

#                 for cam_id in range(self.num_cams):
#                     entries = cam_map.get(cam_id, [])
#                     lines = []
#                     for idx, (st, en) in enumerate(entries, start=1):
#                         lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
#                     row.append("\n".join(lines))

#                 row.append(self._fmt_total(totals.get(name, 0.0)))
#                 w.writerow(row)


# # ----------------------------
# # Name de-duplication + global multi-camera ownership
# # ----------------------------


# @dataclass
# class DrawItem:
#     tid: int
#     bbox: Tuple[int, int, int, int]
#     name: str
#     user_id: int
#     face_sim: float
#     stable_score: float
#     det_conf: Optional[float]
#     face_hit: bool


# def _box_area_xyxy(b: Tuple[int, int, int, int]) -> int:
#     x1, y1, x2, y2 = b
#     return int(max(0, x2 - x1) * max(0, y2 - y1))


# def _priority_tuple(it: DrawItem) -> tuple:
#     return (
#         1 if it.face_hit else 0,
#         float(it.face_sim),
#         float(it.stable_score),
#         float(it.det_conf or 0.0),
#         float(_box_area_xyxy(it.bbox)),
#     )


# def deduplicate_draw_items(items: List[DrawItem], iou_thresh: float) -> List[DrawItem]:
#     if not items:
#         return []

#     groups: Dict[str, List[DrawItem]] = defaultdict(list)
#     for it in items:
#         if it.name:
#             groups[it.name].append(it)

#     kept: List[DrawItem] = []
#     for _name, group in groups.items():
#         group_sorted = sorted(group, key=_priority_tuple, reverse=True)

#         if float(iou_thresh) <= 0.0:
#             kept.append(group_sorted[0])
#             continue

#         selected: List[DrawItem] = []
#         for it in group_sorted:
#             ok = True
#             for s in selected:
#                 if iou_xyxy(it.bbox, s.bbox) >= float(iou_thresh):
#                     ok = False
#                     break
#             if ok:
#                 selected.append(it)
#         kept.extend(selected)

#     return kept


# class GlobalNameOwner:
#     def __init__(self, hold_seconds: float = 0.5, switch_margin: float = 0.02):
#         self.hold_seconds = float(max(0.0, hold_seconds))
#         self.switch_margin = float(max(0.0, switch_margin))
#         self._lock = threading.Lock()
#         self._state: Dict[str, Dict[str, Any]] = {}

#     def _cleanup(self, now: float) -> None:
#         if self.hold_seconds <= 0:
#             return
#         dead = []
#         for name, st in self._state.items():
#             ts = float(st.get("ts", 0.0))
#             if (now - ts) > self.hold_seconds:
#                 dead.append(name)
#         for name in dead:
#             self._state.pop(name, None)

#     def allow(self, name: str, sid: int, score: float) -> bool:
#         if not name:
#             return False
#         now = time.time()
#         with self._lock:
#             self._cleanup(now)

#             st = self._state.get(name)
#             if st is None:
#                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
#                 return True

#             owner_sid = int(st.get("sid", -1))
#             owner_score = float(st.get("score", 0.0))
#             owner_ts = float(st.get("ts", 0.0))

#             if owner_sid == int(sid):
#                 st["score"] = max(owner_score, float(score))
#                 st["ts"] = now
#                 return True

#             if self.hold_seconds > 0 and (now - owner_ts) > self.hold_seconds:
#                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
#                 return True

#             if float(score) > (owner_score + self.switch_margin):
#                 self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
#                 return True

#             return False


# # ----------------------------
# # CLI / args parsing
# # ----------------------------


# def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
#     ap = argparse.ArgumentParser(
#         "YOLO -> DeepSORT (TorchReID) with DB gallery tagging + Face-only labeling (body supports only).",
#         conflict_handler="resolve",
#     )

#     # Sources:
#     # - Manual mode: pass RTSP urls (cam_id = 0..N-1)
#     # - DB mode: pass camera IDs (numbers) OR omit entirely (cam_id = DB camera.id, started on-demand)
#     ap.add_argument("--src", nargs="*", default=[], help="Video sources OR DB camera IDs.")

#     # DB
#     ap.add_argument("--use-db", action="store_true", help="Enable DB gallery.")
#     ap.add_argument("--db-url", default="", help="SQLAlchemy DB URL.")
#     ap.add_argument("--db-refresh-seconds", type=float, default=60.0, help="Reload DB gallery every N seconds (0=off).")
#     ap.add_argument("--db-max-bank", type=int, default=256, help="Max embeddings per member to load (0=all).")
#     ap.add_argument("--db-include-inactive", action="store_true", help="Include inactive members.")

#     # YOLO
#     ap.add_argument("--yolo-weights", default="yolov8n.pt")
#     ap.add_argument("--yolo-imgsz", type=int, default=1280)
#     ap.add_argument("--device", default="cuda:0")
#     ap.add_argument("--conf", type=float, default=0.30)
#     ap.add_argument("--iou", type=float, default=0.40)
#     ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")

#     # Runtime/perf
#     ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
#     ap.add_argument("--rtsp-buffer", type=int, default=2)
#     ap.add_argument("--decode-skip", type=int, default=0)
#     ap.add_argument("--reader", choices=["latest", "adaptive", "ffmpeg"], default="adaptive")
#     ap.add_argument("--queue-size", type=int, default=128)
#     ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
#     ap.add_argument("--ffmpeg-cuda", action="store_true")
#     ap.add_argument("--ffmpeg-width", type=int, default=0)
#     ap.add_argument("--ffmpeg-height", type=int, default=0)
#     ap.add_argument("--resize", type=int, nargs=2, default=[0, 0])

#     # Adaptive skipping
#     ap.add_argument("--max-queue-age-ms", type=int, default=1000)
#     ap.add_argument("--max-drain-per-cycle", type=int, default=32)

#     # DeepSORT / TorchReID
#     g = ap.add_mutually_exclusive_group()
#     g.add_argument("--no-deepsort", action="store_true")
#     g.add_argument("--use-deepsort", action="store_true")

#     ap.add_argument("--reid-model", default="osnet_x0_25")
#     ap.add_argument("--reid-weights", default="")
#     ap.add_argument("--reid-batch-size", type=int, default=16)

#     ap.add_argument("--max-age", type=int, default=15)
#     ap.add_argument("--n-init", type=int, default=3)
#     ap.add_argument("--nn-budget", type=int, default=200)
#     ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
#     ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)

#     # Body matching (support only)
#     ap.add_argument("--gallery-thresh", type=float, default=0.70)
#     ap.add_argument("--gallery-gap", type=float, default=0.08)
#     ap.add_argument("--reid-topk", type=int, default=3)
#     ap.add_argument("--min-box-wh", type=int, default=40)

#     # Face (main identity)
#     ap.add_argument("--use-face", action="store_true")
#     ap.add_argument("--face-model", default="buffalo_l")
#     ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
#     ap.add_argument("--face-thresh", type=float, default=0.40)
#     ap.add_argument("--face-gap", type=float, default=0.05)
#     ap.add_argument("--face-every-n", type=int, default=1)
#     ap.add_argument("--face-hold-frames", type=int, default=2)
#     ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
#     ap.add_argument("--ort-log", action="store_true")

#     ap.add_argument("--face-iou-link", type=float, default=0.35)
#     ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
#     ap.add_argument("--face-center-in-person", action="store_true")

#     # Extra heuristics
#     ap.add_argument("--min-face-px", type=int, default=24)
#     ap.add_argument("--min-face-area-ratio", type=float, default=0.006)
#     ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70)
#     ap.add_argument("--face-strong-thresh", type=float, default=0.50)

#     # Identity smoothing
#     ap.add_argument("--name-decay", type=float, default=0.85)
#     ap.add_argument("--name-min-score", type=float, default=0.60)
#     ap.add_argument("--name-margin", type=float, default=0.30)
#     ap.add_argument("--name-ttl", type=int, default=20)
#     ap.add_argument("--name-face-weight", type=float, default=1.2)
#     ap.add_argument("--name-body-weight", type=float, default=0.5)

#     # Drawing/ghost control
#     ap.add_argument("--draw-only-matched", action="store_true")
#     ap.add_argument("--min-det-conf", type=float, default=0.45)
#     ap.add_argument("--iou-max-miss", type=int, default=5)

#     # Duplicate-name suppression / multi-camera ownership
#     ap.add_argument("--allow-duplicate-names", action="store_true")
#     ap.add_argument("--dedup-iou", type=float, default=0.0)
#     ap.add_argument("--no-global-unique-names", action="store_true")
#     ap.add_argument("--global-hold-seconds", type=float, default=0.5)
#     ap.add_argument("--global-switch-margin", type=float, default=0.02)
#     ap.add_argument("--show-global-id", action="store_true")

#     # Output & view (accepted for compatibility; service ignores)
#     ap.add_argument("--show", action="store_true")

#     # CSV report
#     ap.add_argument("--save-csv", action="store_true")
#     ap.add_argument("--csv", default="detections_summary.csv")
#     ap.add_argument("--report-gap-seconds", type=float, default=2.0)
#     ap.add_argument("--report-time-format", default="%H:%M:%S")

#     # FPS overlays (still useful in MJPEG output)
#     ap.add_argument("--overlay-fps", action="store_true")

#     args = ap.parse_args(argv)
#     return args


# def parse_pipeline_args(pipeline_args: str | None) -> argparse.Namespace:
#     """
#     Used by the FastAPI service to parse PIPELINE_ARGS/pipeline_args env string.
#     """
#     s = str(pipeline_args or "").strip()
#     argv = shlex.split(s) if s else []
#     return parse_args(argv)


# # ----------------------------
# # Face-only process_one_frame
# # ----------------------------


# def process_one_frame(
#     frame_idx: int,
#     frame_bgr: np.ndarray,
#     sid: int,
#     yolo,
#     args,
#     deep_tracker,
#     iou_tracker: IOUTracker,
#     people: list[PersonEntry],
#     reid_extractor,
#     face_app,
#     face_gallery: FaceGallery,
#     identity_state: dict,
#     device_is_cuda: bool,
#     global_owner: Optional[GlobalNameOwner] = None,
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """
#     RULES (your requirement):
#       ✅ Only draw bounding boxes when a face is visible AND recognized (KNOWN).
#       ✅ Unknown persons: NO bounding box at all.
#       ✅ Body never labels alone. It can only support the face label when it agrees.
#     """

#     # Optional resize
#     rw, rh = int(args.resize[0]), int(args.resize[1])
#     if rw > 0 and rh > 0:
#         frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

#     H, W = frame_bgr.shape[:2]

#     # Map name -> member_id
#     name_to_uid: Dict[str, int] = {}
#     try:
#         for p in people:
#             name_to_uid[str(p.name)] = int(p.user_id)
#     except Exception:
#         name_to_uid = {}

#     # YOLO detect
#     tlwh_conf: list[list[float]] = []
#     if yolo is not None:
#         try:
#             res = _yolo_forward_safe(yolo, frame_bgr, args)
#             boxes = res[0].boxes if (res and len(res)) else None
#             if boxes is not None:
#                 xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                 conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                 cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
#                 keep = cls == 0  # person
#                 xyxy, conf = xyxy[keep], conf[keep]
#                 for (x1, y1, x2, y2), c in zip(xyxy, conf):
#                     x1f = float(max(0, min(W - 1, x1)))
#                     y1f = float(max(0, min(H - 1, y1)))
#                     x2f = float(max(0, min(W - 1, x2)))
#                     y2f = float(max(0, min(H - 1, y2)))
#                     ww = float(max(1.0, x2f - x1f))
#                     hh = float(max(1.0, y2f - y1f))
#                     if ww < args.min_box_wh or hh < args.min_box_wh:
#                         continue
#                     tlwh_conf.append([x1f, y1f, ww, hh, float(c)])
#         except Exception as e:
#             print(f"[SRC {sid}] YOLO error:", e)

#     dets_np = np.asarray(tlwh_conf, dtype=np.float32)
#     if dets_np.ndim != 2:
#         dets_np = dets_np.reshape((0, 5)).astype(np.float32)

#     dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

#     # Tracker
#     out_tracks = []
#     if deep_tracker is not None:
#         try:
#             out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
#         except Exception as e:
#             print(f"[SRC {sid}] DeepSORT update_tracks error:", e)
#             out_tracks = []
#     else:
#         out_tracks = iou_tracker.update(dets_np)

#     # Face detect (every N frames)
#     recognized_faces: List[Dict[str, Any]] = []
#     do_face = (
#         face_app is not None
#         and face_gallery is not None
#         and (not face_gallery.is_empty())
#         and (frame_idx % max(1, int(args.face_every_n)) == 0)
#     )
#     if do_face:
#         try:
#             with _face_lock:
#                 faces = face_app.get(np.ascontiguousarray(frame_bgr))
#             for f in safe_iter_faces(faces):
#                 bbox = getattr(f, "bbox", None)
#                 if bbox is None:
#                     continue
#                 b = np.asarray(bbox).reshape(-1)
#                 if b.size < 4:
#                     continue
#                 fx1, fy1, fx2, fy2 = map(float, b[:4])
#                 fw = max(0.0, fx2 - fx1)
#                 fh = max(0.0, fy2 - fy1)
#                 if fw < float(args.min_face_px) or fh < float(args.min_face_px):
#                     continue

#                 emb = extract_face_embedding(f)
#                 if emb is None:
#                     continue
#                 emb = l2_normalize(np.asarray(emb, dtype=np.float32))
#                 flabel, fsim, fsecond = best_face_label_top2(emb, face_gallery)
#                 if flabel is None:
#                     continue

#                 gap = float(fsim - fsecond)
#                 if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
#                     recognized_faces.append(
#                         {
#                             "bbox": (fx1, fy1, fx2, fy2),
#                             "label": str(flabel),
#                             "sim": float(fsim),
#                             "second": float(fsecond),
#                             "gap": float(gap),
#                         }
#                     )
#         except Exception as e:
#             print(f"[SRC {sid}] FaceAnalysis error:", e)

#     out = frame_bgr.copy()
#     tracks_info: List[Dict[str, Any]] = []

#     for t in out_tracks:
#         time_since_update = getattr(t, "time_since_update", 0)
#         had_match_this_frame = (time_since_update == 0) or (getattr(t, "last_detection", None) is not None)
#         if args.draw_only_matched and not had_match_this_frame:
#             continue

#         try:
#             if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
#                 continue
#             if hasattr(t, "to_tlbr"):
#                 ltrb = t.to_tlbr()
#             elif hasattr(t, "to_ltrb"):
#                 ltrb = t.to_ltrb()
#             else:
#                 ltrb = t.to_tlbr()

#             x1, y1, x2, y2 = map(int, ltrb)
#             x1 = int(max(0, min(W - 1, x1)))
#             y1 = int(max(0, min(H - 1, y1)))
#             x2 = int(max(0, min(W, x2)))
#             y2 = int(max(0, min(H, y2)))
#             if x2 <= x1 or y2 <= y1:
#                 continue

#             tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
#         except Exception:
#             continue

#         det_conf = None
#         try:
#             if hasattr(t, "det_conf") and t.det_conf is not None:
#                 det_conf = float(t.det_conf)
#             elif hasattr(t, "last_detection") and t.last_detection is not None:
#                 ld = t.last_detection
#                 if isinstance(ld, (list, tuple)) and len(ld) >= 2:
#                     det_conf = float(ld[1])
#                 elif isinstance(ld, dict):
#                     det_conf = float(ld.get("confidence", ld.get("det_conf", 0.0)))
#         except Exception:
#             det_conf = None

#         if args.min_det_conf > 0 and det_conf is not None and det_conf < args.min_det_conf:
#             if args.draw_only_matched:
#                 continue

#         # Face link to person track (recognized faces only)
#         face_label, face_sim, face_gap = "", 0.0, 0.0
#         face_hit = False

#         if recognized_faces:
#             t_xyxy = (float(x1), float(y1), float(x2), float(y2))
#             p_area = float(max(1.0, (x2 - x1) * (y2 - y1)))

#             best_idx, best_score = -1, -1e9
#             for idx, fm in enumerate(recognized_faces):
#                 fbox = fm["bbox"]

#                 cx = 0.5 * (float(fbox[0]) + float(fbox[2]))
#                 cy = 0.5 * (float(fbox[1]) + float(fbox[3]))
#                 if args.face_center_in_person and not _point_in_xyxy(cx, cy, t_xyxy):
#                     continue

#                 top_limit = float(y1) + float(args.face_center_y_max_ratio) * float(max(1, (y2 - y1)))
#                 if cy > top_limit:
#                     continue

#                 link = ioa_xyxy(fbox, t_xyxy) if args.face_link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)
#                 if link < float(args.face_iou_link):
#                     continue

#                 if float(args.min_face_area_ratio) > 0:
#                     f_area = float(max(1.0, (float(fbox[2]) - float(fbox[0])) * (float(fbox[3]) - float(fbox[1]))))
#                     if (f_area / p_area) < float(args.min_face_area_ratio):
#                         continue

#                 score = float(link) + 0.25 * float(fm["sim"])
#                 if score > best_score:
#                     best_score = score
#                     best_idx = idx

#             if best_idx >= 0:
#                 fm = recognized_faces[best_idx]
#                 face_label = str(fm["label"])
#                 face_sim = float(fm["sim"])
#                 face_gap = float(fm["gap"])
#                 face_hit = True

#         # Update face visibility TTL for this track
#         entry = identity_state.setdefault(
#             tid,
#             {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
#         )

#         # ✅ Anti-flicker: only decay face_vis_ttl on frames where face ran
#         dec = 1 if do_face else 0
#         entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - dec)

#         if face_hit and face_label:
#             entry["face_vis_ttl"] = max(1, int(args.face_hold_frames))
#             entry["last_face_label"] = face_label
#             entry["last_face_sim"] = float(face_sim)

#         tracks_info.append(
#             {
#                 "tid": tid,
#                 "bbox": (x1, y1, x2, y2),
#                 "det_conf": det_conf,
#                 "face_hit": face_hit,
#                 "face_label": face_label,
#                 "face_sim": face_sim,
#                 "face_gap": face_gap,
#                 "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
#                 "last_face_sim": float(entry.get("last_face_sim", 0.0)),
#             }
#         )

#     # Body support: compute ONLY for tracks that have a recognized face hit this frame.
#     face_hit_tracks = [r for r in tracks_info if r["face_hit"] and r["face_label"]]
#     body_by_tid: Dict[int, Tuple[str, float, float]] = {}

#     if face_hit_tracks and (reid_extractor is not None) and people:
#         crops: List[np.ndarray] = []
#         tids: List[int] = []
#         for r in face_hit_tracks:
#             x1, y1, x2, y2 = r["bbox"]
#             crop = frame_bgr[y1:y2, x1:x2]
#             crops.append(crop)
#             tids.append(int(r["tid"]))

#         bs = 16
#         try:
#             bs = max(1, int(getattr(args, "reid_batch_size", 16)))
#         except Exception:
#             bs = 16

#         for start in range(0, len(crops), bs):
#             chunk = crops[start : start + bs]
#             chunk_tids = tids[start : start + bs]
#             feats = extract_body_embeddings_batch(reid_extractor, chunk, device_is_cuda=device_is_cuda, use_half=bool(args.half))
#             if feats is None:
#                 continue
#             for tid, emb in zip(chunk_tids, feats):
#                 blabel, bsim, bsecond = best_body_label_from_emb(emb, people, topk=max(1, int(args.reid_topk)))
#                 if blabel is None:
#                     continue
#                 body_by_tid[int(tid)] = (str(blabel), float(bsim), float(bsecond))

#     draw_candidates: List[DrawItem] = []

#     for r in tracks_info:
#         tid = int(r["tid"])
#         x1, y1, x2, y2 = r["bbox"]

#         face_hit = bool(r["face_hit"])
#         face_label = str(r["face_label"])
#         face_sim = float(r["face_sim"])
#         last_face_sim = float(r.get("last_face_sim", 0.0))

#         candidates: List[Tuple[str, float, str]] = []

#         if face_hit and face_label:
#             candidates.append((face_label, face_sim, "face"))

#             if tid in body_by_tid:
#                 b_label, b_sim, b_second = body_by_tid[tid]
#                 b_gap = float(b_sim - b_second)
#                 if (b_label == face_label) and (b_sim >= float(args.gallery_thresh)) and (b_gap >= float(args.gallery_gap)):
#                     candidates.append((face_label, b_sim, "body"))
#                 else:
#                     # strong body conflict can suppress weak face
#                     if (b_label != face_label) and (b_sim >= max(float(args.gallery_thresh), 0.80)) and (b_gap >= max(float(args.gallery_gap), 0.10)):
#                         if face_sim < float(args.face_strong_thresh):
#                             candidates = []

#         # ✅ If face DID NOT run this frame and we have NO new evidence:
#         # keep last identity without decaying (reduces flicker in MJPEG)
#         entry = identity_state.setdefault(
#             tid,
#             {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
#         )

#         if (not do_face) and (not candidates):
#             stable_name = str(entry.get("last", "") or "")
#             stable_score = float(entry.get("scores", {}).get(stable_name, 0.0)) if stable_name else 0.0
#         else:
#             stable_name, stable_score, entry = update_track_identity(
#                 identity_state,
#                 tid,
#                 candidates,
#                 decay=args.name_decay,
#                 min_score=args.name_min_score,
#                 margin=args.name_margin,
#                 ttl_reset=args.name_ttl,
#                 w_face=args.name_face_weight,
#                 w_body=args.name_body_weight,
#             )

#         # ✅ SHOW ONLY KNOWN (face recognized + stable + face_vis_ttl>0)
#         show_label = bool(stable_name) and (int(entry.get("face_vis_ttl", 0)) > 0)
#         if not show_label:
#             continue

#         disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))
#         uid = int(name_to_uid.get(stable_name, -1))

#         draw_candidates.append(
#             DrawItem(
#                 tid=tid,
#                 bbox=(int(x1), int(y1), int(x2), int(y2)),
#                 name=str(stable_name),
#                 user_id=uid,
#                 face_sim=float(disp_face_sim),
#                 stable_score=float(stable_score),
#                 det_conf=r.get("det_conf", None),
#                 face_hit=bool(face_hit),
#             )
#         )

#     # For report: names present in this camera frame
#     present_names = sorted({it.name for it in draw_candidates if it.name})

#     # Per-camera duplicate suppression
#     if not bool(getattr(args, "allow_duplicate_names", False)):
#         draw_final = deduplicate_draw_items(draw_candidates, iou_thresh=float(getattr(args, "dedup_iou", 0.0)))
#     else:
#         draw_final = draw_candidates

#     # Optional: cross-camera global unique names (display only)
#     if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
#         gated: List[DrawItem] = []
#         for it in draw_final:
#             score = float(it.face_sim)
#             if global_owner.allow(it.name, sid=int(sid), score=score):
#                 gated.append(it)
#         draw_final = gated

#     # ✅ DRAW ONLY KNOWN (draw_final already filtered)
#     shown = 0
#     for it in draw_final:
#         x1, y1, x2, y2 = it.bbox
#         color = (0, 255, 0)
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

#         if bool(getattr(args, "show_global_id", False)) and it.user_id >= 0:
#             label_txt = f"{it.name} [{it.user_id}]"
#         else:
#             label_txt = f"{it.name}"

#         if it.face_hit:
#             label_txt = f"{label_txt} | F {float(it.face_sim):.2f}"

#         cv2.putText(out, label_txt, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         shown += 1

#     meta = {
#         "tracks": int(len(tracks_info)),
#         "shown": int(shown),
#         "faces_recognized": int(len(recognized_faces)),
#         "do_face": bool(do_face),
#         "present_names": present_names,
#     }
#     return out, meta


# def processor_thread(
#     sid: int,
#     camera_id_for_gallery: Optional[int],
#     vs,
#     out_buf: FrameBuffer,
#     yolo,
#     args,
#     deep_tracker,
#     iou_tracker: IOUTracker,
#     gallery_mgr: GalleryManager,
#     reid_extractor,
#     face_app,
#     global_owner: Optional[GlobalNameOwner],
#     report: Optional[SummaryReport],
#     report_cam_index: Optional[int],
#     stop_event: threading.Event,
# ):
#     frame_idx = 0
#     identity_state: dict[int, dict] = {}

#     device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

#     last_t = time.time()
#     fps_ema = 0.0
#     alpha = 0.10

#     while not stop_event.is_set():
#         ok, frame, ts_cap = vs.read()
#         if not ok or frame is None:
#             time.sleep(0.005)
#             continue

#         # Drop stale frames policy
#         if int(args.max_queue_age_ms) > 0:
#             now = time.time()
#             age_ms = (now - float(ts_cap)) * 1000.0
#             dropped_here = 0
#             while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
#                 try:
#                     vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
#                 except Exception:
#                     pass
#                 ok2, frame2, ts2 = vs.read()
#                 if not ok2 or frame2 is None:
#                     break
#                 frame, ts_cap = frame2, ts2
#                 age_ms = (time.time() - float(ts_cap)) * 1000.0
#                 dropped_here += 1

#         try:
#             gallery_mgr.maybe_reload(args)
#             people, face_gallery = gallery_mgr.snapshot(camera_id=camera_id_for_gallery)

#             out, meta = process_one_frame(
#                 frame_idx=frame_idx,
#                 frame_bgr=frame,
#                 sid=sid,
#                 yolo=yolo,
#                 args=args,
#                 deep_tracker=deep_tracker,
#                 iou_tracker=iou_tracker,
#                 people=people,
#                 reid_extractor=reid_extractor,
#                 face_app=face_app,
#                 face_gallery=face_gallery,
#                 identity_state=identity_state,
#                 device_is_cuda=device_is_cuda,
#                 global_owner=global_owner,
#             )

#             # Debug overlay counts (helps verify it's working in MJPEG)
#             tracks_count = meta.get("tracks", 0)
#             shown_count = meta.get("shown", 0)
#             faces_count = meta.get("faces_recognized", 0)

#             # cv2.putText(out, f"Tracks: {tracks_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             # cv2.putText(out, f"Known Shown: {shown_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(out, f"Faces(recognized): {faces_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#             # FPS overlay
#             if bool(getattr(args, "overlay_fps", False)):
#                 now = time.time()
#                 dt = max(1e-6, now - last_t)
#                 inst_fps = 1.0 / dt
#                 fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
#                 last_t = now
#                 cv2.putText(out, f"FPS: {fps_ema:.1f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             # Report update
#             if report is not None and report_cam_index is not None:
#                 names = meta.get("present_names", []) or []
#                 ts_use = float(ts_cap) if ts_cap else time.time()
#                 report.update(cam_id=int(report_cam_index), present_names=list(names), ts=ts_use)

#             out_buf.set(out, meta=meta)
#             frame_idx += 1

#             # Terminal log every 30 frames
#             if frame_idx % 30 == 0:
#                 print(f"[SRC {sid}] Tracks={tracks_count} KnownShown={shown_count} Faces={faces_count}")

#         except Exception as e:
#             print(f"[PROC {sid}] error:", e)
#             time.sleep(0.001)


# # ----------------------------
# # Runner (service mode)
# # ----------------------------


# def _rtsp_url_from_db_camera_ip(ip: str) -> str:
#     """
#     Builds RTSP URL for DB camera row using env template:
#       RTSP_URL_TEMPLATE=rtsp://{username}:{password}@{ip}:{port}/Streaming/channels/{stream}

#     Uses env:
#       RTSP_USER, RTSP_PASS, RTSP_PORT, RTSP_STREAM, RTSP_SCHEME, RTSP_PATH
#     """
#     ip = str(ip or "").strip()

#     username = os.environ.get("RTSP_USER", "admin")
#     password = os.environ.get("RTSP_PASS", "admin")
#     port = os.environ.get("RTSP_PORT", "554")
#     stream = os.environ.get("RTSP_STREAM", "101")
#     scheme = os.environ.get("RTSP_SCHEME", "rtsp")
#     path = os.environ.get("RTSP_PATH", "Streaming/channels/")

#     # sanitize credentials
#     user_enc = quote(unquote(username or ""), safe="")
#     pass_enc = quote(unquote(password or ""), safe="")

#     tmpl = os.environ.get("RTSP_URL_TEMPLATE", "").strip()
#     if tmpl:
#         try:
#             return tmpl.format(
#                 username=user_enc,
#                 password=pass_enc,
#                 ip=ip,
#                 port=str(port),
#                 stream=str(stream),
#                 scheme=str(scheme),
#                 path=str(path),
#             )
#         except Exception:
#             pass

#     # fallback
#     path = str(path or "").lstrip("/")
#     if not path.endswith("/"):
#         path = path + "/"
#     return f"{scheme}://{user_enc}:{pass_enc}@{ip}:{port}/{path}{stream}"


# class TrackingRunner:
#     """
#     Service runner.

#     Modes:
#     - manual sources mode: args.src contains URLs => cam_id = 0..N-1 (pre-started)
#     - DB camera mode:
#         * if args.src contains numeric IDs => pre-start those camera IDs
#         * else => start cameras on-demand when /mjpeg/{cam_id} requested
#       cam_id = DB cameras.id
#     """

#     def __init__(self, args: argparse.Namespace):
#         self.args = args
#         self._lock = threading.Lock()
#         self._started = False

#         self.gallery_mgr: Optional[GalleryManager] = None
#         self.global_owner: Optional[GlobalNameOwner] = None

#         self.yolo = None
#         self.reid_extractor = None
#         self.face_app = None

#         self.report: Optional[SummaryReport] = None

#         self.stop_event = threading.Event()

#         # Manual sources mode storage (cam_id -> state)
#         self._manual_states: dict[int, dict[str, Any]] = {}

#         # DB mode storage (camera_id -> state)
#         self._db_states: dict[int, dict[str, Any]] = {}

#         # DB camera id -> report column index (0..num_cams-1)
#         self._camid_to_report_index: dict[int, int] = {}

#         # DB engine for cameras lookup
#         self._db_engine = None
#         self._Session = None

#         self._mode: str = "db"  # "manual" or "db"

#     def start(self) -> None:
#         with self._lock:
#             if self._started:
#                 return
#             self._started = True

#         args = self.args

#         if not args.use_db:
#             raise RuntimeError("DB-first pipeline: pass --use-db")
#         if not args.db_url:
#             raise RuntimeError("--db-url is required")

#         # Decide mode
#         src_list = list(getattr(args, "src", []) or [])
#         if any(_looks_like_url(s) for s in src_list):
#             self._mode = "manual"
#         else:
#             self._mode = "db"

#         # Global unique-name behavior
#         args.global_unique_names = not bool(getattr(args, "no_global_unique_names", False))

#         if args.cudnn_benchmark:
#             torch.backends.cudnn.benchmark = True

#         try:
#             torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
#         except Exception:
#             pass

#         gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
#         if args.half and not gpu:
#             print("[WARN] --half requested but CUDA not available; disabling FP16.")
#             args.half = False

#         print(f"[INIT] device={args.device} cuda_available={torch.cuda.is_available()} half={args.half}")
#         print(f"[INIT] mode={self._mode}")

#         # DB gallery
#         self.gallery_mgr = GalleryManager(args)

#         # DB session (for cameras) in DB mode
#         if self._mode == "db":
#             self._db_engine = create_engine(args.db_url, pool_pre_ping=True)
#             self._Session = sessionmaker(bind=self._db_engine)
#             self._build_report_camera_index_map()

#         # Global unique names gate
#         if bool(getattr(args, "global_unique_names", False)):
#             self.global_owner = GlobalNameOwner(
#                 hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
#                 switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
#             )
#             print(f"[INIT] Global unique names: ON (hold={self.global_owner.hold_seconds}s, margin={self.global_owner.switch_margin})")
#         else:
#             print("[INIT] Global unique names: OFF")

#         # YOLO
#         if YOLO is not None:
#             try:
#                 weights = args.yolo_weights
#                 if not Path(weights).exists():
#                     print(f"[INIT] {weights} not found, falling back to yolov8n.pt")
#                     weights = "yolov8n.pt"
#                 self.yolo = YOLO(weights)
#                 if gpu:
#                     try:
#                         self.yolo.to(args.device)
#                     except Exception as e:
#                         print("[WARN] YOLO .to(device) failed:", e)
#                 print("[INIT] YOLO ready")
#             except Exception as e:
#                 print("[ERROR] YOLO load failed:", e)
#                 self.yolo = None
#         else:
#             print("[WARN] ultralytics not installed; detection disabled")

#         # TorchReID extractor (body support only)
#         if TorchreidExtractor is not None:
#             try:
#                 dev = args.device if gpu else "cpu"
#                 if args.reid_weights and Path(args.reid_weights).exists():
#                     self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, model_path=args.reid_weights, device=dev)
#                 else:
#                     self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, device=dev)
#                 print(f"[INIT] TorchReID ready (model={args.reid_model}, device={dev})")
#             except Exception as e:
#                 print("[WARN] TorchReID init failed; body support disabled:", e)
#                 self.reid_extractor = None
#         else:
#             print("[WARN] torchreid not installed; body support disabled")

#         # Face
#         self.face_app = init_face_engine(
#             args.use_face,
#             args.device,
#             args.face_model,
#             int(args.face_det_size[0]),
#             int(args.face_det_size[1]),
#             face_provider=getattr(args, "face_provider", "auto"),
#             ort_log=getattr(args, "ort_log", False),
#         )

#         # DeepSORT availability note
#         if args.use_deepsort and DeepSort is None:
#             print("[WARN] --use-deepsort requested but deep-sort-realtime not installed.")

#         # Summary report
#         if args.save_csv:
#             num_cols = self._report_num_cams()
#             self.report = SummaryReport(
#                 num_cams=max(1, int(num_cols)),
#                 gap_seconds=float(getattr(args, "report_gap_seconds", 2.0)),
#                 time_format=str(getattr(args, "report_time_format", "%H:%M:%S")),
#             )
#             print(f"[INIT] Summary CSV report: ON -> {args.csv}")
#         else:
#             self.report = None

#         # Start initial sources
#         if self._mode == "manual":
#             self._start_manual_sources(src_list)
#         else:
#             # DB mode: start numeric camera ids listed in --src (optional)
#             for s in src_list:
#                 if _is_int_str(s):
#                     self.ensure_camera_started(int(s))

#         print("[Runner] Background detection ready.")

#     # ---------- report mapping ----------
#     def _build_report_camera_index_map(self) -> None:
#         """
#         Creates a stable mapping from DB camera.id -> column index for CSV report.
#         """
#         self._camid_to_report_index = {}
#         if self._Session is None:
#             return

#         Base = declarative_base()

#         class CameraRow(Base):
#             __tablename__ = "cameras"
#             id = Column(Integer, primary_key=True)
#             name = Column(String(64))
#             ip_address = Column(String(64))
#             is_active = Column(Boolean)

#         with self._Session() as session:
#             stmt = select(CameraRow.id).order_by(CameraRow.id.asc())
#             rows = session.execute(stmt).all()
#             cam_ids = [int(r.id) for r in rows]

#         for idx, cam_id in enumerate(cam_ids):
#             self._camid_to_report_index[int(cam_id)] = int(idx)

#     def _report_num_cams(self) -> int:
#         if self._mode == "manual":
#             return len(self._manual_states) if self._manual_states else max(1, len(getattr(self.args, "src", []) or []))
#         # DB mode
#         return max(1, len(self._camid_to_report_index) if self._camid_to_report_index else 1)

#     # ---------- manual sources ----------
#     def _start_manual_sources(self, src_list: list[str]) -> None:
#         args = self.args
#         gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

#         for i, raw_src in enumerate(src_list):
#             cam_id = int(i)  # API cam_id is index
#             src = raw_src.strip() if isinstance(raw_src, str) else raw_src

#             vs = self._open_stream(src)

#             deep_tracker = self._make_deepsort(gpu=gpu) if self._should_use_deepsort() else None
#             iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=0.3)

#             buf = FrameBuffer()
#             self._manual_states[cam_id] = {
#                 "cam_id": cam_id,
#                 "sid": cam_id,
#                 "src": src,
#                 "vs": vs,
#                 "deep": deep_tracker,
#                 "iou": iou_tracker,
#                 "buf": buf,
#                 "thread": None,
#             }

#             if not vs.is_opened():
#                 print(f"[SRC {cam_id}] open=False :: {src}")
#                 continue

#             t = threading.Thread(
#                 target=processor_thread,
#                 args=(
#                     int(cam_id),  # sid
#                     None,  # camera_id_for_gallery (unknown in manual mode)
#                     vs,
#                     buf,
#                     self.yolo,
#                     args,
#                     deep_tracker,
#                     iou_tracker,
#                     self.gallery_mgr,
#                     self.reid_extractor,
#                     self.face_app,
#                     self.global_owner,
#                     self.report,
#                     int(cam_id),  # report column index = cam_id
#                     self.stop_event,
#                 ),
#                 daemon=True,
#             )
#             t.start()
#             self._manual_states[cam_id]["thread"] = t

#             print(f"[SRC {cam_id}] open=True :: {src}")

#     # ---------- DB cameras ----------
#     def ensure_camera_started(self, camera_id: int) -> bool:
#         """
#         DB mode: starts processing for a DB camera.id if not already running.
#         Returns True if running/opened, else False.
#         """
#         if self._mode != "db":
#             return False

#         cam_id = int(camera_id)
#         with self._lock:
#             if cam_id in self._db_states:
#                 st = self._db_states[cam_id]
#                 vs = st.get("vs")
#                 return bool(vs is not None and vs.is_opened())

#         # Lookup camera in DB
#         cam = self._get_db_camera(cam_id)
#         if cam is None:
#             return False

#         ip = str(cam.get("ip_address") or "").strip()
#         if not ip:
#             return False

#         src = _rtsp_url_from_db_camera_ip(ip)

#         vs = self._open_stream(src)
#         opened = vs.is_opened()

#         gpu = torch.cuda.is_available() and ("cuda" in str(self.args.device).lower())

#         deep_tracker = self._make_deepsort(gpu=gpu) if self._should_use_deepsort() else None
#         iou_tracker = IOUTracker(max_miss=max(1, int(self.args.iou_max_miss)), iou_thresh=0.3)

#         buf = FrameBuffer()

#         # Stable internal sid for global gating + logging:
#         sid = self._camid_to_report_index.get(cam_id, cam_id)

#         report_idx = self._camid_to_report_index.get(cam_id, None)

#         with self._lock:
#             self._db_states[cam_id] = {
#                 "cam_id": cam_id,
#                 "sid": int(sid),
#                 "camera": cam,
#                 "src": src,
#                 "vs": vs,
#                 "deep": deep_tracker,
#                 "iou": iou_tracker,
#                 "buf": buf,
#                 "thread": None,
#                 "started_at": time.time(),
#             }

#         print(f"[SRC {cam_id}] open={opened} :: {src}")

#         if not opened:
#             return False

#         t = threading.Thread(
#             target=processor_thread,
#             args=(
#                 int(sid),  # sid
#                 int(cam_id),  # camera_id_for_gallery (important for camera-specific embeddings)
#                 vs,
#                 buf,
#                 self.yolo,
#                 self.args,
#                 deep_tracker,
#                 iou_tracker,
#                 self.gallery_mgr,
#                 self.reid_extractor,
#                 self.face_app,
#                 self.global_owner,
#                 self.report,
#                 report_idx,
#                 self.stop_event,
#             ),
#             daemon=True,
#         )
#         t.start()

#         with self._lock:
#             if cam_id in self._db_states:
#                 self._db_states[cam_id]["thread"] = t

#         return True

#     def _get_db_camera(self, camera_id: int) -> Optional[dict[str, Any]]:
#         if self._Session is None:
#             return None

#         Base = declarative_base()

#         class CameraRow(Base):
#             __tablename__ = "cameras"
#             id = Column(Integer, primary_key=True)
#             name = Column(String(64))
#             ip_address = Column(String(64))
#             is_active = Column(Boolean)

#         with self._Session() as session:
#             stmt = select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).where(CameraRow.id == int(camera_id))
#             row = session.execute(stmt).first()
#             if not row:
#                 return None
#             return {
#                 "id": int(row.id),
#                 "name": str(row.name) if row.name is not None else "",
#                 "ip_address": str(row.ip_address) if row.ip_address is not None else "",
#                 "is_active": bool(row.is_active) if row.is_active is not None else True,
#             }

#     # ---------- common ----------
#     def _should_use_deepsort(self) -> bool:
#         args = self.args
#         if bool(getattr(args, "no_deepsort", False)):
#             return False
#         if bool(getattr(args, "use_deepsort", False)):
#             return DeepSort is not None
#         return DeepSort is not None

#     def _make_deepsort(self, gpu: bool):
#         args = self.args
#         if DeepSort is None:
#             return None
#         try:
#             return DeepSort(
#                 max_age=int(args.max_age),
#                 n_init=int(args.n_init),
#                 nn_budget=int(args.nn_budget),
#                 max_cosine_distance=float(args.tracker_max_cosine),
#                 nms_max_overlap=float(args.tracker_nms_overlap),
#                 embedder="torchreid",
#                 embedder_gpu=bool(gpu),
#                 half=(bool(gpu) and bool(args.half)),
#                 bgr=True,
#             )
#         except Exception as e:
#             print("[WARN] DeepSORT init failed, fallback to IoU tracker:", e)
#             return None

#     def _open_stream(self, src: str):
#         args = self.args
#         src = str(src)
#         if args.reader == "latest":
#             return LatestStream(src, rtsp_buffer=args.rtsp_buffer, decode_skip=args.decode_skip, rtsp_transport=args.rtsp_transport)
#         if args.reader == "ffmpeg":
#             return FFmpegStream(
#                 src,
#                 queue_size=args.queue_size,
#                 rtsp_transport=args.rtsp_transport,
#                 use_cuda=bool(args.ffmpeg_cuda),
#                 force_w=int(args.ffmpeg_width),
#                 force_h=int(args.ffmpeg_height),
#             )
#         return AdaptiveQueueStream(src, queue_size=args.queue_size, rtsp_transport=args.rtsp_transport, use_opencv=True)

#     def stop(self) -> None:
#         with self._lock:
#             if not self._started:
#                 return
#             self._started = False

#         print("[Runner] Stopping...")
#         self.stop_event.set()

#         # stop streams
#         if self._mode == "manual":
#             for st in list(self._manual_states.values()):
#                 try:
#                     st["vs"].release()
#                 except Exception:
#                     pass
#             for st in list(self._manual_states.values()):
#                 t = st.get("thread")
#                 if t is not None:
#                     try:
#                         t.join(timeout=2.0)
#                     except Exception:
#                         pass
#         else:
#             for st in list(self._db_states.values()):
#                 try:
#                     st["vs"].release()
#                 except Exception:
#                     pass
#             for st in list(self._db_states.values()):
#                 t = st.get("thread")
#                 if t is not None:
#                     try:
#                         t.join(timeout=2.0)
#                     except Exception:
#                         pass

#         # report final write
#         if self.report is not None:
#             try:
#                 self.report.stop()
#                 self.report.write_csv(self.args.csv)
#                 print(f"[DONE] Summary CSV saved: {self.args.csv}")
#             except Exception as e:
#                 print("[WARN] Failed to write summary CSV:", e)

#         print("[Runner] Stopped.")

#     # ---------- API helpers ----------
#     def get_camera_buffer(self, cam_id: int) -> Optional[FrameBuffer]:
#         """
#         API cam_id:
#           - manual mode: cam_id is index 0..N-1
#           - db mode: cam_id is DB cameras.id
#         """
#         if self._mode == "manual":
#             st = self._manual_states.get(int(cam_id))
#             if not st:
#                 return None
#             return st.get("buf")

#         # db mode
#         ok = self.ensure_camera_started(int(cam_id))
#         if not ok:
#             return None
#         st = self._db_states.get(int(cam_id))
#         if not st:
#             return None
#         return st.get("buf")

#     def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
#         """
#         Returns camera list for /cameras endpoint.
#         """
#         if self._mode == "manual":
#             out = []
#             for cid, st in sorted(self._manual_states.items(), key=lambda kv: kv[0]):
#                 vs = st.get("vs")
#                 out.append(
#                     {
#                         "id": int(cid),
#                         "mode": "manual",
#                         "opened": bool(vs.is_opened()) if vs else False,
#                         "src": str(st.get("src", "")),
#                     }
#                 )
#             return out

#         if self._Session is None:
#             return []

#         Base = declarative_base()

#         class CameraRow(Base):
#             __tablename__ = "cameras"
#             id = Column(Integer, primary_key=True)
#             name = Column(String(64))
#             ip_address = Column(String(64))
#             is_active = Column(Boolean)

#         cams = []
#         with self._Session() as session:
#             stmt = select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).order_by(CameraRow.id.asc())
#             rows = session.execute(stmt).all()
#             for r in rows:
#                 is_active = bool(r.is_active) if r.is_active is not None else True
#                 if active_only and (not is_active):
#                     continue
#                 cam_id = int(r.id)
#                 st = self._db_states.get(cam_id)
#                 opened = bool(st["vs"].is_opened()) if st and st.get("vs") else False
#                 cams.append(
#                     {
#                         "id": cam_id,
#                         "name": str(r.name) if r.name is not None else "",
#                         "ip_address": str(r.ip_address) if r.ip_address is not None else "",
#                         "is_active": is_active,
#                         "running": bool(st is not None),
#                         "opened": opened,
#                     }
#                 )
#         return cams

#     def status(self) -> Dict[str, Any]:
#         st = {
#             "started": bool(self._started),
#             "mode": self._mode,
#             "gallery_last_reload_ts": float(self.gallery_mgr.last_load_ts) if self.gallery_mgr else 0.0,
#             "running": [],
#         }

#         if self._mode == "manual":
#             for cid, s in sorted(self._manual_states.items(), key=lambda kv: kv[0]):
#                 vs = s.get("vs")
#                 st["running"].append({"id": int(cid), "opened": bool(vs.is_opened()) if vs else False, "src": str(s.get("src", ""))})
#         else:
#             for cam_id, s in sorted(self._db_states.items(), key=lambda kv: kv[0]):
#                 vs = s.get("vs")
#                 st["running"].append({"id": int(cam_id), "opened": bool(vs.is_opened()) if vs else False, "src": str(s.get("src", ""))})

#         return st

#     def write_report_snapshot(self, path: Optional[str] = None) -> str:
#         if self.report is None:
#             raise RuntimeError("CSV reporting not enabled. Add --save-csv in PIPELINE_ARGS.")
#         out_path = str(path or self.args.csv)
#         self.report.write_csv_snapshot(out_path)
#         return out_path

"""
Person tracking pipeline with YOLO + DeepSORT.

✅ IMPORTANT (your requirement):
- DO NOT draw bounding boxes for UNKNOWN people
- Draw GREEN boxes only for KNOWN / identified members (face-recognized + stable)

✅ DB changes:
- members table instead of users
- embeddings are in member_embeddings table (member_id + camera_id)

✅ Camera source:
- If you pass --src RTSP URLs => manual mode (cam_id = 0..N-1)
- If you pass --src as camera IDs (numbers) OR omit --src => DB camera mode (cam_id = DB cameras.id)
  -> RTSP URL is built from cameras.ip_address + RTSP_* env template

✅ NEW (your ingestion requirement):
- Every person detection (known or unknown) is logged
- Logs go into a temporary raw table
- A new raw table is created every 5 minutes (configurable)
- Batch inserts (buffered in memory) to avoid DB overload
- Optional cleanup (drop old raw tables after retention window)

Raw table schema:
CREATE TABLE raw_data_<uuid> (
  data_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  camera_id INTEGER NOT NULL CHECK (camera_id >= 0),
  member_id INTEGER CHECK (member_id >= 0),
  guest_temp_id VARCHAR(64),
  guest_data_vector JSONB,
  match_value INTEGER CHECK (match_value >= 0)
);

Designed for FastAPI service mode (MJPEG output).
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gzip
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np
import torch
import json

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

# --- TorchReID gallery extractor (body support only) ---
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
    from sqlalchemy import (
        ARRAY,
        BigInteger,
        Boolean,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        LargeBinary,
        String,
        create_engine,
        func,
        select,
        text,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
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
_yolo_lock = threading.Lock()  # avoid concurrent YOLO kernels on same instance
_reid_lock = threading.Lock()  # avoid concurrent TorchReID kernels on same extractor
_face_lock = threading.Lock()  # InsightFace (ORT/CUDA) guard


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


def _looks_like_url(s: str) -> bool:
    s = str(s or "").strip().lower()
    return "://" in s or s.startswith("rtsp:") or s.startswith("http:") or s.startswith("https:")


def _is_int_str(s: str) -> bool:
    s = str(s or "").strip()
    if not s:
        return False
    if s.startswith("-"):
        return s[1:].isdigit()
    return s.isdigit()


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
# DB gallery + cameras (members + member_embeddings)
# ----------------------------
def decode_bank_blob(raw: bytes | None) -> np.ndarray | None:
    """
    Supports:
      - gzip(np.load(npy))
      - raw npy bytes
    Returns normalized float32 [N,512] or None.
    """
    if not raw:
        return None

    for mode in ("gzip", "plain"):
        try:
            if mode == "gzip":
                data = gzip.decompress(raw)
            else:
                data = raw

            arr = np.load(BytesIO(data), allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float32)

            if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
                continue
            if not np.isfinite(arr).all():
                continue
            return l2_normalize_rows(arr)
        except Exception:
            continue

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


@dataclass
class PersonEntry:
    user_id: int  # kept name for compatibility -> member_id
    name: str
    body_bank: np.ndarray | None
    body_centroid: np.ndarray | None
    face_centroid: np.ndarray | None


@dataclass
class FaceGallery:
    names: list[str]
    mat: np.ndarray

    def is_empty(self) -> bool:
        return (not self.names) or (self.mat is None) or (self.mat.size == 0)


@dataclass
class _Agg:
    body_vecs: list[np.ndarray]
    face_vecs: list[np.ndarray]
    body_banks: list[np.ndarray]
    face_banks: list[np.ndarray]


class GalleryData:
    def __init__(self):
        self.people_global: list[PersonEntry] = []
        self.face_global: FaceGallery = FaceGallery([], np.zeros((0, EXPECTED_DIM), dtype=np.float32))
        self.people_by_cam: dict[int, list[PersonEntry]] = {}
        self.face_by_cam: dict[int, FaceGallery] = {}


def _member_label(member_number: str | None, first_name: str | None, last_name: str | None, member_id: int) -> str:
    fn = (first_name or "").strip()
    ln = (last_name or "").strip()
    mn = (member_number or "").strip()
    if fn and ln:
        return f"{fn} {ln}".strip()
    if fn:
        return fn
    if mn:
        return mn
    return f"member_{int(member_id)}"


def build_galleries_from_db(
    db_url: str,
    active_only: bool = True,
    max_bank_per_member: int = 0,
) -> GalleryData:
    """
    Loads:
      - members
      - member_embeddings (per camera)

    Builds:
      - global gallery (all cameras)
      - per-camera galleries (camera_id-specific), with fallback possible.
    """
    Base = declarative_base()

    class MemberRow(Base):
        __tablename__ = "members"
        id = Column(Integer, primary_key=True)
        member_number = Column(String(16))
        first_name = Column(String(64))
        last_name = Column(String(64))
        is_active = Column(Boolean)

    class MemberEmbeddingRow(Base):
        __tablename__ = "member_embeddings"
        id = Column(BigInteger, primary_key=True)

        member_id = Column(Integer, ForeignKey("members.id"))
        camera_id = Column(Integer, ForeignKey("cameras.id"))

        if Vector is not None:
            body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            back_body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
        else:
            body_embedding = Column(ARRAY(Float), nullable=True)
            face_embedding = Column(ARRAY(Float), nullable=True)
            back_body_embedding = Column(ARRAY(Float), nullable=True)

        body_embeddings_raw = Column(LargeBinary, nullable=True)
        face_embeddings_raw = Column(LargeBinary, nullable=True)
        back_body_embeddings_raw = Column(LargeBinary, nullable=True)

        last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True, server_default=func.now())

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)

    member_meta: dict[int, dict[str, Any]] = {}
    agg_global: dict[int, _Agg] = defaultdict(lambda: _Agg([], [], [], []))
    agg_by_cam: dict[int, dict[int, _Agg]] = defaultdict(lambda: defaultdict(lambda: _Agg([], [], [], [])))

    with Session() as session:
        stmt = (
            select(
                MemberRow.id,
                MemberRow.member_number,
                MemberRow.first_name,
                MemberRow.last_name,
                MemberRow.is_active,
                MemberEmbeddingRow.camera_id,
                MemberEmbeddingRow.body_embedding,
                MemberEmbeddingRow.face_embedding,
                MemberEmbeddingRow.back_body_embedding,
                MemberEmbeddingRow.body_embeddings_raw,
                MemberEmbeddingRow.face_embeddings_raw,
                MemberEmbeddingRow.back_body_embeddings_raw,
            )
            .select_from(MemberRow)
            .join(MemberEmbeddingRow, MemberEmbeddingRow.member_id == MemberRow.id, isouter=False)
        )

        rows = session.execute(stmt).all()

        raw_loaded_body: dict[int, int] = defaultdict(int)
        raw_loaded_face: dict[int, int] = defaultdict(int)
        raw_loaded_body_cam: dict[tuple[int, int], int] = defaultdict(int)
        raw_loaded_face_cam: dict[tuple[int, int], int] = defaultdict(int)

        for r in rows:
            member_id = int(r.id)
            is_active = bool(r.is_active) if r.is_active is not None else True
            if active_only and (not is_active):
                continue

            cam_id = int(r.camera_id)

            if member_id not in member_meta:
                member_meta[member_id] = {
                    "member_number": r.member_number,
                    "first_name": r.first_name,
                    "last_name": r.last_name,
                    "is_active": is_active,
                }

            b = _as_vec512(r.body_embedding)
            f = _as_vec512(r.face_embedding)
            bb = _as_vec512(r.back_body_embedding)

            if b is not None:
                agg_global[member_id].body_vecs.append(b)
                agg_by_cam[cam_id][member_id].body_vecs.append(b)

            if bb is not None:
                agg_global[member_id].body_vecs.append(bb)
                agg_by_cam[cam_id][member_id].body_vecs.append(bb)

            if f is not None:
                agg_global[member_id].face_vecs.append(f)
                agg_by_cam[cam_id][member_id].face_vecs.append(f)

            max_bank = int(max_bank_per_member or 0)

            if r.body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[member_id] < max_bank):
                bank = decode_bank_blob(r.body_embeddings_raw)
                if bank is not None and bank.size > 0:
                    remain = max_bank - raw_loaded_body[member_id] if max_bank > 0 else bank.shape[0]
                    take = bank[: max(0, remain)] if max_bank > 0 else bank
                    if take.size > 0:
                        agg_global[member_id].body_banks.append(take)
                        raw_loaded_body[member_id] += int(take.shape[0])

                    key = (member_id, cam_id)
                    if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
                        remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
                        take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
                        if take2.size > 0:
                            agg_by_cam[cam_id][member_id].body_banks.append(take2)
                            raw_loaded_body_cam[key] += int(take2.shape[0])

            if r.face_embeddings_raw and (max_bank <= 0 or raw_loaded_face[member_id] < max_bank):
                bank = decode_bank_blob(r.face_embeddings_raw)
                if bank is not None and bank.size > 0:
                    remain = max_bank - raw_loaded_face[member_id] if max_bank > 0 else bank.shape[0]
                    take = bank[: max(0, remain)] if max_bank > 0 else bank
                    if take.size > 0:
                        agg_global[member_id].face_banks.append(take)
                        raw_loaded_face[member_id] += int(take.shape[0])

                    key = (member_id, cam_id)
                    if max_bank <= 0 or raw_loaded_face_cam[key] < max_bank:
                        remain2 = max_bank - raw_loaded_face_cam[key] if max_bank > 0 else bank.shape[0]
                        take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
                        if take2.size > 0:
                            agg_by_cam[cam_id][member_id].face_banks.append(take2)
                            raw_loaded_face_cam[key] += int(take2.shape[0])

            if r.back_body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[member_id] < max_bank):
                bank = decode_bank_blob(r.back_body_embeddings_raw)
                if bank is not None and bank.size > 0:
                    remain = max_bank - raw_loaded_body[member_id] if max_bank > 0 else bank.shape[0]
                    take = bank[: max(0, remain)] if max_bank > 0 else bank
                    if take.size > 0:
                        agg_global[member_id].body_banks.append(take)
                        raw_loaded_body[member_id] += int(take.shape[0])

                    key = (member_id, cam_id)
                    if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
                        remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
                        take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
                        if take2.size > 0:
                            agg_by_cam[cam_id][member_id].body_banks.append(take2)
                            raw_loaded_body_cam[key] += int(take2.shape[0])

    def _build_people_from_agg(agg: dict[int, _Agg]) -> tuple[list[PersonEntry], FaceGallery]:
        people: list[PersonEntry] = []
        face_names: list[str] = []
        face_vecs: list[np.ndarray] = []

        for member_id, a in agg.items():
            meta = member_meta.get(member_id, {})
            name = _member_label(meta.get("member_number"), meta.get("first_name"), meta.get("last_name"), member_id)

            bank_parts = []
            for bnk in a.body_banks:
                if bnk is not None and bnk.size > 0:
                    bank_parts.append(np.asarray(bnk, dtype=np.float32))
            body_bank = np.concatenate(bank_parts, axis=0) if bank_parts else None
            if body_bank is not None:
                body_bank = l2_normalize_rows(body_bank)

            body_cent = None
            if a.body_vecs:
                body_cent = l2_normalize(np.mean(np.stack(a.body_vecs, axis=0), axis=0))
            elif body_bank is not None and body_bank.size > 0:
                body_cent = l2_normalize(np.mean(body_bank, axis=0))

            face_cent = None
            if a.face_vecs:
                face_cent = l2_normalize(np.mean(np.stack(a.face_vecs, axis=0), axis=0))
            else:
                fbank_parts = []
                for bnk in a.face_banks:
                    if bnk is not None and bnk.size > 0:
                        fbank_parts.append(np.asarray(bnk, dtype=np.float32))
                if fbank_parts:
                    fb = l2_normalize_rows(np.concatenate(fbank_parts, axis=0))
                    face_cent = l2_normalize(np.mean(fb, axis=0))

            if body_bank is None and body_cent is not None:
                body_bank = body_cent.reshape(1, -1).astype(np.float32)

            people.append(
                PersonEntry(
                    user_id=int(member_id),
                    name=str(name),
                    body_bank=body_bank,
                    body_centroid=body_cent,
                    face_centroid=face_cent,
                )
            )

            if face_cent is not None:
                face_names.append(str(name))
                face_vecs.append(face_cent.astype(np.float32))

        face_mat = (
            l2_normalize_rows(np.stack(face_vecs, axis=0))
            if face_vecs
            else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        )
        return people, FaceGallery(face_names, face_mat)

    gd = GalleryData()
    gd.people_global, gd.face_global = _build_people_from_agg(agg_global)

    for cam_id, member_map in agg_by_cam.items():
        ppl, fg = _build_people_from_agg(member_map)
        gd.people_by_cam[int(cam_id)] = ppl
        gd.face_by_cam[int(cam_id)] = fg

    body_count = sum(1 for p in gd.people_global if p.body_bank is not None and p.body_bank.size > 0)
    print(f"[DB] Loaded identities (global): {len(gd.people_global)} (body={body_count}, face={len(gd.face_global.names)})")
    return gd


class GalleryManager:
    """Thread-safe DB gallery with optional periodic refresh."""

    def __init__(self, args):
        if not getattr(args, "db_url", ""):
            raise RuntimeError("db_url is required")
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self.data: GalleryData = GalleryData()
        self.last_load_ts: float = 0.0
        self.load(args)

    def load(self, args) -> None:
        active_only = not bool(getattr(args, "db_include_inactive", False))
        max_bank = int(getattr(args, "db_max_bank", 0) or 0)
        data = build_galleries_from_db(args.db_url, active_only=active_only, max_bank_per_member=max_bank)
        with self._lock:
            self.data = data
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

    def snapshot(self, camera_id: Optional[int] = None) -> tuple[list[PersonEntry], FaceGallery]:
        """Returns per-camera gallery when available; otherwise falls back to global."""
        with self._lock:
            data = self.data
            if camera_id is None:
                return data.people_global, data.face_global

            cam_id = int(camera_id)
            ppl = data.people_by_cam.get(cam_id)
            fg = data.face_by_cam.get(cam_id)
            if ppl is not None and fg is not None and (not fg.is_empty()):
                return ppl, fg

            return data.people_global, data.face_global


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
    """Intersection over AREA(inner). Good for linking small face box to big person box."""
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
    """
    Exponentially decayed per-name score accumulator.

    NOTE: Body votes are only accepted when they agree with the face label (see caller).
    """
    entry = state.setdefault(
        tid,
        {
            "scores": defaultdict(float),
            "last": "",
            "ttl": 0,
            "face_vis_ttl": 0,
            "last_face_label": "",
            "last_face_sim": 0.0,
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
def init_face_engine(
    use_face: bool,
    device: str,
    face_model: str,
    det_w: int,
    det_h: int,
    face_provider: str,
    ort_log: bool,
):
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


def extract_body_embeddings_batch(
    extractor,
    crops_bgr: List[np.ndarray],
    device_is_cuda: bool,
    use_half: bool,
) -> Optional[np.ndarray]:
    """Batched TorchReID embedding extraction. Returns [N,512] normalized."""
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


def best_face_label_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> tuple[str | None, float, float]:
    """Returns (best_label, best_sim, second_sim)."""
    if emb is None or face_gallery is None or face_gallery.is_empty():
        return None, 0.0, 0.0
    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, 0.0, 0.0

    sims = face_gallery.mat @ q
    if sims.size == 0:
        return None, 0.0, 0.0

    if sims.size == 1:
        return face_gallery.names[0], float(sims[0]), 0.0

    idxs = np.argpartition(sims, -2)[-2:]
    i1, i2 = int(idxs[0]), int(idxs[1])
    if sims[i2] > sims[i1]:
        i1, i2 = i2, i1
    best_idx, second_idx = i1, i2
    return face_gallery.names[best_idx], float(sims[best_idx]), float(sims[second_idx])


# ----------------------------
# Threaded readers
# ----------------------------
class AdaptiveQueueStream:
    """Ordered reader with bounded queue; drops oldest when full."""

    def __init__(self, src: str, queue_size: int, rtsp_transport: str, use_opencv: bool = True):
        self.src = src
        src_use = src
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            sep = "&" if "?" in src_use else "?"
            src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"

        self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG) if use_opencv else cv2.VideoCapture(src_use)
        self.ok = self.cap.isOpened()
        if not self.ok:
            print(f"[WARN] cannot open source {src}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(queue_size)))
        except Exception:
            pass

        self.q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.stop_flag = threading.Event()
        self.dropped = 0
        self.read_dropped = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _drop_oldest(self) -> None:
        try:
            _ = self.q.get_nowait()
            self.dropped += 1
        except queue.Empty:
            return

    def _loop(self):
        while not self.stop_flag.is_set() and self.ok:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            item = (frame, time.time())
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self._drop_oldest()
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    self._drop_oldest()

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
        return bool(self.ok)

    def release(self):
        self.stop_flag.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


class LatestStream:
    """Latest-frame reader (keeps only 1 frame)."""

    def __init__(self, src: str, rtsp_buffer: int, decode_skip: int, rtsp_transport: str):
        self.src = src
        src_use = src
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            sep = "&" if "?" in src_use else "?"
            src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"
        self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG)
        self.ok = self.cap.isOpened()
        if not self.ok:
            print(f"[WARN] cannot open source {src}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(rtsp_buffer)))
        except Exception:
            pass
        self.decode_skip = max(0, int(decode_skip))
        self.q = queue.Queue(maxsize=1)
        self.stop_flag = threading.Event()
        self.dropped = 0
        self.read_dropped = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        idx = 0
        while not self.stop_flag.is_set() and self.ok:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if self.decode_skip > 0:
                if (idx % (self.decode_skip + 1)) != 0:
                    idx += 1
                    continue
                idx += 1
            item = (frame, time.time())
            if not self.q.empty():
                try:
                    _ = self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        try:
            frame, ts = self.q.get(timeout=0.02)
            return True, frame, float(ts)
        except queue.Empty:
            return False, None, 0.0

    def qsize(self) -> int:
        return 1 if not self.q.empty() else 0

    def is_opened(self) -> bool:
        return bool(self.ok)

    def release(self):
        self.stop_flag.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


class FFmpegStream:
    """FFmpeg pipe reader (ordered). NVDEC optional."""

    def __init__(
        self,
        src: str,
        queue_size: int,
        rtsp_transport: str,
        use_cuda: bool,
        force_w: int = 0,
        force_h: int = 0,
    ):
        self.src = _sanitize_rtsp_url(src) if isinstance(src, str) and src.lower().startswith("rtsp") else src
        self.q = queue.Queue(maxsize=max(1, int(queue_size)))
        self.stop_flag = threading.Event()
        self.proc = None
        self.ok = False
        self.dropped = 0
        self.read_dropped = 0

        probed_w = probed_h = 0
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            if cap.isOpened():
                probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            cap.release()
        except Exception:
            pass

        scheme = ""
        try:
            scheme = urlsplit(self.src).scheme.lower()
        except Exception:
            pass

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", self.src, "-an", "-dn", "-sn", "-vsync", "0"]
        if scheme == "rtsp":
            cmd = [
                "ffmpeg",
                "-rtsp_transport",
                rtsp_transport,
                "-flags",
                "+genpts",
                "-fflags",
                "+genpts",
                "-use_wallclock_as_timestamps",
                "1",
                "-i",
                self.src,
                "-an",
                "-dn",
                "-sn",
                "-vsync",
                "0",
            ]
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]

        out_w = int(force_w) if force_w and force_w > 0 else int(probed_w)
        out_h = int(force_h) if force_h and force_h > 0 else int(probed_h)
        if out_w > 0 and out_h > 0:
            cmd += ["-vf", f"scale={out_w}:{out_h}"]
        else:
            out_w, out_h = 1280, 720
            cmd += ["-vf", "scale=1280:720"]

        if use_cuda:
            hw = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-vf",
                f"hwdownload,format=bgr24,scale={out_w}:{out_h}",
            ]
            if "-vf" in cmd:
                i = cmd.index("-vf")
                cmd.pop(i)
                prev = cmd.pop(i)
                hw[-1] = f"hwdownload,format=bgr24,{prev},scale={out_w}:{out_h}" if "scale=" in prev else hw[-1]
            cmd += hw

        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

        self.cmd = cmd
        self.width, self.height = int(out_w), int(out_h)
        self.frame_bytes = self.width * self.height * 3

        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            self.ok = True
        except Exception as e:
            print(f"[FFMPEG] start failed: {e}")
            self.ok = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log_thread = threading.Thread(target=self._log_stderr, daemon=True)
        self.log_thread.start()

    def _log_stderr(self):
        if not self.proc or not self.proc.stderr:
            return
        try:
            for _line in iter(self.proc.stderr.readline, b""):
                if not _line:
                    break
        except Exception:
            pass

    def _drop_oldest(self) -> None:
        try:
            _ = self.q.get_nowait()
            self.dropped += 1
        except queue.Empty:
            return

    def _loop(self):
        if not self.proc or not self.proc.stdout or not self.ok:
            return
        fb = int(self.frame_bytes)
        w, h = int(self.width), int(self.height)
        while not self.stop_flag.is_set():
            buf = self.proc.stdout.read(fb)
            if not buf or len(buf) < fb:
                time.sleep(0.001)
                continue
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            item = (frame, time.time())
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self._drop_oldest()
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    self._drop_oldest()

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
        return bool(self.ok)

    def release(self):
        self.stop_flag.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.terminate()
                self.proc.kill()
        except Exception:
            pass


# ----------------------------
# Frame buffer for MJPEG (single latest frame + cached JPEG)
# ----------------------------
class FrameBuffer:
    """
    Stores latest annotated frame + metadata.

    Compatibility with mjpeg.py:
      - wait_for_seq(last_seq, timeout) -> (frame, ts, meta, seq)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._frame_bgr: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._meta: Dict[str, Any] = {}
        self._seq: int = 0

        self._jpeg: Optional[bytes] = None
        self._jpeg_ts: float = 0.0
        self._encode_lock = threading.Lock()

        self._clients = 0

    def add_client(self) -> None:
        with self._lock:
            self._clients += 1

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def set(self, frame_bgr: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._cond:
            self._frame_bgr = frame_bgr
            self._ts = time.time()
            self._meta = dict(meta or {})
            self._seq += 1
            self._jpeg = None
            self._jpeg_ts = 0.0
            self._cond.notify_all()

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame_bgr

    def get_meta(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._meta)

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame_bgr, float(self._ts)

    def get_seq(self) -> int:
        with self._lock:
            return int(self._seq)

    def wait_for_seq(
        self,
        last_seq: int,
        timeout: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
        try:
            last_seq_i = int(last_seq)
        except Exception:
            last_seq_i = -1

        with self._cond:
            if self._seq == last_seq_i:
                self._cond.wait(timeout=float(timeout))

            frame = self._frame_bgr
            ts = float(self._ts)
            meta = dict(self._meta)
            seq = int(self._seq)

        return frame, ts, meta, seq

    def wait_jpeg(
        self,
        last_ts: float,
        timeout: float,
        jpeg_quality: int = 80,
    ) -> Tuple[Optional[bytes], float]:
        with self._cond:
            if self._ts <= float(last_ts):
                self._cond.wait(timeout=float(timeout))
            ts = float(self._ts)
            frame = self._frame_bgr
            clients = int(self._clients)

        if frame is None or ts <= 0:
            return None, 0.0
        if clients <= 0:
            return None, ts

        with self._lock:
            if self._jpeg is not None and self._jpeg_ts == ts:
                return self._jpeg, ts

        with self._encode_lock:
            with self._lock:
                if self._jpeg is not None and self._jpeg_ts == ts:
                    return self._jpeg, ts
                frame_ref = self._frame_bgr
                ts_copy = float(self._ts)

            if frame_ref is None or ts_copy <= 0:
                return None, 0.0

            q = int(max(30, min(95, int(jpeg_quality))))
            ok, enc = cv2.imencode(".jpg", frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return None, ts_copy

            jpg = enc.tobytes()

            with self._lock:
                if float(self._ts) == ts_copy:
                    self._jpeg = jpg
                    self._jpeg_ts = ts_copy

            return jpg, ts_copy


RenderedFrame = FrameBuffer  # compatibility alias


# ----------------------------
# CSV SUMMARY REPORT (unchanged)
# ----------------------------
@dataclass
class _ActiveSession:
    start_ts: float
    last_seen_ts: float


class SummaryReport:
    """
    Builds a summary CSV:
      Member | c1 | c2 | ... | Total time
    Each camera cell contains multiple lines:
      L1 - HH:MM:SS to HH:MM:SS
    """

    def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")
        self._lock = threading.Lock()
        self._disabled = False

        self._first_seen: Dict[str, float] = {}
        self._active: Dict[Tuple[str, int], _ActiveSession] = {}
        self._logs: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        self._total_seconds: Dict[str, float] = defaultdict(float)

    def stop(self) -> None:
        with self._lock:
            self._disabled = True

    def update(self, cam_id: int, present_names: List[str], ts: float) -> None:
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

                if name not in self._first_seen:
                    self._first_seen[name] = float(ts)

                key = (name, cam_id)
                sess = self._active.get(key)
                if sess is None:
                    self._active[key] = _ActiveSession(start_ts=float(ts), last_seen_ts=float(ts))
                else:
                    sess.last_seen_ts = float(ts)

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
        self._logs[name][int(cam_id)].append((st, en))
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

    def write_csv(self, path: str) -> None:
        self.close_all()

        with self._lock:
            items = list(self._first_seen.items())
            logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
            totals = dict(self._total_seconds)

        items.sort(key=lambda kv: kv[1])
        names_order = [nm for nm, _ in items]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for name in names_order:
                row = [name]
                cam_map = logs.get(name, {})

                for cam_id in range(self.num_cams):
                    entries = cam_map.get(cam_id, [])
                    lines = []
                    for idx, (st, en) in enumerate(entries, start=1):
                        lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
                    row.append("\n".join(lines))

                row.append(self._fmt_total(totals.get(name, 0.0)))
                w.writerow(row)

    def write_csv_snapshot(self, path: str) -> None:
        now_ts = time.time()

        with self._lock:
            if self._disabled:
                return

            self._close_expired_locked(now_ts=now_ts)

            items = list(self._first_seen.items())
            logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
            totals = dict(self._total_seconds)

            for (name, cam_id), sess in self._active.items():
                st = float(sess.start_ts)
                en = float(sess.last_seen_ts)
                logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en))
                totals[name] = totals.get(name, 0.0) + max(0.0, (en - st))

        items.sort(key=lambda kv: kv[1])
        names_order = [nm for nm, _ in items]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for name in names_order:
                row = [name]
                cam_map = logs.get(name, {})

                for cam_id in range(self.num_cams):
                    entries = cam_map.get(cam_id, [])
                    lines = []
                    for idx, (st, en) in enumerate(entries, start=1):
                        lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}")
                    row.append("\n".join(lines))

                row.append(self._fmt_total(totals.get(name, 0.0)))
                w.writerow(row)


# ----------------------------
# Name de-duplication + global multi-camera ownership
# ----------------------------
@dataclass
class DrawItem:
    tid: int
    bbox: Tuple[int, int, int, int]
    name: str
    user_id: int
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
    if not items:
        return []

    groups: Dict[str, List[DrawItem]] = defaultdict(list)
    for it in items:
        if it.name:
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
# NEW: High-throughput ingestion (raw table rotation + batch writes)
# ----------------------------
@dataclass
class DetectionRow:
    data_ts: datetime
    camera_id: int
    member_id: Optional[int]
    guest_temp_id: Optional[str]
    guest_data_vector: Optional[dict]
    match_value: int


def _utc_dt_from_ts(ts: float) -> datetime:
    try:
        return datetime.utcfromtimestamp(float(ts))
    except Exception:
        return datetime.utcnow()


def _align_next_rotation(now_utc: datetime, rotate_seconds: int) -> datetime:
    # Align to rotate boundary from epoch (stable, deterministic)
    sec = int(now_utc.timestamp())
    nxt = ((sec // rotate_seconds) + 1) * rotate_seconds
    return datetime.utcfromtimestamp(nxt)


# ----------------------------
# FIXED: High-throughput ingestion (raw table rotation + batch writes)
# ----------------------------
class RawIngestionManager:
    """
    Optimized ingestion workflow:
    camera detections -> in-memory buffer -> periodic batch insert -> rotating raw tables

    FIXED:
    - Rotation uses epoch time (time.time()).
    - No UTC alignment drift.
    - No infinite rapid table creation.
    """

    def __init__(
        self,
        db_url: str,
        rotate_seconds: int = 300,
        flush_interval: float = 1.0,
        batch_size: int = 500,
        retention_minutes: int = 0,
        enable_cleanup: bool = False,
    ):
        self.db_url = str(db_url)
        self.engine = create_engine(self.db_url, pool_pre_ping=True)

        self.rotate_seconds = int(max(60, rotate_seconds))
        self.flush_interval = float(max(0.1, flush_interval))
        self.batch_size = int(max(1, batch_size))

        self.retention_minutes = int(max(0, retention_minutes))
        self.enable_cleanup = bool(enable_cleanup and self.retention_minutes > 0)

        self._q: "queue.Queue[DetectionRow]" = queue.Queue(maxsize=200000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        self._table_lock = threading.Lock()
        self._cur_table: Optional[str] = None

        # ✅ FIX: epoch-based rotation
        self._next_rotate_epoch = time.time() + self.rotate_seconds

        self._last_cleanup_at: float = 0.0

        self._ensure_pg()
        self._rotate_table(force=True)

        self._thread.start()

    def _ensure_pg(self) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise RuntimeError(f"[INGEST] DB connect failed: {e}")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=3.0)
        except Exception:
            pass
        try:
            self._flush(max_rows=1000000)
        except Exception as e:
            print("[INGEST] final flush failed:", e)

    def push_many(self, rows: List[DetectionRow]) -> None:
        if not rows:
            return
        for r in rows:
            try:
                self._q.put_nowait(r)
            except queue.Full:
                try:
                    _ = self._q.get_nowait()
                except Exception:
                    pass
                try:
                    self._q.put_nowait(r)
                except Exception:
                    pass

    def _make_table_name(self) -> str:
        return f"raw_data_{uuid.uuid4().hex}"

    # ✅ FIXED ROTATION LOGIC
    def _rotate_table(self, force: bool = False) -> None:
        now = time.time()

        if self._cur_table is None:
            force = True

        if not force and now < self._next_rotate_epoch:
            return

        new_table = self._make_table_name()

        ddl = f"""
        CREATE TABLE IF NOT EXISTS {new_table} (
          data_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          camera_id INTEGER NOT NULL CHECK (camera_id >= 0),
          member_id INTEGER CHECK (member_id >= 0),
          guest_temp_id VARCHAR(64),
          guest_data_vector JSONB,
          match_value INTEGER CHECK (match_value >= 0)
        );
        """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(ddl))
        except Exception as e:
            print("[INGEST] table create failed:", e)
            return

        with self._table_lock:
            self._cur_table = new_table
            self._next_rotate_epoch = now + self.rotate_seconds

        print(f"[INGEST] Rotated raw table -> {new_table}")

    def _flush(self, max_rows: int) -> int:
        rows: List[DetectionRow] = []
        for _ in range(max_rows):
            try:
                rows.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not rows:
            return 0

        self._rotate_table(force=False)

        with self._table_lock:
            table = self._cur_table
        if not table:
            return 0

        ins = text(
            f"""
            INSERT INTO {table}
            (data_ts, camera_id, member_id, guest_temp_id, guest_data_vector, match_value)
            VALUES (:data_ts, :camera_id, :member_id, :guest_temp_id,
                    CAST(:guest_data_vector AS JSONB), :match_value)
            """
        )

        payload = []
        for r in rows:
            payload.append(
                {
                    "data_ts": r.data_ts,
                    "camera_id": int(r.camera_id),
                    "member_id": int(r.member_id) if r.member_id is not None else None,
                    "guest_temp_id": str(r.guest_temp_id) if r.guest_temp_id else None,
                    # FIX: serialize dict → JSON string
                    "guest_data_vector": json.dumps(r.guest_data_vector) if r.guest_data_vector is not None else None,
                    "match_value": int(max(0, r.match_value)),
                }
            )
        try:
            with self.engine.begin() as conn:
                conn.execute(ins, payload)
            return len(rows)
        except Exception as e:
            print("[INGEST] batch insert failed:", e)
            return 0

    def _loop(self) -> None:
        last_flush = 0.0

        while not self._stop.is_set():

            # rotation check
            self._rotate_table(force=False)

            qsz = 0
            try:
                qsz = self._q.qsize()
            except Exception:
                pass

            now = time.time()

            if qsz >= self.batch_size or (now - last_flush) >= self.flush_interval:
                flushed = self._flush(max_rows=self.batch_size)
                if flushed > 0:
                    last_flush = now

            time.sleep(0.05)



# ----------------------------
# CLI / args parsing
# ----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "YOLO -> DeepSORT (TorchReID) with DB gallery tagging + Face-only labeling (body supports only).",
        conflict_handler="resolve",
    )

    ap.add_argument("--src", nargs="*", default=[], help="Video sources OR DB camera IDs.")

    # DB
    ap.add_argument("--use-db", action="store_true", help="Enable DB gallery.")
    ap.add_argument("--db-url", default="", help="SQLAlchemy DB URL.")
    ap.add_argument("--db-refresh-seconds", type=float, default=60.0, help="Reload DB gallery every N seconds (0=off).")
    ap.add_argument("--db-max-bank", type=int, default=256, help="Max embeddings per member to load (0=all).")
    ap.add_argument("--db-include-inactive", action="store_true", help="Include inactive members.")

    # YOLO
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")

    # Runtime/perf
    ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
    ap.add_argument("--rtsp-buffer", type=int, default=2)
    ap.add_argument("--decode-skip", type=int, default=0)
    ap.add_argument("--reader", choices=["latest", "adaptive", "ffmpeg"], default="adaptive")
    ap.add_argument("--queue-size", type=int, default=128)
    ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
    ap.add_argument("--ffmpeg-cuda", action="store_true")
    ap.add_argument("--ffmpeg-width", type=int, default=0)
    ap.add_argument("--ffmpeg-height", type=int, default=0)
    ap.add_argument("--resize", type=int, nargs=2, default=[0, 0])

    # Adaptive skipping
    ap.add_argument("--max-queue-age-ms", type=int, default=1000)
    ap.add_argument("--max-drain-per-cycle", type=int, default=32)

    # DeepSORT / TorchReID
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-deepsort", action="store_true")
    g.add_argument("--use-deepsort", action="store_true")

    ap.add_argument("--reid-model", default="osnet_x0_25")
    ap.add_argument("--reid-weights", default="")
    ap.add_argument("--reid-batch-size", type=int, default=16)

    ap.add_argument("--max-age", type=int, default=15)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--nn-budget", type=int, default=200)
    ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
    ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)

    # Body matching (support only)
    ap.add_argument("--gallery-thresh", type=float, default=0.70)
    ap.add_argument("--gallery-gap", type=float, default=0.08)
    ap.add_argument("--reid-topk", type=int, default=3)
    ap.add_argument("--min-box-wh", type=int, default=40)

    # Face (main identity)
    ap.add_argument("--use-face", action="store_true")
    ap.add_argument("--face-model", default="buffalo_l")
    ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
    ap.add_argument("--face-thresh", type=float, default=0.40)
    ap.add_argument("--face-gap", type=float, default=0.05)
    ap.add_argument("--face-every-n", type=int, default=1)
    ap.add_argument("--face-hold-frames", type=int, default=2)
    ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--ort-log", action="store_true")

    ap.add_argument("--face-iou-link", type=float, default=0.35)
    ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
    ap.add_argument("--face-center-in-person", action="store_true")

    ap.add_argument("--min-face-px", type=int, default=24)
    ap.add_argument("--min-face-area-ratio", type=float, default=0.006)
    ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70)
    ap.add_argument("--face-strong-thresh", type=float, default=0.50)

    # Identity smoothing
    ap.add_argument("--name-decay", type=float, default=0.85)
    ap.add_argument("--name-min-score", type=float, default=0.60)
    ap.add_argument("--name-margin", type=float, default=0.30)
    ap.add_argument("--name-ttl", type=int, default=20)
    ap.add_argument("--name-face-weight", type=float, default=1.2)
    ap.add_argument("--name-body-weight", type=float, default=0.5)

    # Drawing/ghost control
    ap.add_argument("--draw-only-matched", action="store_true")
    ap.add_argument("--min-det-conf", type=float, default=0.45)
    ap.add_argument("--iou-max-miss", type=int, default=5)

    # Duplicate-name suppression / multi-camera ownership
    ap.add_argument("--allow-duplicate-names", action="store_true")
    ap.add_argument("--dedup-iou", type=float, default=0.0)
    ap.add_argument("--no-global-unique-names", action="store_true")
    ap.add_argument("--global-hold-seconds", type=float, default=0.5)
    ap.add_argument("--global-switch-margin", type=float, default=0.02)
    ap.add_argument("--show-global-id", action="store_true")

    # Output & view (accepted for compatibility; service ignores)
    ap.add_argument("--show", action="store_true")

    # CSV report
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--csv", default="detections_summary.csv")
    ap.add_argument("--report-gap-seconds", type=float, default=2.0)
    ap.add_argument("--report-time-format", default="%H:%M:%S")

    # FPS overlays
    ap.add_argument("--overlay-fps", action="store_true")

    # ✅ NEW ingestion args
    ap.add_argument("--enable-ingest", action="store_true", help="Enable raw-table ingestion logging.")
    ap.add_argument("--raw-rotate-seconds", type=int, default=300, help="Rotate raw table every N seconds (default 300=5min).")
    ap.add_argument("--ingest-flush-interval", type=float, default=1.0, help="Flush buffered detections every N seconds.")
    ap.add_argument("--ingest-batch-size", type=int, default=500, help="Batch insert size for detections.")
    ap.add_argument("--raw-retention-minutes", type=int, default=0, help="If >0 and --raw-cleanup, drop raw tables older than this.")
    ap.add_argument("--raw-cleanup", action="store_true", help="Enable dropping old raw tables (uses raw_data_registry).")
    ap.add_argument("--store-guest-embeddings", action="store_true",
                    help="If set, unknown guest_data_vector stores large embedding arrays (JSONB). Default OFF to reduce DB load.")

    args = ap.parse_args(argv)
    return args


def parse_pipeline_args(pipeline_args: str | None) -> argparse.Namespace:
    s = str(pipeline_args or "").strip()
    argv = shlex.split(s) if s else []
    return parse_args(argv)


# ----------------------------
# Face-only process_one_frame + logging outputs
# ----------------------------
def process_one_frame(
    frame_idx: int,
    frame_bgr: np.ndarray,
    sid: int,  # internal stream id for gating; not necessarily DB camera_id
    camera_id_for_logging: int,  # actual camera_id to store in raw table
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    people: list[PersonEntry],
    reid_extractor,
    face_app,
    face_gallery: FaceGallery,
    identity_state: dict,
    device_is_cuda: bool,
    global_owner: Optional[GlobalNameOwner] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[DetectionRow]]:
    """
    RULES (your requirement):
      ✅ Draw ONLY KNOWN (face recognized + stable)
      ✅ Unknown persons: NO bounding box at all
      ✅ Body never labels alone (support only)

    NEW:
      ✅ Log every tracked person (known/unknown) as DetectionRow per frame (buffered & batch inserted)
    """

    rw, rh = int(args.resize[0]), int(args.resize[1])
    if rw > 0 and rh > 0:
        frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)

    H, W = frame_bgr.shape[:2]

    # Map name -> member_id
    name_to_uid: Dict[str, int] = {}
    try:
        for p in people:
            name_to_uid[str(p.name)] = int(p.user_id)
    except Exception:
        name_to_uid = {}

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
                keep = cls == 0  # person
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

                emb = extract_face_embedding(f)
                if emb is None:
                    continue
                emb = l2_normalize(np.asarray(emb, dtype=np.float32))
                flabel, fsim, fsecond = best_face_label_top2(emb, face_gallery)
                if flabel is None:
                    continue

                gap = float(fsim - fsecond)
                if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
                    recognized_faces.append(
                        {
                            "bbox": (fx1, fy1, fx2, fy2),
                            "label": str(flabel),
                            "sim": float(fsim),
                            "second": float(fsecond),
                            "gap": float(gap),
                            # store emb only if guest embedding mode is enabled (used for unknown if desired)
                            "emb": emb if bool(getattr(args, "store_guest_embeddings", False)) else None,
                        }
                    )
        except Exception as e:
            print(f"[SRC {sid}] FaceAnalysis error:", e)

    out = frame_bgr.copy()
    tracks_info: List[Dict[str, Any]] = []

    # Build per-track info + face-link (recognized faces only)
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

        face_label, face_sim, face_gap = "", 0.0, 0.0
        face_hit = False
        face_emb = None

        if recognized_faces:
            t_xyxy = (float(x1), float(y1), float(x2), float(y2))
            p_area = float(max(1.0, (x2 - x1) * (y2 - y1)))

            best_idx, best_score = -1, -1e9
            for idx, fm in enumerate(recognized_faces):
                fbox = fm["bbox"]

                cx = 0.5 * (float(fbox[0]) + float(fbox[2]))
                cy = 0.5 * (float(fbox[1]) + float(fbox[3]))
                if args.face_center_in_person and not _point_in_xyxy(cx, cy, t_xyxy):
                    continue

                top_limit = float(y1) + float(args.face_center_y_max_ratio) * float(max(1, (y2 - y1)))
                if cy > top_limit:
                    continue

                link = ioa_xyxy(fbox, t_xyxy) if args.face_link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)
                if link < float(args.face_iou_link):
                    continue

                if float(args.min_face_area_ratio) > 0:
                    f_area = float(max(1.0, (float(fbox[2]) - float(fbox[0])) * (float(fbox[3]) - float(fbox[1]))))
                    if (f_area / p_area) < float(args.min_face_area_ratio):
                        continue

                score = float(link) + 0.25 * float(fm["sim"])
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                fm = recognized_faces[best_idx]
                face_label = str(fm["label"])
                face_sim = float(fm["sim"])
                face_gap = float(fm["gap"])
                face_hit = True
                face_emb = fm.get("emb", None)

        entry = identity_state.setdefault(
            tid,
            {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
        )

        dec = 1 if do_face else 0
        entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - dec)

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
                "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
                "last_face_sim": float(entry.get("last_face_sim", 0.0)),
                "face_emb": face_emb,  # may be None unless store_guest_embeddings
            }
        )

    # Body support: compute only for tracks with recognized face hit this frame
    face_hit_tracks = [r for r in tracks_info if r["face_hit"] and r["face_label"]]
    body_by_tid: Dict[int, Tuple[str, float, float]] = {}

    if face_hit_tracks and (reid_extractor is not None) and people:
        crops: List[np.ndarray] = []
        tids: List[int] = []
        for r in face_hit_tracks:
            x1, y1, x2, y2 = r["bbox"]
            crop = frame_bgr[y1:y2, x1:x2]
            crops.append(crop)
            tids.append(int(r["tid"]))

        bs = max(1, int(getattr(args, "reid_batch_size", 16)))
        for start in range(0, len(crops), bs):
            chunk = crops[start: start + bs]
            chunk_tids = tids[start: start + bs]
            feats = extract_body_embeddings_batch(reid_extractor, chunk, device_is_cuda=device_is_cuda, use_half=bool(args.half))
            if feats is None:
                continue
            for tid, emb in zip(chunk_tids, feats):
                blabel, bsim, bsecond = best_body_label_from_emb(emb, people, topk=max(1, int(args.reid_topk)))
                if blabel is None:
                    continue
                body_by_tid[int(tid)] = (str(blabel), float(bsim), float(bsecond))

    draw_candidates: List[DrawItem] = []

    # NEW: detection rows to return
    det_rows: List[DetectionRow] = []
    # guest temp ids stable per (camera_id, tid)
    # store in identity_state so it persists across frames
    # identity_state[tid]["guest_temp_id"] = "guest_{cameraid}_{tid}" (or uuid style)
    # (tid uniqueness is per stream; include camera id)
    now_dt = datetime.utcnow()

    for r in tracks_info:
        tid = int(r["tid"])
        x1, y1, x2, y2 = r["bbox"]

        face_hit = bool(r["face_hit"])
        face_label = str(r["face_label"])
        face_sim = float(r["face_sim"])
        last_face_sim = float(r.get("last_face_sim", 0.0))

        candidates: List[Tuple[str, float, str]] = []

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
                            candidates = []

        entry = identity_state.setdefault(
            tid,
            {"scores": defaultdict(float), "last": "", "ttl": 0, "face_vis_ttl": 0, "last_face_label": "", "last_face_sim": 0.0},
        )

        if (not do_face) and (not candidates):
            stable_name = str(entry.get("last", "") or "")
            stable_score = float(entry.get("scores", {}).get(stable_name, 0.0)) if stable_name else 0.0
        else:
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

        # "known shown" rule:
        show_label = bool(stable_name) and (int(entry.get("face_vis_ttl", 0)) > 0)

        # ---- NEW: log row for EVERY track (known or unknown) ----
        if "guest_temp_id" not in entry:
            entry["guest_temp_id"] = f"guest_{int(camera_id_for_logging)}_{tid}"
        guest_temp_id = str(entry.get("guest_temp_id", "")) or None

        # match_value rule:
        # - known: face similarity * 100
        # - unknown: 0 (or det_conf*100); here use det_conf if present else 0
        det_conf = r.get("det_conf", None)
        mv_known = int(max(0, min(100, round(float(face_sim if face_hit else entry.get("last_face_sim", last_face_sim)) * 100))))
        mv_unknown = int(max(0, min(100, round(float(det_conf) * 100)))) if det_conf is not None else 0
        match_value = mv_known if show_label else mv_unknown

        member_id = int(name_to_uid.get(stable_name, -1)) if show_label else None
        if member_id is not None and member_id < 0:
            member_id = None

        guest_data_vector = None
        if not show_label:
            # Unknown: store compact JSONB by default.
            guest_data_vector = {
                "tid": tid,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": float(det_conf) if det_conf is not None else None,
                "last_face_sim": float(entry.get("last_face_sim", 0.0)),
                "face_ran": bool(do_face),
            }
            # optionally store face embedding when enabled (HEAVY!)
            if bool(getattr(args, "store_guest_embeddings", False)) and r.get("face_emb", None) is not None:
                emb = np.asarray(r["face_emb"], dtype=np.float32).reshape(-1)
                if emb.size == EXPECTED_DIM and np.isfinite(emb).all():
                    guest_data_vector["face_embedding"] = emb.astype(float).tolist()

        det_rows.append(
            DetectionRow(
                data_ts=now_dt,
                camera_id=int(camera_id_for_logging),
                member_id=int(member_id) if member_id is not None else None,
                guest_temp_id=None if member_id is not None else guest_temp_id,
                guest_data_vector=None if member_id is not None else guest_data_vector,
                match_value=int(max(0, match_value)),
            )
        )

        # ---- Drawing candidates only if KNOWN ----
        if not show_label:
            continue

        disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))
        uid = int(name_to_uid.get(stable_name, -1))

        draw_candidates.append(
            DrawItem(
                tid=tid,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                name=str(stable_name),
                user_id=uid,
                face_sim=float(disp_face_sim),
                stable_score=float(stable_score),
                det_conf=r.get("det_conf", None),
                face_hit=bool(face_hit),
            )
        )

    present_names = sorted({it.name for it in draw_candidates if it.name})

    if not bool(getattr(args, "allow_duplicate_names", False)):
        draw_final = deduplicate_draw_items(draw_candidates, iou_thresh=float(getattr(args, "dedup_iou", 0.0)))
    else:
        draw_final = draw_candidates

    if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
        gated: List[DrawItem] = []
        for it in draw_final:
            score = float(it.face_sim)
            if global_owner.allow(it.name, sid=int(sid), score=score):
                gated.append(it)
        draw_final = gated

    shown = 0
    for it in draw_final:
        x1, y1, x2, y2 = it.bbox
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if bool(getattr(args, "show_global_id", False)) and it.user_id >= 0:
            label_txt = f"{it.name} [{it.user_id}]"
        else:
            label_txt = f"{it.name}"

        if it.face_hit:
            label_txt = f"{label_txt} | F {float(it.face_sim):.2f}"

        cv2.putText(out, label_txt, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        shown += 1

    meta = {
        "tracks": int(len(tracks_info)),
        "shown": int(shown),
        "faces_recognized": int(len(recognized_faces)),
        "do_face": bool(do_face),
        "present_names": present_names,
        "camera_id": int(camera_id_for_logging),
    }
    return out, meta, det_rows


# ----------------------------
# Processor thread (now logs detections via ingestion manager)
# ----------------------------
def processor_thread(
    sid: int,
    camera_id_for_gallery: Optional[int],
    camera_id_for_logging: int,
    vs,
    out_buf: FrameBuffer,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    gallery_mgr: GalleryManager,
    reid_extractor,
    face_app,
    global_owner: Optional[GlobalNameOwner],
    report: Optional[SummaryReport],
    report_cam_index: Optional[int],
    ingest: Optional[RawIngestionManager],
    stop_event: threading.Event,
):
    frame_idx = 0
    identity_state: dict[int, dict] = {}

    device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10

    while not stop_event.is_set():
        ok, frame, ts_cap = vs.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        # Drop stale frames policy
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
            people, face_gallery = gallery_mgr.snapshot(camera_id=camera_id_for_gallery)

            out, meta, det_rows = process_one_frame(
                frame_idx=frame_idx,
                frame_bgr=frame,
                sid=sid,
                camera_id_for_logging=int(camera_id_for_logging),
                yolo=yolo,
                args=args,
                deep_tracker=deep_tracker,
                iou_tracker=iou_tracker,
                people=people,
                reid_extractor=reid_extractor,
                face_app=face_app,
                face_gallery=face_gallery,
                identity_state=identity_state,
                device_is_cuda=device_is_cuda,
                global_owner=global_owner,
            )

            # ✅ NEW: push detections for ingestion (every person, known/unknown)
            if ingest is not None and det_rows:
                ingest.push_many(det_rows)

            if bool(getattr(args, "overlay_fps", False)):
                now = time.time()
                dt = max(1e-6, now - last_t)
                inst_fps = 1.0 / dt
                fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
                last_t = now
                cv2.putText(out, f"FPS: {fps_ema:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Report update (only known shown)
            if report is not None and report_cam_index is not None:
                names = meta.get("present_names", []) or []
                ts_use = float(ts_cap) if ts_cap else time.time()
                report.update(cam_id=int(report_cam_index), present_names=list(names), ts=ts_use)

            out_buf.set(out, meta=meta)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"[SRC {sid}] Tracks={meta.get('tracks', 0)} KnownShown={meta.get('shown', 0)} Faces={meta.get('faces_recognized', 0)}")

        except Exception as e:
            print(f"[PROC {sid}] error:", e)
            time.sleep(0.001)


# ----------------------------
# Runner (service mode)
# ----------------------------
def _rtsp_url_from_db_camera_ip(ip: str) -> str:
    """
    Builds RTSP URL for DB camera row using env template:
      RTSP_URL_TEMPLATE=rtsp://{username}:{password}@{ip}:{port}/Streaming/channels/{stream}
    Uses env:
      RTSP_USER, RTSP_PASS, RTSP_PORT, RTSP_STREAM, RTSP_SCHEME, RTSP_PATH
    """
    ip = str(ip or "").strip()

    username = os.environ.get("RTSP_USER", "admin")
    password = os.environ.get("RTSP_PASS", "admin")
    port = os.environ.get("RTSP_PORT", "554")
    stream = os.environ.get("RTSP_STREAM", "101")
    scheme = os.environ.get("RTSP_SCHEME", "rtsp")
    path = os.environ.get("RTSP_PATH", "Streaming/channels/")

    user_enc = quote(unquote(username or ""), safe="")
    pass_enc = quote(unquote(password or ""), safe="")

    tmpl = os.environ.get("RTSP_URL_TEMPLATE", "").strip()
    if tmpl:
        try:
            return tmpl.format(
                username=user_enc,
                password=pass_enc,
                ip=ip,
                port=str(port),
                stream=str(stream),
                scheme=str(scheme),
                path=str(path),
            )
        except Exception:
            pass

    path = str(path or "").lstrip("/")
    if not path.endswith("/"):
        path = path + "/"
    return f"{scheme}://{user_enc}:{pass_enc}@{ip}:{port}/{path}{stream}"


class TrackingRunner:
    """
    Service runner.

    Modes:
    - manual sources mode: args.src contains URLs => cam_id = 0..N-1 (pre-started)
    - DB camera mode:
        * if args.src contains numeric IDs => pre-start those camera IDs
        * else => start cameras on-demand when /mjpeg/{cam_id} requested
      cam_id = DB cameras.id
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()
        self._started = False

        self.gallery_mgr: Optional[GalleryManager] = None
        self.global_owner: Optional[GlobalNameOwner] = None

        self.yolo = None
        self.reid_extractor = None
        self.face_app = None

        self.report: Optional[SummaryReport] = None

        # ✅ NEW: ingestion manager
        self.ingest: Optional[RawIngestionManager] = None

        self.stop_event = threading.Event()

        self._manual_states: dict[int, dict[str, Any]] = {}
        self._db_states: dict[int, dict[str, Any]] = {}
        self._camid_to_report_index: dict[int, int] = {}

        self._db_engine = None
        self._Session = None

        self._mode: str = "db"

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True

        args = self.args

        if not args.use_db:
            raise RuntimeError("DB-first pipeline: pass --use-db")
        if not args.db_url:
            raise RuntimeError("--db-url is required")

        src_list = list(getattr(args, "src", []) or [])
        if any(_looks_like_url(s) for s in src_list):
            self._mode = "manual"
        else:
            self._mode = "db"

        args.global_unique_names = not bool(getattr(args, "no_global_unique_names", False))

        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

        gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
        if args.half and not gpu:
            print("[WARN] --half requested but CUDA not available; disabling FP16.")
            args.half = False

        print(f"[INIT] device={args.device} cuda_available={torch.cuda.is_available()} half={args.half}")
        print(f"[INIT] mode={self._mode}")

        # DB gallery
        self.gallery_mgr = GalleryManager(args)

        # DB session (cameras)
        if self._mode == "db":
            self._db_engine = create_engine(args.db_url, pool_pre_ping=True)
            self._Session = sessionmaker(bind=self._db_engine)
            self._build_report_camera_index_map()

        # Global unique names gate
        if bool(getattr(args, "global_unique_names", False)):
            self.global_owner = GlobalNameOwner(
                hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
                switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
            )
            print(f"[INIT] Global unique names: ON (hold={self.global_owner.hold_seconds}s, margin={self.global_owner.switch_margin})")
        else:
            print("[INIT] Global unique names: OFF")

        # YOLO
        if YOLO is not None:
            try:
                weights = args.yolo_weights
                if not Path(weights).exists():
                    print(f"[INIT] {weights} not found, falling back to yolov8n.pt")
                    weights = "yolov8n.pt"
                self.yolo = YOLO(weights)
                if gpu:
                    try:
                        self.yolo.to(args.device)
                    except Exception as e:
                        print("[WARN] YOLO .to(device) failed:", e)
                print("[INIT] YOLO ready")
            except Exception as e:
                print("[ERROR] YOLO load failed:", e)
                self.yolo = None
        else:
            print("[WARN] ultralytics not installed; detection disabled")

        # TorchReID extractor (body support only)
        if TorchreidExtractor is not None:
            try:
                dev = args.device if gpu else "cpu"
                if args.reid_weights and Path(args.reid_weights).exists():
                    self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, model_path=args.reid_weights, device=dev)
                else:
                    self.reid_extractor = TorchreidExtractor(model_name=args.reid_model, device=dev)
                print(f"[INIT] TorchReID ready (model={args.reid_model}, device={dev})")
            except Exception as e:
                print("[WARN] TorchReID init failed; body support disabled:", e)
                self.reid_extractor = None
        else:
            print("[WARN] torchreid not installed; body support disabled")

        # Face
        self.face_app = init_face_engine(
            args.use_face,
            args.device,
            args.face_model,
            int(args.face_det_size[0]),
            int(args.face_det_size[1]),
            face_provider=getattr(args, "face_provider", "auto"),
            ort_log=getattr(args, "ort_log", False),
        )

        if args.use_deepsort and DeepSort is None:
            print("[WARN] --use-deepsort requested but deep-sort-realtime not installed.")

        # Summary report
        if args.save_csv:
            num_cols = self._report_num_cams()
            self.report = SummaryReport(
                num_cams=max(1, int(num_cols)),
                gap_seconds=float(getattr(args, "report_gap_seconds", 2.0)),
                time_format=str(getattr(args, "report_time_format", "%H:%M:%S")),
            )
            print(f"[INIT] Summary CSV report: ON -> {args.csv}")
        else:
            self.report = None

        # ✅ NEW: ingestion init
        if bool(getattr(args, "enable_ingest", False)):
            self.ingest = RawIngestionManager(
                db_url=args.db_url,
                rotate_seconds=int(getattr(args, "raw_rotate_seconds", 300)),
                flush_interval=float(getattr(args, "ingest_flush_interval", 1.0)),
                batch_size=int(getattr(args, "ingest_batch_size", 500)),
                retention_minutes=int(getattr(args, "raw_retention_minutes", 0)),
                enable_cleanup=bool(getattr(args, "raw_cleanup", False)),
            )
            print("[INIT] Ingestion: ON (raw table rotation + batch inserts)")
        else:
            self.ingest = None
            print("[INIT] Ingestion: OFF")

        # Start initial sources
        if self._mode == "manual":
            self._start_manual_sources(src_list)
        else:
            for s in src_list:
                if _is_int_str(s):
                    self.ensure_camera_started(int(s))

        print("[Runner] Background detection ready.")

    # ---------- report mapping ----------
    def _build_report_camera_index_map(self) -> None:
        self._camid_to_report_index = {}
        if self._Session is None:
            return

        Base = declarative_base()

        class CameraRow(Base):
            __tablename__ = "cameras"
            id = Column(Integer, primary_key=True)
            name = Column(String(64))
            ip_address = Column(String(64))
            is_active = Column(Boolean)

        with self._Session() as session:
            stmt = select(CameraRow.id).order_by(CameraRow.id.asc())
            rows = session.execute(stmt).all()
            cam_ids = [int(r.id) for r in rows]

        for idx, cam_id in enumerate(cam_ids):
            self._camid_to_report_index[int(cam_id)] = int(idx)

    def _report_num_cams(self) -> int:
        if self._mode == "manual":
            return len(self._manual_states) if self._manual_states else max(1, len(getattr(self.args, "src", []) or []))
        return max(1, len(self._camid_to_report_index) if self._camid_to_report_index else 1)

    # ---------- manual sources ----------
    def _start_manual_sources(self, src_list: list[str]) -> None:
        args = self.args
        gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())

        for i, raw_src in enumerate(src_list):
            cam_id = int(i)  # API cam_id is index
            src = raw_src.strip() if isinstance(raw_src, str) else raw_src

            vs = self._open_stream(src)

            deep_tracker = self._make_deepsort(gpu=gpu) if self._should_use_deepsort() else None
            iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=0.3)

            buf = FrameBuffer()
            self._manual_states[cam_id] = {
                "cam_id": cam_id,
                "sid": cam_id,
                "src": src,
                "vs": vs,
                "deep": deep_tracker,
                "iou": iou_tracker,
                "buf": buf,
                "thread": None,
            }

            if not vs.is_opened():
                print(f"[SRC {cam_id}] open=False :: {src}")
                continue

            t = threading.Thread(
                target=processor_thread,
                args=(
                    int(cam_id),   # sid
                    None,          # camera_id_for_gallery
                    int(cam_id),   # camera_id_for_logging (manual uses index)
                    vs,
                    buf,
                    self.yolo,
                    args,
                    deep_tracker,
                    iou_tracker,
                    self.gallery_mgr,
                    self.reid_extractor,
                    self.face_app,
                    self.global_owner,
                    self.report,
                    int(cam_id),   # report col index
                    self.ingest,
                    self.stop_event,
                ),
                daemon=True,
            )
            t.start()
            self._manual_states[cam_id]["thread"] = t

            print(f"[SRC {cam_id}] open=True :: {src}")

    # ---------- DB cameras ----------
    def ensure_camera_started(self, camera_id: int) -> bool:
        if self._mode != "db":
            return False

        cam_id = int(camera_id)
        with self._lock:
            if cam_id in self._db_states:
                st = self._db_states[cam_id]
                vs = st.get("vs")
                return bool(vs is not None and vs.is_opened())

        cam = self._get_db_camera(cam_id)
        if cam is None:
            return False

        ip = str(cam.get("ip_address") or "").strip()
        if not ip:
            return False

        src = _rtsp_url_from_db_camera_ip(ip)

        vs = self._open_stream(src)
        opened = vs.is_opened()

        gpu = torch.cuda.is_available() and ("cuda" in str(self.args.device).lower())

        deep_tracker = self._make_deepsort(gpu=gpu) if self._should_use_deepsort() else None
        iou_tracker = IOUTracker(max_miss=max(1, int(self.args.iou_max_miss)), iou_thresh=0.3)

        buf = FrameBuffer()

        sid = self._camid_to_report_index.get(cam_id, cam_id)
        report_idx = self._camid_to_report_index.get(cam_id, None)

        with self._lock:
            self._db_states[cam_id] = {
                "cam_id": cam_id,
                "sid": int(sid),
                "camera": cam,
                "src": src,
                "vs": vs,
                "deep": deep_tracker,
                "iou": iou_tracker,
                "buf": buf,
                "thread": None,
                "started_at": time.time(),
            }

        print(f"[SRC {cam_id}] open={opened} :: {src}")

        if not opened:
            return False

        t = threading.Thread(
            target=processor_thread,
            args=(
                int(sid),    # sid (internal)
                int(cam_id), # camera_id_for_gallery (use camera-specific embeddings)
                int(cam_id), # camera_id_for_logging (DB camera.id)
                vs,
                buf,
                self.yolo,
                self.args,
                deep_tracker,
                iou_tracker,
                self.gallery_mgr,
                self.reid_extractor,
                self.face_app,
                self.global_owner,
                self.report,
                report_idx,
                self.ingest,
                self.stop_event,
            ),
            daemon=True,
        )
        t.start()

        with self._lock:
            if cam_id in self._db_states:
                self._db_states[cam_id]["thread"] = t

        return True

    def _get_db_camera(self, camera_id: int) -> Optional[dict[str, Any]]:
        if self._Session is None:
            return None

        Base = declarative_base()

        class CameraRow(Base):
            __tablename__ = "cameras"
            id = Column(Integer, primary_key=True)
            name = Column(String(64))
            ip_address = Column(String(64))
            is_active = Column(Boolean)

        with self._Session() as session:
            stmt = select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).where(CameraRow.id == int(camera_id))
            row = session.execute(stmt).first()
            if not row:
                return None
            return {
                "id": int(row.id),
                "name": str(row.name) if row.name is not None else "",
                "ip_address": str(row.ip_address) if row.ip_address is not None else "",
                "is_active": bool(row.is_active) if row.is_active is not None else True,
            }

    # ---------- common ----------
    def _should_use_deepsort(self) -> bool:
        args = self.args
        if bool(getattr(args, "no_deepsort", False)):
            return False
        if bool(getattr(args, "use_deepsort", False)):
            return DeepSort is not None
        return DeepSort is not None

    def _make_deepsort(self, gpu: bool):
        args = self.args
        if DeepSort is None:
            return None
        try:
            return DeepSort(
                max_age=int(args.max_age),
                n_init=int(args.n_init),
                nn_budget=int(args.nn_budget),
                max_cosine_distance=float(args.tracker_max_cosine),
                nms_max_overlap=float(args.tracker_nms_overlap),
                embedder="torchreid",
                embedder_gpu=bool(gpu),
                half=(bool(gpu) and bool(args.half)),
                bgr=True,
            )
        except Exception as e:
            print("[WARN] DeepSORT init failed, fallback to IoU tracker:", e)
            return None

    def _open_stream(self, src: str):
        args = self.args
        src = str(src)
        if args.reader == "latest":
            return LatestStream(src, rtsp_buffer=args.rtsp_buffer, decode_skip=args.decode_skip, rtsp_transport=args.rtsp_transport)
        if args.reader == "ffmpeg":
            return FFmpegStream(
                src,
                queue_size=args.queue_size,
                rtsp_transport=args.rtsp_transport,
                use_cuda=bool(args.ffmpeg_cuda),
                force_w=int(args.ffmpeg_width),
                force_h=int(args.ffmpeg_height),
            )
        return AdaptiveQueueStream(src, queue_size=args.queue_size, rtsp_transport=args.rtsp_transport, use_opencv=True)

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False

        print("[Runner] Stopping...")
        self.stop_event.set()

        # stop streams
        if self._mode == "manual":
            for st in list(self._manual_states.values()):
                try:
                    st["vs"].release()
                except Exception:
                    pass
            for st in list(self._manual_states.values()):
                t = st.get("thread")
                if t is not None:
                    try:
                        t.join(timeout=2.0)
                    except Exception:
                        pass
        else:
            for st in list(self._db_states.values()):
                try:
                    st["vs"].release()
                except Exception:
                    pass
            for st in list(self._db_states.values()):
                t = st.get("thread")
                if t is not None:
                    try:
                        t.join(timeout=2.0)
                    except Exception:
                        pass

        # stop ingestion
        if self.ingest is not None:
            try:
                self.ingest.stop()
                print("[DONE] Ingestion stopped (final flush attempted).")
            except Exception as e:
                print("[WARN] Failed to stop ingestion:", e)

        # report final write
        if self.report is not None:
            try:
                self.report.stop()
                self.report.write_csv(self.args.csv)
                print(f"[DONE] Summary CSV saved: {self.args.csv}")
            except Exception as e:
                print("[WARN] Failed to write summary CSV:", e)

        print("[Runner] Stopped.")

    # ---------- API helpers ----------
    def get_camera_buffer(self, cam_id: int) -> Optional[FrameBuffer]:
        """
        API cam_id:
          - manual mode: cam_id is index 0..N-1
          - db mode: cam_id is DB cameras.id
        """
        if self._mode == "manual":
            st = self._manual_states.get(int(cam_id))
            if not st:
                return None
            return st.get("buf")

        ok = self.ensure_camera_started(int(cam_id))
        if not ok:
            return None
        st = self._db_states.get(int(cam_id))
        if not st:
            return None
        return st.get("buf")

    def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        if self._mode == "manual":
            out = []
            for cid, st in sorted(self._manual_states.items(), key=lambda kv: kv[0]):
                vs = st.get("vs")
                out.append(
                    {
                        "id": int(cid),
                        "mode": "manual",
                        "opened": bool(vs.is_opened()) if vs else False,
                        "src": str(st.get("src", "")),
                    }
                )
            return out

        if self._Session is None:
            return []

        Base = declarative_base()

        class CameraRow(Base):
            __tablename__ = "cameras"
            id = Column(Integer, primary_key=True)
            name = Column(String(64))
            ip_address = Column(String(64))
            is_active = Column(Boolean)

        cams = []
        with self._Session() as session:
            stmt = select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).order_by(CameraRow.id.asc())
            rows = session.execute(stmt).all()
            for r in rows:
                is_active = bool(r.is_active) if r.is_active is not None else True
                if active_only and (not is_active):
                    continue
                cam_id = int(r.id)
                st = self._db_states.get(cam_id)
                opened = bool(st["vs"].is_opened()) if st and st.get("vs") else False
                cams.append(
                    {
                        "id": cam_id,
                        "name": str(r.name) if r.name is not None else "",
                        "ip_address": str(r.ip_address) if r.ip_address is not None else "",
                        "is_active": is_active,
                        "running": bool(st is not None),
                        "opened": opened,
                    }
                )
        return cams

    def status(self) -> Dict[str, Any]:
        st = {
            "started": bool(self._started),
            "mode": self._mode,
            "gallery_last_reload_ts": float(self.gallery_mgr.last_load_ts) if self.gallery_mgr else 0.0,
            "ingest_enabled": bool(self.ingest is not None),
            "running": [],
        }

        if self._mode == "manual":
            for cid, s in sorted(self._manual_states.items(), key=lambda kv: kv[0]):
                vs = s.get("vs")
                st["running"].append({"id": int(cid), "opened": bool(vs.is_opened()) if vs else False, "src": str(s.get("src", ""))})
        else:
            for cam_id, s in sorted(self._db_states.items(), key=lambda kv: kv[0]):
                vs = s.get("vs")
                st["running"].append({"id": int(cam_id), "opened": bool(vs.is_opened()) if vs else False, "src": str(s.get("src", ""))})

        return st

    def write_report_snapshot(self, path: Optional[str] = None) -> str:
        if self.report is None:
            raise RuntimeError("CSV reporting not enabled. Add --save-csv in PIPELINE_ARGS.")
        out_path = str(path or self.args.csv)
        self.report.write_csv_snapshot(out_path)
        return out_path


# ----------------------------
# Optional: CLI standalone runner (debug only)
# ----------------------------
def main():
    args = parse_args()
    runner = TrackingRunner(args)
    runner.start()

    # Debug view loop (only if --show and manual URLs provided)
    if args.show and runner._mode == "manual":
        try:
            while True:
                for cam_id in sorted(runner._manual_states.keys()):
                    buf = runner.get_camera_buffer(cam_id)
                    if not buf:
                        continue
                    frame = buf.get()
                    if frame is None:
                        continue
                    cv2.imshow(f"cam_{cam_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    runner.stop()


if __name__ == "__main__":
    main()
