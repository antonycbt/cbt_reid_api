# from __future__ import annotations

# import math
# import time
# from typing import Iterable, List, Optional, Tuple, Union

# import cv2
# import numpy as np


# BOUNDARY = b"frame"


# def _pack(jpg: bytes) -> bytes:
#     return (
#         b"--" + BOUNDARY + b"\r\n"
#         b"Content-Type: image/jpeg\r\n"
#         b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
#         + jpg + b"\r\n"
#     )


# # ----------------------------
# # Frame coercion (robust)
# # ----------------------------

# def _is_jpeg_bytes(b: bytes) -> bool:
#     # JPEG SOI marker FF D8
#     return len(b) > 4 and b[0] == 0xFF and b[1] == 0xD8


# def _coerce_to_bgr_uint8(
#     frm: object,
# ) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
#     """
#     Returns (bgr_ndarray_uint8, jpeg_bytes_if_already_encoded)
#     - If frm is already JPEG bytes -> returns (None, bytes)
#     - Else tries to convert to np.ndarray uint8 BGR-ish for cv2.imencode
#     """
#     if frm is None:
#         return None, None

#     # If your pipeline ever stores encoded JPEG bytes in the buffer
#     if isinstance(frm, (bytes, bytearray, memoryview)):
#         jb = bytes(frm)
#         if _is_jpeg_bytes(jb):
#             return None, jb
#         # not jpeg, ignore
#         return None, None

#     # Unwrap accidental nesting like (frame, ...) or [frame]
#     if isinstance(frm, tuple) and len(frm) >= 1:
#         frm = frm[0]
#     if isinstance(frm, list) and len(frm) >= 1:
#         frm = frm[0]

#     # Torch tensor -> numpy
#     try:
#         import torch  # optional
#         if isinstance(frm, torch.Tensor):
#             frm = frm.detach().to("cpu").numpy()
#     except Exception:
#         pass

#     # PIL -> numpy
#     try:
#         from PIL import Image  # optional
#         if isinstance(frm, Image.Image):
#             arr = np.array(frm)
#             # PIL is RGB typically
#             if arr.ndim == 3 and arr.shape[2] == 3:
#                 arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#             frm = arr
#     except Exception:
#         pass

#     if not isinstance(frm, np.ndarray):
#         return None, None

#     if frm.size == 0:
#         return None, None

#     # Ensure contiguous
#     if not frm.flags["C_CONTIGUOUS"]:
#         frm = np.ascontiguousarray(frm)

#     # Normalize dtype
#     if frm.dtype != np.uint8:
#         # clip to 0..255 then convert
#         frm = np.clip(frm, 0, 255).astype(np.uint8)

#     # Handle channels
#     if frm.ndim == 2:
#         # grayscale is ok for imencode; keep as is
#         return frm, None

#     if frm.ndim == 3:
#         if frm.shape[2] == 4:
#             # BGRA -> BGR
#             try:
#                 frm = cv2.cvtColor(frm, cv2.COLOR_BGRA2BGR)
#             except Exception:
#                 frm = frm[:, :, :3]
#         elif frm.shape[2] != 3:
#             # unknown channel count
#             frm = frm[:, :, :3] if frm.shape[2] > 3 else None
#             if frm is None:
#                 return None, None
#         return frm, None

#     # anything else unsupported
#     return None, None


# # ----------------------------
# # Grid helpers (self-contained)
# # ----------------------------

# def _resize_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
#     if img is None or img.size == 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)
#     ih, iw = img.shape[:2]
#     if ih <= 0 or iw <= 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)

#     s = max(w / float(iw), h / float(ih))
#     nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
#     r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

#     x1 = max(0, (nw - w) // 2)
#     y1 = max(0, (nh - h) // 2)
#     crop = r[y1:y1 + h, x1:x1 + w]
#     if crop.shape[0] != h or crop.shape[1] != w:
#         crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
#     return crop


# def _resize_contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
#     if img is None or img.size == 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)
#     ih, iw = img.shape[:2]
#     if ih <= 0 or iw <= 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)

#     s = min(w / float(iw), h / float(ih))
#     nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
#     r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

#     out = np.zeros((h, w, 3), dtype=np.uint8)
#     x1 = (w - nw) // 2
#     y1 = (h - nh) // 2
#     out[y1:y1 + nh, x1:x1 + nw] = r
#     return out


# def _choose_grid(n: int, out_w: int, out_h: int) -> Tuple[int, int]:
#     if n <= 1:
#         return 1, 1
#     best = None
#     for rows in range(1, n + 1):
#         cols = int(math.ceil(n / rows))
#         cw = out_w / float(cols)
#         ch = out_h / float(rows)
#         aspect = cw / max(1.0, ch)
#         penalty = abs(math.log(max(1e-6, aspect / (16.0 / 9.0))))
#         blanks = (rows * cols) - n
#         score = penalty + 0.02 * blanks
#         cand = (score, rows, cols)
#         if best is None or cand < best:
#             best = cand
#     assert best is not None
#     return int(best[1]), int(best[2])


# def make_grid_view(
#     frames: List[np.ndarray],
#     out_w: int,
#     out_h: int,
#     mode: str = "cover",
#     grid_rows: int = 0,
#     grid_cols: int = 0,
# ) -> np.ndarray:
#     frames = [f for f in frames if isinstance(f, np.ndarray) and f.size > 0]
#     if not frames:
#         return np.zeros((out_h, out_w, 3), dtype=np.uint8)

#     n = len(frames)
#     if grid_rows > 0 and grid_cols > 0:
#         rows, cols = int(grid_rows), int(grid_cols)
#         if rows * cols < n:
#             cols = int(math.ceil(n / rows))
#     elif grid_rows > 0:
#         rows = int(grid_rows)
#         cols = int(math.ceil(n / rows))
#     elif grid_cols > 0:
#         cols = int(grid_cols)
#         rows = int(math.ceil(n / cols))
#     else:
#         rows, cols = _choose_grid(n, out_w, out_h)

#     rows = max(1, rows)
#     cols = max(1, cols)

#     cell_w = max(1, int(out_w // cols))
#     cell_h = max(1, int(out_h // rows))

#     mode = (mode or "cover").lower()
#     resize_fn = _resize_cover if mode == "cover" else _resize_contain

#     tiles: List[np.ndarray] = []
#     for i in range(rows * cols):
#         if i < n:
#             img = frames[i]
#             if img.ndim == 2:
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             tiles.append(resize_fn(img, cell_w, cell_h))
#         else:
#             tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

#     row_imgs = []
#     idx = 0
#     for _ in range(rows):
#         row_imgs.append(np.concatenate(tiles[idx:idx + cols], axis=1))
#         idx += cols

#     vis = np.concatenate(row_imgs, axis=0)
#     if vis.shape[1] != out_w or vis.shape[0] != out_h:
#         vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
#     return vis


# # ----------------------------
# # Buffer compatibility (duck typing)
# # ----------------------------

# def _buf_wait_new(buf, last_seq: int, timeout: float) -> int:
#     if hasattr(buf, "wait_new") and callable(getattr(buf, "wait_new")):
#         try:
#             return int(buf.wait_new(last_seq, timeout=timeout))
#         except Exception:
#             pass
#     time.sleep(max(0.0, timeout))
#     frm, ts, _meta = _buf_get_frame(buf)
#     if ts is None:
#         return last_seq
#     return int(ts * 1000)


# def _buf_get_frame(buf) -> Tuple[object, Optional[float], dict]:
#     """
#     Supports:
#       - buf.get_latest() -> (frame, seq, ts, meta)
#       - buf.get() -> (frame, ts, meta)  (your RenderedFrame)
#       - buf.get_frame() -> frame
#     """
#     if hasattr(buf, "get_latest") and callable(getattr(buf, "get_latest")):
#         try:
#             frm, _seq, ts, meta = buf.get_latest()
#             return frm, float(ts), dict(meta or {})
#         except Exception:
#             pass

#     if hasattr(buf, "get") and callable(getattr(buf, "get")):
#         try:
#             out = buf.get()
#             if isinstance(out, tuple) and len(out) == 3:
#                 frm, ts, meta = out
#                 return frm, float(ts), dict(meta or {})
#         except Exception:
#             pass

#     if hasattr(buf, "get_frame") and callable(getattr(buf, "get_frame")):
#         try:
#             frm = buf.get_frame()
#             return frm, None, {}
#         except Exception:
#             pass

#     return None, None, {}


# def _buf_get_jpeg(buf, quality: int) -> Tuple[Optional[bytes], int]:
#     # If your buffer already supports cached JPEG, prefer it
#     if hasattr(buf, "get_jpeg") and callable(getattr(buf, "get_jpeg")):
#         try:
#             jpg, seq = buf.get_jpeg(quality=int(quality))
#             if jpg is not None:
#                 return bytes(jpg), int(seq)
#         except Exception:
#             pass

#     frm_obj, ts, _meta = _buf_get_frame(buf)
#     arr, jpeg_direct = _coerce_to_bgr_uint8(frm_obj)

#     if jpeg_direct is not None:
#         # already JPEG
#         seq = int(ts * 1000) if ts else -1
#         return jpeg_direct, seq

#     if arr is None:
#         return None, -1

#     q = int(max(30, min(95, int(quality))))
#     try:
#         ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#         if not ok:
#             return None, -1
#         seq = int(ts * 1000) if ts else -1
#         return enc.tobytes(), seq
#     except Exception:
#         return None, -1


# # ----------------------------
# # MJPEG generators (smooth / immediate)
# # ----------------------------

# def mjpeg_generator(buf, max_fps: int = 15, jpeg_quality: int = 80) -> Iterable[bytes]:
#     fps = int(max(1, min(30, int(max_fps))))
#     period = 1.0 / float(fps)

#     last_seq = -1
#     last_sent = 0.0

#     while True:
#         try:
#             seq_now = _buf_wait_new(buf, last_seq, timeout=period)

#             now = time.time()
#             if seq_now == last_seq and (now - last_sent) < period:
#                 continue

#             jpg, seq = _buf_get_jpeg(buf, jpeg_quality)
#             if jpg is None:
#                 time.sleep(0.01)
#                 continue

#             # enforce max fps, but allow immediate push on new seq
#             if (now - last_sent) < period and seq == last_seq:
#                 continue

#             last_seq = seq if seq != -1 else seq_now
#             last_sent = now
#             yield _pack(jpg)
#         except GeneratorExit:
#             return
#         except Exception:
#             # never crash the stream
#             time.sleep(0.02)
#             continue


# def mjpeg_generator_multi(
#     bufs: List,
#     max_fps: int = 10,
#     jpeg_quality: int = 80,
#     out_w: int = 1280,
#     out_h: int = 720,
#     grid_mode: str = "cover",
#     grid_rows: int = 0,
#     grid_cols: int = 0,
# ) -> Iterable[bytes]:
#     fps = int(max(1, min(30, int(max_fps))))
#     period = 1.0 / float(fps)

#     last_seqs = [-1] * len(bufs)
#     last_sent = 0.0

#     while True:
#         try:
#             t0 = time.time()
#             changed = False

#             # wait up to period for ANY buffer update
#             while (time.time() - t0) < period:
#                 for i, b in enumerate(bufs):
#                     s = _buf_wait_new(b, last_seqs[i], timeout=0.0)
#                     if s != last_seqs[i]:
#                         changed = True
#                 if changed:
#                     break
#                 time.sleep(0.004)

#             now = time.time()
#             if not changed and (now - last_sent) < period:
#                 continue

#             frames: List[np.ndarray] = []
#             for i, b in enumerate(bufs):
#                 frm_obj, ts, _meta = _buf_get_frame(b)
#                 arr, jpeg_direct = _coerce_to_bgr_uint8(frm_obj)

#                 if arr is None:
#                     # placeholder if missing / invalid
#                     arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#                 elif arr.ndim == 2:
#                     arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

#                 frames.append(arr)
#                 if ts:
#                     last_seqs[i] = int(ts * 1000)

#             grid = make_grid_view(
#                 frames,
#                 out_w=int(out_w),
#                 out_h=int(out_h),
#                 mode=str(grid_mode),
#                 grid_rows=int(grid_rows),
#                 grid_cols=int(grid_cols),
#             )

#             q = int(max(30, min(95, int(jpeg_quality))))
#             ok, enc = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#             if not ok:
#                 time.sleep(0.01)
#                 continue

#             last_sent = now
#             yield _pack(enc.tobytes())
#         except GeneratorExit:
#             return
#         except Exception:
#             time.sleep(0.02)
#             continue


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.tracking.pipeline import RenderedFrame


_BOUNDARY = b"--frame\r\n"
_HEADER = b"Content-Type: image/jpeg\r\n\r\n"
_TAIL = b"\r\n"


def _ensure_frame(x) -> Optional[np.ndarray]:
    """
    Defensive: buffer might return tuple in some buggy cases.
    We only accept a real numpy image.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], np.ndarray):
        return x[0]
    return None


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> bytes | None:
    q = int(max(30, min(95, int(quality))))
    try:
        ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        return enc.tobytes()
    except Exception:
        return None


def _placeholder(w: int = 1280, h: int = 720, text: str = "Waiting for frames...") -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def mjpeg_generator(buf: RenderedFrame, max_fps: int = 10, jpeg_quality: int = 80):
    """
    Smooth MJPEG: waits on new frames using wait_for_seq().
    """
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)

    last_seq = -1
    last_send = 0.0

    while True:
        # Wait up to 1s for a new frame
        frm, _ts, _meta, seq = buf.wait_for_seq(last_seq, timeout=1.0)
        frame = _ensure_frame(frm)

        if frame is None:
            frame = _placeholder(text="Starting camera / waiting for first frame...")

        now = time.time()
        dt = now - last_send
        if dt < min_dt:
            time.sleep(min_dt - dt)

        jpg = _encode_jpeg(frame, jpeg_quality)
        if jpg is None:
            # fallback placeholder
            jpg = _encode_jpeg(_placeholder(text="JPEG encode error"), jpeg_quality)
            if jpg is None:
                time.sleep(0.05)
                continue

        yield _BOUNDARY + _HEADER + jpg + _TAIL

        last_send = time.time()
        last_seq = int(seq)


def _grid_size(n: int, rows: int, cols: int) -> Tuple[int, int]:
    if rows > 0 and cols > 0:
        return rows, cols
    if rows > 0:
        cols = int(np.ceil(n / rows))
        return rows, cols
    if cols > 0:
        rows = int(np.ceil(n / cols))
        return rows, cols
    # auto
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _resize_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = max(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, (nw - w) // 2)
    y1 = max(0, (nh - h) // 2)
    out = r[y1:y1 + h, x1:x1 + w]
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _resize_contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    out[y1:y1 + nh, x1:x1 + nw] = r
    return out


def _make_grid(frames: List[np.ndarray], out_w: int, out_h: int, mode: str, rows: int, cols: int) -> np.ndarray:
    n = len(frames)
    rows, cols = _grid_size(n, rows, cols)
    cell_w = max(1, out_w // cols)
    cell_h = max(1, out_h // rows)

    resize_fn = _resize_cover if str(mode).lower() == "cover" else _resize_contain

    tiles: List[np.ndarray] = []
    for i in range(rows * cols):
        if i < n:
            tiles.append(resize_fn(frames[i], cell_w, cell_h))
        else:
            tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row_imgs = []
    idx = 0
    for _r in range(rows):
        row_imgs.append(np.concatenate(tiles[idx:idx + cols], axis=1))
        idx += cols

    grid = np.concatenate(row_imgs, axis=0)
    if grid.shape[1] != out_w or grid.shape[0] != out_h:
        grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return grid


def mjpeg_generator_multi(
    bufs: List[RenderedFrame],
    max_fps: int = 10,
    jpeg_quality: int = 80,
    out_w: int = 1280,
    out_h: int = 720,
    grid_mode: str = "cover",
    grid_rows: int = 0,
    grid_cols: int = 0,
):
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)

    last_seqs = [-1 for _ in bufs]
    last_send = 0.0

    while True:
        # Pull latest frames
        frames: List[np.ndarray] = []
        any_changed = False

        for i, b in enumerate(bufs):
            frm, _ts, _meta, seq = b.wait_for_seq(last_seqs[i], timeout=0.0)
            if int(seq) != int(last_seqs[i]):
                any_changed = True
            last_seqs[i] = int(seq)

            frame = _ensure_frame(frm)
            if frame is None:
                frame = _placeholder(text=f"Waiting cam {i} ...")
            frames.append(frame)

        if not any_changed:
            # Wait a bit on first buffer
            if bufs:
                frm, _ts, _meta, seq = bufs[0].wait_for_seq(last_seqs[0], timeout=min_dt)
                last_seqs[0] = int(seq)

        grid = _make_grid(frames, out_w=int(out_w), out_h=int(out_h), mode=grid_mode, rows=int(grid_rows), cols=int(grid_cols))

        now = time.time()
        dt = now - last_send
        if dt < min_dt:
            time.sleep(min_dt - dt)

        jpg = _encode_jpeg(grid, jpeg_quality)
        if jpg is None:
            jpg = _encode_jpeg(_placeholder(text="JPEG encode error"), jpeg_quality)
            if jpg is None:
                time.sleep(0.05)
                continue

        yield _BOUNDARY + _HEADER + jpg + _TAIL
        last_send = time.time()
