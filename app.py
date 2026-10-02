"""
Flask backend for vehicle detection/tracking and number-plate OCR.

Endpoints
---------
GET  /health          -> device + model status
POST /vehicle/detect  -> image or video in, per-frame detections with tracking IDs
POST /plate/detect    -> image + vehicle bbox in, plate text (EasyOCR) out
"""
import base64
import json
import logging
import os
import tempfile
import threading

import cv2
import easyocr
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

# Config

VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "vehicle_best.pt")
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "plate_best.pt")
TRACKER_CFG = os.getenv("TRACKER_CFG", "bytetrack.yaml")  # or "botsort.yaml"
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_GPU else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("anpr")
log.info("Using device: %s", DEVICE)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# Model loading (once at startup)

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_reader = easyocr.Reader(OCR_LANGS, gpu=USE_GPU)

# Ultralytics trackers hold state, and models aren't safe to share across threads
vehicle_lock = threading.Lock()
plate_lock = threading.Lock()
ocr_lock = threading.Lock()



# Helpers

class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(e):
    return jsonify({"error": e.message}), e.status


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)"}), 413


def encode_jpg_b64(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ApiError("Failed to encode image", 500)
    return base64.b64encode(buf).decode("utf-8")


def decode_image_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError("Could not decode image")
    return img


def get_param(name, default=None):
    """Read a param from form-data, query string, or JSON body."""
    if name in request.form:
        return request.form[name]
    if name in request.args:
        return request.args[name]
    body = request.get_json(silent=True)
    if isinstance(body, dict) and name in body:
        return body[name]
    return default


def to_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_boxes(raw):
    """
    Accepts: "[x1,y1,x2,y2]", "[[x1,y1,x2,y2], ...]", "x1,y1,x2,y2",
    or the equivalent python lists / dicts {"x1":..,"y1":..,"x2":..,"y2":..}.
    Always returns a list of [x1, y1, x2, y2].
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw if raw.strip().startswith(("[", "{")) else f"[{raw}]")
        except json.JSONDecodeError:
            raise ApiError("bbox must be JSON like [x1,y1,x2,y2] or [[...],[...]]")
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list) and raw and all(isinstance(x, (int, float)) for x in raw):
        raw = [raw]

    boxes = []
    for b in raw:
        if isinstance(b, dict):
            try:
                b = [b["x1"], b["y1"], b["x2"], b["y2"]]
            except KeyError:
                raise ApiError("bbox dict needs keys x1, y1, x2, y2")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise ApiError("Each bbox must have exactly 4 numbers [x1,y1,x2,y2]")
        try:
            boxes.append([float(v) for v in b])
        except (TypeError, ValueError):
            raise ApiError("bbox values must be numeric")
    return boxes


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ApiError(f"bbox {box} is empty or outside the image ({w}x{h})")
    return x1, y1, x2, y2


def load_image_from_request():
    """Image from multipart file ('image' or 'file') or JSON {'image_base64': ...}."""
    f = request.files.get("image") or request.files.get("file")
    if f is not None:
        return decode_image_bytes(f.read())
    b64 = get_param("image_base64")
    if b64:
        if "," in b64[:100]:  # strip data-URL prefix
            b64 = b64.split(",", 1)[1]
        try:
            return decode_image_bytes(base64.b64decode(b64))
        except ApiError:
            raise
        except Exception:
            raise ApiError("Invalid base64 image")
    raise ApiError("Provide an image as multipart field 'image' or JSON 'image_base64'")



# Vehicle detection + tracking

def extract_detections(result):
    dets = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
    names = result.names
    for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
        dets.append({
            "track_id": int(tid) if tid is not None else None,
            "class_id": int(k),
            "class_name": names[int(k)],
            "confidence": round(float(c), 4),
            "bbox": {
                "x1": round(float(x1), 2), "y1": round(float(y1), 2),
                "x2": round(float(x2), 2), "y2": round(float(y2), 2),
            },
        })
    return dets


def frame_payload(result, frame_index, fps, include_images, jpg_quality):
    h, w = result.orig_shape
    payload = {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / fps, 3) if fps else None,
        "width": int(w),
        "height": int(h),
        "detections": extract_detections(result),
    }
    if include_images:
        # result.plot() draws boxes, labels and track IDs on the frame (BGR)
        payload["image_base64"] = encode_jpg_b64(result.plot(), jpg_quality)
        payload["image_mime"] = "image/jpeg"
    return payload


@app.route("/vehicle/detect", methods=["POST"])
def vehicle_detect():
    """
    multipart/form-data:
      file             image or video (required)
      conf             confidence threshold, default 0.25
      iou              NMS IoU threshold, default 0.5
      frame_stride     process every Nth video frame, default 1
      max_frames       stop after N processed frames (video), optional
      include_images   return annotated frame as base64 JPEG, default true
      only_tracked     drop detections without a track id (video), default false
      jpg_quality      1-100, default 85
    """
    f = request.files.get("file") or request.files.get("image") or request.files.get("video")
    if f is None:
        raise ApiError("Upload an image or video as multipart field 'file'")

    conf = float(get_param("conf", 0.25))
    iou = float(get_param("iou", 0.5))
    stride = max(1, int(get_param("frame_stride", 1)))
    max_frames = get_param("max_frames")
    max_frames = int(max_frames) if max_frames not in (None, "") else None
    include_images = to_bool(get_param("include_images"), True)
    only_tracked = to_bool(get_param("only_tracked"), False)
    jpg_quality = min(100, max(1, int(get_param("jpg_quality", 85))))

    ext = os.path.splitext(f.filename or "")[1].lower()
    is_video = ext in VIDEO_EXTS or (f.mimetype or "").startswith("video/")

    track_kwargs = dict(
        conf=conf, iou=iou, device=DEVICE, half=USE_GPU,
        tracker=TRACKER_CFG, persist=False, verbose=False,
    )

    # ---------------- image ----------------
    if not is_video:
        img = decode_image_bytes(f.read())
        with vehicle_lock:
            results = vehicle_model.track(img, **track_kwargs)
        frame = frame_payload(results[0], 0, None, include_images, jpg_quality)
        return jsonify({
            "type": "image",
            "device": DEVICE,
            "total_frames": 1,
            "frames": [frame],
        })

    # ---------------- video ----------------
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False)
    try:
        f.save(tmp)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ApiError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        frames, unique = [], {}
        with vehicle_lock:  # fresh tracker per call (persist=False), lock keeps it isolated
            gen = vehicle_model.track(
                source=tmp.name, stream=True, vid_stride=stride, **track_kwargs
            )
            for i, result in enumerate(gen):
                if max_frames is not None and i >= max_frames:
                    break
                payload = frame_payload(result, i * stride, fps, include_images, jpg_quality)
                if only_tracked:
                    payload["detections"] = [d for d in payload["detections"] if d["track_id"] is not None]
                for d in payload["detections"]:
                    if d["track_id"] is not None:
                        unique[d["track_id"]] = d["class_name"]
                frames.append(payload)

        class_counts = {}
        for cname in unique.values():
            class_counts[cname] = class_counts.get(cname, 0) + 1

        return jsonify({
            "type": "video",
            "device": DEVICE,
            "fps": fps,
            "source_total_frames": total_src_frames,
            "frame_stride": stride,
            "total_frames": len(frames),
            "unique_vehicles": len(unique),
            "unique_vehicles_by_class": class_counts,
            "frames": frames,
        })
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass



# Plate detection + OCR

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 64:  # upscale small plates, OCR does much better
        scale = 64 / h
        gray = cv2.resize(gray, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return gray


def read_plate_text(plate_crop):
    """Run EasyOCR and merge fragments into reading order (supports 2-line plates)."""
    img = preprocess_for_ocr(plate_crop)
    with ocr_lock:
        results = ocr_reader.readtext(img, allowlist=OCR_ALLOWLIST, detail=1, paragraph=False)
    if not results:
        return "", 0.0, []

    items = []
    for pts, text, c in results:
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        items.append({
            "text": text.upper().replace(" ", ""),
            "conf": float(c),
            "cx": sum(xs) / 4, "cy": sum(ys) / 4,
            "h": max(ys) - min(ys),
        })

    # group into lines by vertical position
    items.sort(key=lambda d: d["cy"])
    avg_h = sum(d["h"] for d in items) / len(items)
    lines, current = [], [items[0]]
    for it in items[1:]:
        if abs(it["cy"] - current[-1]["cy"]) < 0.6 * avg_h:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)

    ordered = [it for line in lines for it in sorted(line, key=lambda d: d["cx"])]
    text = "".join(d["text"] for d in ordered)
    avg_conf = sum(d["conf"] for d in ordered) / len(ordered)
    raw = [{"text": d["text"], "confidence": round(d["conf"], 4)} for d in ordered]
    return text, avg_conf, raw


@app.route("/plate/detect", methods=["POST"])
def plate_detect():
    """
    Input:
      image            multipart file, or JSON 'image_base64'  (required)
      bbox             vehicle bbox [x1,y1,x2,y2] or list of them (optional; whole image if omitted)
      conf             plate detection threshold, default 0.25
      pad              padding ratio around plate crop, default 0.05
      include_images   return plate crop as base64 JPEG, default false

    bbox is in the coordinates of the supplied image, i.e. pass the same frame
    you got from /vehicle/detect together with that vehicle's bbox.
    """
    img = load_image_from_request()
    H, W = img.shape[:2]
    conf = float(get_param("conf", 0.25))
    pad = float(get_param("pad", 0.05))
    include_images = to_bool(get_param("include_images"), False)

    boxes = parse_boxes(get_param("bbox"))
    if boxes is None:
        boxes = [[0, 0, W, H]]

    out = []
    for raw_box in boxes:
        x1, y1, x2, y2 = clamp_box(raw_box, W, H)
        vehicle_crop = img[y1:y2, x1:x2]

        with plate_lock:
            res = plate_model.predict(vehicle_crop, conf=conf, device=DEVICE,
                                      half=USE_GPU, verbose=False)[0]

        plates = []
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.conf.cpu().numpy()
            for (px1, py1, px2, py2), pconf in sorted(zip(pb, pc), key=lambda t: -t[1]):
                ch, cw = vehicle_crop.shape[:2]
                dx, dy = (px2 - px1) * pad, (py2 - py1) * pad
                cx1, cy1 = max(0, int(px1 - dx)), max(0, int(py1 - dy))
                cx2, cy2 = min(cw, int(px2 + dx)), min(ch, int(py2 + dy))
                plate_crop = vehicle_crop[cy1:cy2, cx1:cx2]
                if plate_crop.size == 0:
                    continue

                text, ocr_conf, parts = read_plate_text(plate_crop)
                entry = {
                    "plate_text": text,
                    "ocr_confidence": round(ocr_conf, 4),
                    "detection_confidence": round(float(pconf), 4),
                    # plate bbox mapped back to the full-image coordinates
                    "plate_bbox": {
                        "x1": round(float(px1) + x1, 2), "y1": round(float(py1) + y1, 2),
                        "x2": round(float(px2) + x1, 2), "y2": round(float(py2) + y1, 2),
                    },
                    "ocr_parts": parts,
                }
                if include_images:
                    entry["plate_image_base64"] = encode_jpg_b64(plate_crop)
                plates.append(entry)

        out.append({
            "vehicle_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "plates": plates,
        })

    return jsonify({"device": DEVICE, "results": out})



# Misc

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if USE_GPU else None,
        "vehicle_model": VEHICLE_MODEL_PATH,
        "plate_model": PLATE_MODEL_PATH,
        "vehicle_classes": vehicle_model.names,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
