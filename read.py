# gauge_lock_and_segment_presence.py
# Alur: YOLO-DET (gauge) -> crop ROI -> YOLOv8-Seg (dial/needle)
# Output ringkas ke STDOUT: GAUGE:YES/NO | DIAL:YES/NO | info tambahan
# Dependensi: onnxruntime, opencv-python, numpy, (opsional) matplotlib

import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# ================== PATH (edit sesuai environment) ==================
# ONNX_DET   = r"C:\Users\fadel\Back Up Syarif\logger\CompVis Logger\models\gauge_detection_model.onnx"
# ONNX_SEG   = r"C:\Users\fadel\Back Up Syarif\logger\CompVis Logger\models\segmentation_model.onnx"
# IMAGE_PATH = r"C:\Users\fadel\Back Up Syarif\logger\CompVis Logger\gauge\gauge_1.png"

# Jika di STM32MP257 (Linux), contoh:
# C:\Users\fadel\Downloads\mini_gauge\models\gauge_detection_model.onnx
ONNX_DET   = "models/gauge_detection_model.onnx"
ONNX_SEG   = "models/segmentation_model.onnx"
IMAGE_PATH = "images/gauge_1.png"

# ================== Opsional: nama kelas & indeks ==================
CLASS_NAMES_DET = []     # contoh: ["gauge"]
DIAL_CLASS_ID   = None   # set ke integer jika kamu tahu ID kelas "dial" di model seg
NEEDLE_CLASS_ID = None   # set ke integer jika kamu tahu ID kelas "needle" di model seg

# ================== PARAMETER ==================
# Deteksi gauge
DET_CONF_THRES = 0.40
DET_IOU_NMS    = 0.50
DET_MAX_DET    = 50
IMG_SIZE_FALLBACK = 640

# Segmentasi dial/needle
SEG_CONF_THRES = 0.55
SEG_MASK_THRES = 0.55
SEG_IOU_NMS    = 0.60
SEG_MAX_DET    = 20

# Debug
SAVE_OVERLAY = False
OVERLAY_PATH = "gauge_dial_overlay.jpg"
SHOW_DEBUG   = False     # True = tampilkan plt

# ================== Utils ==================
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_coords_back(xyxy, r, dw, dh, ow, oh):
    x1, y1, x2, y2 = xyxy
    x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
    x1 = max(0, min(ow-1, x1)); y1 = max(0, min(oh-1, y1))
    x2 = max(0, min(ow-1, x2)); y2 = max(0, min(oh-1, y2))
    return [int(x1), int(y1), int(x2), int(y2)]

def nms_xyxy(boxes, scores, iou_thres=0.5, max_keep=300):
    if boxes.size == 0: return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (np.maximum(0, x2-x1) * np.maximum(0, y2-y1)).astype(np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if len(keep) >= max_keep or order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds+1]
    return np.array(keep, dtype=int)

def ensure_NxD(a):
    if a.ndim == 3 and a.shape[0] == 1: a = a[0]
    if a.ndim != 2:
        raise RuntimeError(f"Pred shape tak terduga: {a.shape}")
    N, D = a.shape
    if D > 4096 or (N < D and (N <= 20 and D >= 80)):
        a = a.T
    return a

def ensure_NxD_seg(p, mask_dim):
    if p.ndim != 2: raise RuntimeError(f"pred shape tak terduga: {p.shape}")
    N, D = p.shape
    if D - 4 - mask_dim <= 0 or D > 4096:
        p = p.T
    return p

def choose_dtype(input_meta, arr_f32):
    t = input_meta.type or ""
    if "float16" in t:
        return arr_f32.astype(np.float16)
    return arr_f32.astype(np.float32)

# ================== ONNX Helpers ==================
def run_yolo_det_onnx(sess_det: ort.InferenceSession, img_bgr: np.ndarray,
                      conf_thres=0.4, iou_nms=0.5, max_det=50):
    """Return best_box (xyxy int) atau None, plus semua boxes & scores (untuk overlay)."""
    in_meta = sess_det.get_inputs()[0]
    _, c, ih, iw = in_meta.shape
    ih = int(ih) if isinstance(ih, (int, np.integer)) and ih > 0 else IMG_SIZE_FALLBACK
    iw = int(iw) if isinstance(iw, (int, np.integer)) and iw > 0 else IMG_SIZE_FALLBACK

    h0, w0 = img_bgr.shape[:2]
    img_lb, r, (dw, dh) = letterbox(img_bgr, (ih, iw))
    inp = img_lb[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0
    inp = choose_dtype(in_meta, inp)

    outs = sess_det.run(None, {in_meta.name: inp})

    # tensor 2D deteksi
    cand = None
    for o in outs:
        try:
            a = ensure_NxD(np.array(o))
            cand = a if cand is None or a.shape[0] > cand.shape[0] else cand
        except Exception:
            pass
    if cand is None:
        raise RuntimeError("ONNX deteksi tidak memiliki tensor [N,D].")

    N, D = cand.shape
    if D == 6:
        xyxy_l = cand[:, :4].astype(np.float32)
        scores = cand[:, 4].astype(np.float32)
        cls_id = cand[:, 5].astype(np.int32)
    else:
        nc = D - 4
        logits = cand[:, 4:4+nc].astype(np.float32)
        probs  = _sigmoid(logits) if (logits.max() > 1.0 or logits.min() < 0.0) else logits
        scores = probs.max(axis=1).astype(np.float32)
        cls_id = probs.argmax(axis=1).astype(np.int32)
        xywh = cand[:, :4].astype(np.float32)
        if xywh.max() <= 1.5:
            xywh[:, [0,2]] *= float(iw); xywh[:, [1,3]] *= float(ih)
        xyxy_l = np.empty_like(xywh)
        xyxy_l[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy_l[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy_l[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy_l[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = scores > conf_thres
    xyxy_l, scores, cls_id = xyxy_l[keep], scores[keep], cls_id[keep]
    if xyxy_l.shape[0] > 1:
        keep_idx = nms_xyxy(xyxy_l.copy(), scores.copy(), iou_thres=iou_nms, max_keep=max_det)
        xyxy_l, scores, cls_id = xyxy_l[keep_idx], scores[keep_idx], cls_id[keep_idx]

    # map balik ke ukuran asli
    boxes = []
    for (x1, y1, x2, y2) in xyxy_l:
        boxes.append(scale_coords_back([x1, y1, x2, y2], r, dw, dh, w0, h0))
    boxes = np.array(boxes, dtype=int)

    # pilih satu terbaik
    best_box = tuple(boxes[np.argmax(scores)]) if boxes.size else None
    return best_box, boxes, scores, cls_id

def run_yolov8_seg_onnx_on_roi(sess_seg: ort.InferenceSession, roi_bgr: np.ndarray,
                               conf_thres=0.55, mask_thres=0.55, iou_nms=0.60, max_det=20):
    """Kembalikan list deteksi pada ROI: {mask(bool ROI), box(xyxy ROI), score, cls}"""
    in_meta = sess_seg.get_inputs()[0]
    _, c, ih, iw = in_meta.shape
    ih = int(ih) if isinstance(ih, (int, np.integer)) and ih > 0 else 640
    iw = int(iw) if isinstance(iw, (int, np.integer)) and iw > 0 else 640

    rh, rw = roi_bgr.shape[:2]
    img_lb, r, (dw, dh) = letterbox(roi_bgr, (ih, iw))
    inp = img_lb[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0
    inp = choose_dtype(in_meta, inp)

    outs = sess_seg.run(None, {in_meta.name: inp})

    pred, proto = None, None
    for o in outs:
        if o.ndim == 3: pred = o
        elif o.ndim == 4: proto = o
    if pred is None or proto is None:
        return []

    mask_dim = int(proto.shape[1])
    p = ensure_NxD_seg(pred[0], mask_dim)
    D = int(p.shape[1])
    nc = D - 4 - mask_dim
    if nc <= 0: return []

    xywh   = p[:, :4].astype(np.float32)
    logits = p[:, 4:4+nc].astype(np.float32)
    coeff  = p[:, 4+nc:].astype(np.float32)

    if xywh.max() <= 1.5:
        xywh[:, [0,2]] *= float(iw); xywh[:, [1,3]] *= float(ih)

    probs = _sigmoid(logits) if (logits.max() > 1.0 or logits.min() < 0.0) else logits
    conf  = probs.max(axis=1); clsid = probs.argmax(axis=1)

    keep = conf > conf_thres
    xywh, coeff, conf, clsid = xywh[keep], coeff[keep], conf[keep], clsid[keep]
    if xywh.shape[0] == 0: return []

    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    if xyxy.shape[0] > 1:
        keep_idx = nms_xyxy(xyxy.copy(), conf.copy(), iou_thres=iou_nms, max_keep=max_det)
        xyxy, coeff, conf, clsid = xyxy[keep_idx], coeff[keep_idx], conf[keep_idx], clsid[keep_idx]

    proto_ = proto[0].astype(np.float32)  # [C, mh, mw]
    C, mh, mw = proto_.shape
    proto_flat = proto_.reshape(C, mh*mw)

    results = []
    for i in range(xyxy.shape[0]):
        # mask kecil -> proto
        msmall = _sigmoid(coeff[i] @ proto_flat).reshape(mh, mw)
        # ke kanvas letterbox
        mask_l = cv2.resize(msmall, (iw, ih), interpolation=cv2.INTER_LINEAR)

        # hapus padding
        x0, y0 = int(round(dw)), int(round(dh))
        x1p, y1p = int(round(iw - dw)), int(round(ih - dh))
        if x1p <= x0 or y1p <= y0:
            continue
        mask_crop = mask_l[y0:y1p, x0:x1p]

        # ke ukuran ROI
        mask_roi = cv2.resize(mask_crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_roi > mask_thres).astype(np.uint8)

        # bbox ke ROI
        box_roi = scale_coords_back(list(xyxy[i]), r, dw, dh, rw, rh)

        results.append({"mask": mask_bin, "box": box_roi, "score": float(conf[i]), "cls": int(clsid[i])})
    return results

# ================== Heuristik "ada dial/needle?" ==================
def has_dial_or_needle(dets, roi_shape):
    """
    True jika:
      - ada deteksi ber-kelas DIAL_CLASS_ID/NEEDLE_CLASS_ID, atau
      - ada mask tipis memanjang (needle-like) di ROI (heuristik),
      - atau ada mask besar membulat (dial face) (fallback).
    """
    rh, rw = roi_shape[:2]
    roi_area = float(rh * rw + 1e-6)

    # 1) Jika indeks kelas diketahui
    for d in dets:
        if (DIAL_CLASS_ID is not None and d["cls"] == DIAL_CLASS_ID) or \
           (NEEDLE_CLASS_ID is not None and d["cls"] == NEEDLE_CLASS_ID):
            return True

    # 2) Needle-like: memanjang dan ujung jauh dari pusat ROI
    cx, cy = rw // 2, rh // 2
    for d in dets:
        m = d["mask"]
        if m.sum() < 60: 
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            continue
        c = max(cnts, key=cv2.contourArea)
        (w, h) = cv2.minAreaRect(c)[1]
        if w == 0 or h == 0: 
            continue
        ratio = max(w, h) / (min(w, h) + 1e-6)
        if ratio >= 2.2:  # cukup memanjang
            pts = c.reshape(-1, 2)
            far = np.hypot(pts[:,0]-cx, pts[:,1]-cy).max()
            if far > 0.35 * min(rw, rh):  # ujung cukup jauh
                return True

    # 3) Dial-face-like: bulat & area moderat
    for d in dets:
        m = d["mask"]; area = float(m.sum())
        if area/roi_area < 0.04:  # terlalu kecil
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True) + 1e-6
        roundness = 4.0*np.pi*cv2.contourArea(c)/(peri*peri)  # 1 = circle
        if roundness > 0.35:
            return True

    return False

# ================== Pipeline ==================
def process_and_check(det_path, seg_path, image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # ORT sessions (CPU-only default, aman untuk STM32MP257)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess_det = ort.InferenceSession(det_path, sess_options=so, providers=["CPUExecutionProvider"])
    sess_seg = ort.InferenceSession(seg_path, sess_options=so, providers=["CPUExecutionProvider"])

    # 1) Deteksi gauge
    best_box, boxes, scores, clsid = run_yolo_det_onnx(
        sess_det, img, conf_thres=DET_CONF_THRES, iou_nms=DET_IOU_NMS, max_det=DET_MAX_DET
    )
    has_gauge = best_box is not None

    if not has_gauge:
        # Tidak usah segmen kalau gauge tidak terdeteksi
        if SAVE_OVERLAY:
            cv2.imwrite(OVERLAY_PATH, img)
        return has_gauge, False, img  # (gauge, dial, overlay)

    # 2) ROI dari gauge terbaik (+sedikit padding)
    x1, y1, x2, y2 = best_box
    pad = 0.05
    w, h = x2-x1, y2-y1
    x1p = max(0, int(x1 - w*pad)); y1p = max(0, int(y1 - h*pad))
    x2p = min(img.shape[1]-1, int(x2 + w*pad)); y2p = min(img.shape[0]-1, int(y2 + h*pad))
    roi = img[y1p:y2p, x1p:x2p].copy()
    if roi.size == 0:
        roi = img.copy(); x1p, y1p = 0, 0

    # 3) Segmentasi pada ROI
    dets = run_yolov8_seg_onnx_on_roi(
        sess_seg, roi,
        conf_thres=SEG_CONF_THRES, mask_thres=SEG_MASK_THRES,
        iou_nms=SEG_IOU_NMS, max_det=SEG_MAX_DET
    )

    # 4) Apakah ada dial/needle?
    has_dial = has_dial_or_needle(dets, roi.shape)

    # (opsional) overlay sederhana
    overlay = img.copy()
    cv2.rectangle(overlay, (x1p, y1p), (x2p, y2p), (0, 140, 255), 2)
    for d in dets:
        m = d["mask"].astype(bool)
        overlay[y1p:y2p, x1p:x2p][m] = (0.35*np.array((0,255,0)) + 0.65*overlay[y1p:y2p, x1p:x2p][m]).astype(np.uint8)

    info = f"GAUGE={'YES' if has_gauge else 'NO'} | DIAL={'YES' if has_dial else 'NO'}"
    cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 1, cv2.LINE_AA)

    if SAVE_OVERLAY:
        cv2.imwrite(OVERLAY_PATH, overlay)

    return has_gauge, has_dial, overlay

# ================== Main ==================
if __name__ == "__main__":
    if not os.path.exists(ONNX_DET):
        raise FileNotFoundError(f"Tidak ditemukan: {ONNX_DET}")
    if not os.path.exists(ONNX_SEG):
        raise FileNotFoundError(f"Tidak ditemukan: {ONNX_SEG}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {IMAGE_PATH}")

    gauge, dial, overlay = process_and_check(ONNX_DET, ONNX_SEG, IMAGE_PATH)

    # Output ringkas (untuk sistem tertanam)
    print(f"GAUGE:{'YES' if gauge else 'NO'} | DIAL:{'YES' if dial else 'NO'}")

    # Debug visual opsional
    if SHOW_DEBUG:
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Gauge ROI + Segmentation Overlay")

        plt.axis('off'); plt.tight_layout(); plt.show()
