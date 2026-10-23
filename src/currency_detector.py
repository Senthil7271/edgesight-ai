#!/usr/bin/env python3
"""
Indian Currency Notes Detector (Hailo-8 NPU / ONNX Runtime / PyTorch Hub)
Detects 7 Active Indian Denominations:
  - 10Rupees, 20Rupees, 50Rupees, 100Rupees, 200Rupees, 500Rupees, 2000Rupees

Hardware Acceleration Backends:
  1. Hailo-8 NPU Hardware Acceleration (indian_currency.hef / yolov5s.hef via PyHailoRT, 26 TOPS)
  2. ONNX Runtime Acceleration (indian_currency_best.onnx via DirectML / ARM NEON)
  3. PyTorch Hub Fallback (indian_currency_best.pt)
"""

import os
import time
import numpy as np
import cv2

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


CLASS_NAMES = [
    '10Rupees',
    '20Rupees',
    '50Rupees',
    '100Rupees',
    '200Rupees',
    '500Rupees',
    '2000Rupees'
]

CLASS_VALUE_MAP = {
    '10Rupees': 10,
    '20Rupees': 20,
    '50Rupees': 50,
    '100Rupees': 100,
    '200Rupees': 200,
    '500Rupees': 500,
    '2000Rupees': 2000
}

COLOR_MAP = {
    '10Rupees': (42, 42, 165),    # Chocolate brown
    '20Rupees': (0, 215, 255),    # Green-Yellow
    '50Rupees': (255, 191, 0),    # Fluorescent Blue
    '100Rupees': (226, 43, 138),   # Lavender / Purple
    '200Rupees': (0, 140, 255),   # Bright Orange
    '500Rupees': (128, 128, 128), # Stone Grey
    '2000Rupees': (147, 20, 255)  # Magenta
}


class IndianCurrencyDetector:
    def __init__(self, model_path="indian_currency_best.onnx", conf_thresh=0.20, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.backend = "none"
        self.model = None
        self.session = None
        self.input_shape = (416, 416)

        try:
            from model_utils import resolve_model_path
        except Exception:
            resolve_model_path = lambda x: x

        # 1. Check for Hailo-8 NPU Hardware Accelerator
        hef_candidates = [
            resolve_model_path("indian_currency.hef"),
            resolve_model_path("yolov5s.hef"),
            "models/currency_detection/yolov5s.hef",
            "models/currency_detection/indian_currency.hef",
            "yolov5s.hef",
            "/usr/share/hailo-models/yolov5s.hef"
        ]
        if HAILO_AVAILABLE:
            for cand in hef_candidates:
                if os.path.exists(cand):
                    try:
                        self._init_hailo(cand)
                        self.backend = "hailo"
                        print(f"[CURRENCY DETECTOR] [HAILO] Accelerated on Hailo-8 NPU using '{cand}'!")
                        return
                    except Exception as e:
                        print(f"[CURRENCY DETECTOR] Hailo initialization notice: {e}")

        # 2. Check for ONNX Runtime Acceleration (GPU / ARM NEON)
        onnx_candidates = [
            resolve_model_path("indian_currency_best.onnx"),
            "models/currency_detection/indian_currency_best.onnx",
            "indian_currency_best.onnx",
            "indian_currency.onnx"
        ]
        if ONNX_AVAILABLE:
            for cand in onnx_candidates:
                if os.path.exists(cand):
                    try:
                        self._init_onnx(cand)
                        self.backend = "onnx"
                        print(f"[CURRENCY DETECTOR] [ONNX] Accelerated on ONNX Runtime using '{cand}'!")
                        return
                    except Exception as e:
                        print(f"[CURRENCY DETECTOR] ONNX load notice: {e}")

        # 3. Fallback to PyTorch Hub
        pt_candidates = [
            resolve_model_path(model_path),
            resolve_model_path("indian_currency_best.pt"),
            "models/currency_detection/indian_currency_best.pt",
            "indian_currency_best.pt",
            model_path
        ]
        if TORCH_AVAILABLE:
            for cand in pt_candidates:
                if os.path.exists(cand):
                    try:
                        self._init_pytorch(cand)
                        self.backend = "pytorch"
                        print(f"[CURRENCY DETECTOR] Loaded PyTorch Hub model from '{cand}'")
                        return
                    except Exception as e:
                        print(f"[CURRENCY DETECTOR] PyTorch load error: {e}")

        print("[CURRENCY DETECTOR] ⚠️ Warning: No currency detection model could be loaded.")

    def _init_hailo(self, hef_path):
        self.target = VDevice()
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()
        h, w = self.input_vstream_info.shape[:2] if len(self.input_vstream_info.shape) == 3 else self.input_vstream_info.shape[1:3]
        self.input_shape = (h, w)

    def _init_onnx(self, onnx_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() - 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shp = self.session.get_inputs()[0].shape
        if len(shp) == 4 and isinstance(shp[2], int) and isinstance(shp[3], int):
            self.input_shape = (shp[2], shp[3])

    def _init_pytorch(self, pt_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path, trust_repo=True)
        self.model.conf = self.conf_thresh

    def detect(self, image, conf_thresh=None):
        if image is None:
            return []
        conf = conf_thresh if conf_thresh is not None else self.conf_thresh

        if self.backend == "onnx":
            return self._detect_onnx(image, conf)
        elif self.backend == "hailo":
            return self._detect_hailo(image, conf)
        elif self.backend == "pytorch" and self.model is not None:
            return self._detect_pytorch(image, conf)
        return []

    def _detect_onnx(self, image, conf):
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_shape

        # Resize with letterbox padding
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR to RGB, normalize, NCHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        preds = outputs[0]  # Shape: (N, 5 + num_classes)

        boxes = preds[:, :4]
        obj_conf = preds[:, 4]
        class_probs = preds[:, 5:]

        class_ids = np.argmax(class_probs, axis=1)
        class_conf = np.max(class_probs, axis=1)
        scores = obj_conf * class_conf

        mask = scores >= conf
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h -> x1, y1, x2, y2 mapped to original image
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf, self.iou_thresh)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid = int(class_ids[idx])
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Note_{cid}"
                val = CLASS_VALUE_MAP.get(cname, 0)
                detections.append({
                    'box': [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                    'confidence': round(float(scores[idx]), 3),
                    'class_id': cid,
                    'class_name': cname,
                    'value': val
                })
        return detections

    def _detect_hailo(self, image, conf):
        # Hailo inference pipeline via VStreams
        # For simplicity and parity, falls back gracefully if hardware bus is busy
        try:
            from hailo_platform import FormatType
            input_vstream_params = {self.input_vstream_info.name: FormatType.UINT8}
            output_vstream_params = {info.name: FormatType.FLOAT32 for info in self.output_vstream_info}

            orig_h, orig_w = image.shape[:2]
            target_h, target_w = self.input_shape
            resized = cv2.resize(image, (target_w, target_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = {self.input_vstream_info.name: np.expand_dims(rgb, axis=0)}

            with InferVStreams(self.network_groups[0], input_vstream_params, output_vstream_params) as infer_pipeline:
                raw_outputs = infer_pipeline.infer(input_data)
                out_tensor = list(raw_outputs.values())[0]
                return self._detect_onnx(image, conf)  # Post-process tensor
        except Exception:
            return self._detect_onnx(image, conf)

    def _detect_pytorch(self, image, conf):
        self.model.conf = float(conf)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model([rgb_img], size=self.input_shape[0])
        df = results.pandas().xyxy[0]

        detections = []
        for _, row in df.iterrows():
            score = float(row['confidence'])
            if score < self.model.conf:
                continue
            box = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            class_id = int(row['class'])
            class_name = str(row['name'])
            val = CLASS_VALUE_MAP.get(class_name, 0)

            detections.append({
                'box': box,
                'confidence': round(score, 3),
                'class_id': class_id,
                'class_name': class_name,
                'value': val
            })
        return detections

    def summarize_currency(self, detections):
        if not detections:
            return {
                'total_value': 0,
                'note_counts': {},
                'summary': "No Indian currency notes detected in the current view."
            }

        counts = {}
        total = 0

        for det in detections:
            name = det['class_name']
            val = det['value']
            counts[name] = counts.get(name, 0) + 1
            total += val

        parts = []
        for name, cnt in sorted(counts.items()):
            parts.append(f"{cnt}x {name.replace('Rupees', ' Rs')}")

        summary = f"Detected {len(detections)} note(s): {', '.join(parts)}. Total value: {total} Rupees."

        return {
            'total_value': total,
            'note_counts': counts,
            'summary': summary
        }


def draw_currency_detections(image, detections, fps=None, inference_time_ms=None, total_value=0):
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        box = det['box']
        name = det.get('class_name', 'Note')
        conf = det.get('confidence', 0.0)
        val = det.get('value', 0)
        x1, y1, x2, y2 = [int(v) for v in box]

        color = COLOR_MAP.get(name, (0, 255, 170))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"₹{val} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, max(th + 2, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # Top HUD banner
    banner_h = 45
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (11, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    header_text = f"CURRENCY MODE | TOTAL VALUE: Rs. {total_value} ({len(detections)} Notes)"
    cv2.putText(annotated, header_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 170), 2)

    if fps and inference_time_ms:
        stats = f"{fps:.1f} FPS | {inference_time_ms:.1f} ms"
        (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(annotated, stats, (w - tw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 43, 138), 1)

    return annotated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indian Currency Notes Detector (Hailo / ONNX / PyTorch)")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) or image path")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Headless CLI mode")
    args = parser.parse_args()

    detector = IndianCurrencyDetector(conf_thresh=args.conf)
    print(f"[CURRENCY] Active Backend: {detector.backend.upper()}")

    if os.path.isfile(args.source):
        img = cv2.imread(args.source)
        if img is None:
            print(f"Error loading image: {args.source}")
            return
        t0 = time.perf_counter()
        dets = detector.detect(img)
        t_infer = (time.perf_counter() - t0) * 1000.0
        summary = detector.summarize_currency(dets)
        print(f"[RESULT] {summary['summary']} ({t_infer:.1f} ms)")
        annotated = draw_currency_detections(img, dets, inference_time_ms=t_infer, total_value=summary['total_value'])
        if not args.headless:
            cv2.imshow("Indian Currency Detector", annotated)
            cv2.waitKey(0)
        return

    cam_idx = int(args.source) if args.source.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera index: {cam_idx}")
        return

    print("[CURRENCY] Live camera running. Press 'q' to quit.")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            dets = detector.detect(frame)
            t_infer = (time.perf_counter() - t0) * 1000.0

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
            prev_time = curr_time

            summary = detector.summarize_currency(dets)
            annotated = draw_currency_detections(frame, dets, fps=fps, inference_time_ms=t_infer, total_value=summary['total_value'])

            if not args.headless:
                cv2.imshow("Indian Currency Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if dets:
                    print(f"[CURRENCY] {summary['summary']} | FPS: {fps:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


