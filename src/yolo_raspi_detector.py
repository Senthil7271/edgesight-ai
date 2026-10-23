#!/usr/bin/env python3
"""
YOLO Real-Time Object Detection Engine for Raspberry Pi & Edge Devices
Integrated with Scene Description & Spatial Relationship Engine:
  - "Person holding mobile"
  - "Person sleeping in couch"
  - "Person using laptop"
  - "Person sitting on chair"
"""

import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import argparse
import numpy as np
import cv2


from scene_descriptor import SceneDescriptor

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

np.random.seed(42)
CLASS_COLORS = np.random.randint(50, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
scene_engine = SceneDescriptor()


class YOLOOnnxDetector:
    def __init__(self, model_path, input_size=(320, 320), conf_thresh=0.25, iou_thresh=0.45):
        import onnxruntime as ort
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() - 1)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[INFO] Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_size = (input_shape[3], input_shape[2])

    def preprocess(self, img):
        h, w = img.shape[:2]
        target_w, target_h = self.input_size
        
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale, dx, dy

    def detect(self, img):
        blob, scale, dx, dy = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        
        preds = np.squeeze(outputs)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
            
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in preds:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence >= self.conf_thresh:
                cx, cy, bw, bh = pred[0:4]
                
                x1 = int((cx - bw / 2 - dx) / scale)
                y1 = int((cy - bh / 2 - dy) / scale)
                width = int(bw / scale)
                height = int(bh / scale)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'class_id': class_ids[i],
                    'class_name': COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f'cls_{class_ids[i]}',
                    'confidence': confidences[i],
                    'box': [max(0, x), max(0, y), w, h]
                })
        return results


class YOLOUltralyticsDetector:
    def __init__(self, model_name='yolov8n.pt', conf_thresh=0.25):
        from ultralytics import YOLO
        print(f"[INFO] Loading PyTorch model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def detect(self, img):
        results_raw = self.model(img, conf=self.conf_thresh, verbose=False)[0]
        detections = []
        for box in results_raw.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'class_id': cls_id,
                'class_name': COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls_{cls_id}',
                'confidence': conf,
                'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            })
        return detections


def draw_detections(img, detections, fps=0.0, inference_time_ms=0.0):
    """Draw stylish bounding boxes and Scene Description Telemetry banner"""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # 1. Bounding Boxes
    for det in detections:
        x, y, bw, bh = det['box']
        cls_id = det['class_id']
        cls_name = det['class_name']
        conf = det['confidence']
        
        color = [int(c) for c in CLASS_COLORS[cls_id % len(CLASS_COLORS)]]
        
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        
        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x, max(0, y - label_h - 6)), (x + label_w + 6, max(label_h + 6, y)), color, -1)
        cv2.putText(img, label, (x + 3, max(label_h + 2, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Scene Description Engine Integration
    scene_descriptions = scene_engine.generate_scene_description(detections)

    # Telemetry Panel Height dynamically sized
    panel_height = 80 + max(0, len(scene_descriptions) * 22)
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 480), 10 + panel_height), (20, 24, 33), -1)
    img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    cv2.rectangle(img, (10, 10), (min(w - 10, 480), 10 + panel_height), (0, 210, 255), 1)

    cv2.putText(img, f"YOLO Scene Description Engine", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f} | Latency: {inference_time_ms:.1f} ms", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Render Scene Descriptions
    y_offset = 74
    for desc in scene_descriptions:
        cv2.putText(img, f"• {desc}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 170), 1, cv2.LINE_AA)
        y_offset += 22

    return img


def open_camera(source):
    if str(source).isdigit():
        source = int(source)
    
    if str(source).lower() == 'picamera':
        try:
            from picamera2 import Picamera2
            print("[INFO] Initializing Picamera2...")
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": (640, 480)})
            picam.configure(config)
            picam.start()
            return ('picamera', picam)
        except Exception as e:
            print(f"[WARN] Picamera2 unavailable ({e}), falling back to OpenCV VideoCapture(0)")
            source = 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return ('opencv', cap)


def read_frame(cam_tuple):
    cam_type, handle = cam_tuple
    if cam_type == 'picamera':
        frame = handle.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = handle.read()
        if not ret:
            return None
        return frame


def ensure_model_exists(model_path):
    try:
        from model_utils import resolve_model_path
        resolved = resolve_model_path(model_path)
        if os.path.exists(resolved):
            return resolved
    except Exception:
        pass

    candidates = [
        model_path,
        os.path.join("models", "object_detection", os.path.basename(model_path)),
        os.path.join("models", model_path),
        os.path.join("models", os.path.basename(model_path))
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    try:
        from ultralytics import YOLO
        pt_model = YOLO("yolov8n.pt")
        onnx_file = pt_model.export(format="onnx", imgsz=320, dynamic=False, simplify=True)
        return onnx_file
    except Exception:
        return "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser(description="YOLO Object & Scene Description Engine")
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Path to model")
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or picamera")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference size")
    parser.add_argument("--headless", action="store_true", help="Headless CLI mode")
    args = parser.parse_args()

    model_file = ensure_model_exists(args.model)
    if model_file.endswith(".onnx"):
        detector = YOLOOnnxDetector(model_file, input_size=(args.imgsz, args.imgsz), conf_thresh=args.conf)
    else:
        detector = YOLOUltralyticsDetector(model_file, conf_thresh=args.conf)

    cam_handle = open_camera(args.source)
    print("[INFO] Starting detection with Scene Descriptor... Press 'q' to quit.")
    
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = read_frame(cam_handle)
            if frame is None:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t_infer = (time.perf_counter() - t0) * 1000.0

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
            prev_time = curr_time

            annotated_frame = draw_detections(frame, detections, fps=fps, inference_time_ms=t_infer)

            if not args.headless:
                cv2.imshow("YOLO Scene Description Engine", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                desc = scene_engine.generate_scene_description(detections)
                print(f"[SCENE] {', '.join(desc)} | FPS: {fps:.1f}")

    finally:
        if cam_handle[0] == 'opencv':
            cam_handle[1].release()
        elif cam_handle[0] == 'picamera':
            cam_handle[1].stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
