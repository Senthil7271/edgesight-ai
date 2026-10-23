#!/usr/bin/env python3
"""
EdgeSight — Edge AI Assistive Vision Application
Thread-Safe Camera Server & Web Interface
Supports Quad Modes:
  1) General Object Detection (YOLOv8) + Spatial Reasoning
  2) Indian Currency Notes Detection (Gowtham-369 YOLOv5)
  3) QR Code & Barcode Scanner Engine (OpenCV + PyZbar)
  4) OCR Text Reader (Tesseract) — Books, Signs, Labels, Documents
Includes Audio Upload Processing (/api/process_voice_audio) via SpeechRecognition
"""

import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import io
import cv2
import threading
import base64
import tempfile
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify

from yolo_raspi_detector import YOLOOnnxDetector, YOLOUltralyticsDetector, draw_detections, ensure_model_exists
from scene_descriptor import FrameSceneDescriptor
from ai_assistant import GeneralAIAssistant
from currency_detector import IndianCurrencyDetector, draw_currency_detections
from qr_barcode_scanner import QRBarcodeScanner, draw_scan_detections
from ocr_reader import OCRReader, draw_ocr_detections

import speech_recognition as sr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_utils import resolve_model_path

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

DEFAULT_MODEL = "yolov8n.onnx"
active_model_path = ensure_model_exists(DEFAULT_MODEL)
scene_engine = FrameSceneDescriptor()
ai_engine = GeneralAIAssistant()
recognizer = sr.Recognizer()

# Detectors & Engines
object_detector = None
currency_detector = None
code_scanner = None
ocr_engine = None

try:
    if active_model_path.endswith(".onnx"):
        object_detector = YOLOOnnxDetector(active_model_path, input_size=(320, 320), conf_thresh=0.25)
    else:
        object_detector = YOLOUltralyticsDetector(active_model_path, conf_thresh=0.25)
    print(f"[SERVER] Active object detector initialized: {active_model_path}")
except Exception as e:
    print(f"[SERVER] Object detector init error: {e}")
    object_detector = None

def get_currency_detector():
    global currency_detector
    if currency_detector is None:
        model_file = "indian_currency_best.onnx"
        print(f"[SERVER] Lazy-loading Indian Currency Detector with '{model_file}'...")
        currency_detector = IndianCurrencyDetector(model_path=model_file, conf_thresh=0.15)
    return currency_detector

def get_code_scanner():
    global code_scanner
    if code_scanner is None:
        print("[SERVER] Lazy-loading QR & Barcode Scanner...")
        code_scanner = QRBarcodeScanner()
    return code_scanner

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        print("[SERVER] Lazy-loading OCR Reader Engine (Tesseract)...")
        ocr_engine = OCRReader(lang="eng", min_confidence=40)
    return ocr_engine

current_settings = {
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 320,
    "mode": "object" # "object", "currency", "scanner", or "ocr"
}


class ThreadSafeCamera:
    def __init__(self, camera_index=None):
        import sys
        if camera_index is None:
            camera_index = os.getenv("CAMERA_INDEX", "0")
        self.camera_index = camera_index
        self.cam_type = 'opencv'
        self.cap = None
        self.picam = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.init_camera()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self, index):
        import sys
        if sys.platform.startswith("win"):
            # Use CAP_DSHOW directly on Windows to prevent MSMF backend hangs
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            # If DSHOW was not opened, attempt default without long timeout
            return cv2.VideoCapture(index)
        else:
            return cv2.VideoCapture(index)

    def init_camera(self):
        import sys
        idx_str = str(self.camera_index).lower()
        print(f"[CAMERA] Initializing camera device: {self.camera_index}...")

        # If explicitly requested or as fallback, check for Picamera2 (Raspberry Pi Camera Module)
        if idx_str == 'picamera':
            return self._init_picamera()

        try:
            numeric_idx = int(self.camera_index)
        except ValueError:
            numeric_idx = 0

        self.cap = self._open_capture(numeric_idx)

        # Fallback chain: if selected index fails, try index 0, then 1, then 2
        if not self.cap.isOpened():
            for fallback_idx in [0, 1, 2]:
                if fallback_idx != numeric_idx:
                    print(f"[CAMERA] Index {numeric_idx} failed, trying fallback index {fallback_idx}...")
                    self.cap = self._open_capture(fallback_idx)
                    if self.cap.isOpened():
                        self.camera_index = fallback_idx
                        numeric_idx = fallback_idx
                        break

        if self.cap.isOpened():
            self.cam_type = 'opencv'
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[SUCCESS] Camera device {self.camera_index} initialized successfully!")
            return True

        # If OpenCV failed and we are on Linux/Raspberry Pi, try Picamera2
        print(f"[CAMERA] OpenCV capture failed. Attempting Picamera2...")
        if self._init_picamera():
            return True

        print(f"[ERROR] Could not open any camera device!")
        return False

    def _init_picamera(self):
        try:
            from picamera2 import Picamera2
            print("[CAMERA] Initializing Picamera2 (Raspberry Pi Camera Module)...")
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(main={"size": (1280, 720)})
            self.picam.configure(config)
            self.picam.start()
            self.cam_type = 'picamera'
            self.camera_index = 'picamera'
            print("[SUCCESS] Picamera2 initialized successfully!")
            return True
        except Exception as e:
            print(f"[CAMERA] Picamera2 initialization failed: {e}")
            self.picam = None
            return False

    def switch_camera(self, new_index):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.picam:
                try:
                    self.picam.stop()
                    self.picam.close()
                except Exception:
                    pass
                self.picam = None
            self.camera_index = new_index
            return self.init_camera()

    def _update(self):
        while self.running:
            if self.cam_type == 'picamera' and self.picam:
                try:
                    frame = self.picam.capture_array()
                    if frame is not None:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        with self.lock:
                            self.latest_frame = bgr
                    else:
                        time.sleep(0.02)
                except Exception:
                    time.sleep(0.05)
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame.copy()
                else:
                    time.sleep(0.02)
            else:
                time.sleep(0.2)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Connecting to Camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)
        return False, blank


camera_server = ThreadSafeCamera()


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/get_live_frame', methods=['GET'])
def get_live_frame():
    success, frame = camera_server.get_frame()
    if success and frame is not None:
        h, w = frame.shape[:2]
        if w > 640:
            preview = cv2.resize(frame, (640, int(640 * h / w)))
        else:
            preview = frame
        ret, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    else:
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        'success': success,
        'camera_index': camera_server.camera_index,
        'image_b64': f"data:image/jpeg;base64,{img_b64}"
    })


@app.route('/api/switch_camera', methods=['POST'])
def switch_cam():
    data = request.json or {}
    raw_idx = data.get('camera_index', 0)
    try:
        idx = int(raw_idx)
    except (ValueError, TypeError):
        idx = str(raw_idx)
    success = camera_server.switch_camera(idx)
    return jsonify({'success': success, 'camera_index': camera_server.camera_index})


@app.route('/api/process_voice_audio', methods=['POST'])
def process_voice_audio():
    """Receives recorded audio file from browser, transcribes speech with Python SpeechRecognition"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 44:
            return jsonify({'success': False, 'error': 'Audio recording was empty. Please ensure microphone access and try again.'})

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data).strip()
            print(f"[VOICE STT] Transcribed Speech: '{text}'")

            text_lower = text.lower()
            is_scanner_cmd = any(word in text_lower for word in ["qr", "barcode", "scan", "scanner", "code", "barcode", "link", "url"])
            is_ocr_cmd = not is_scanner_cmd and any(word in text_lower for word in ["read", "text", "sign", "ocr", "book", "page", "label", "notice", "document", "letter", "what does this say", "read this"])
            is_currency_cmd = not is_scanner_cmd and not is_ocr_cmd and any(word in text_lower for word in ["rupee", "rupees", "currency", "money", "cash", "note", "notes"])
            is_capture_cmd = is_scanner_cmd or is_ocr_cmd or is_currency_cmd or any(word in text_lower for word in ["capture", "photo", "picture", "snap", "describe", "take"])

            if is_scanner_cmd:
                target_mode = "scanner"
            elif is_ocr_cmd:
                target_mode = "ocr"
            elif is_currency_cmd:
                target_mode = "currency"
            else:
                target_mode = current_settings["mode"]

            return jsonify({
                'success': True,
                'transcript': text,
                'is_capture_command': is_capture_cmd,
                'is_currency_command': is_currency_cmd,
                'is_scanner_command': is_scanner_cmd,
                'is_ocr_command': is_ocr_cmd,
                'target_mode': target_mode
            })

    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Could not understand audio. Please speak clearly.'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Speech recognition error: {str(e)}'})
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass


@app.route('/api/ask_ai', methods=['POST'])
def ask_ai():
    data = request.json or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    answer = ai_engine.ask(question)
    return jsonify({
        'success': True,
        'question': question,
        'answer': answer
    })


@app.route('/api/describe_current_frame', methods=['POST'])
def describe_current_frame():
    success, frame = camera_server.get_frame()

    if not success or frame is None:
        return jsonify({'error': 'Camera frame not available.'}), 500

    req_data = request.json or {}
    conf = float(req_data.get('conf', current_settings['conf']))
    mode = req_data.get('mode', current_settings['mode'])

    # 0. OCR Text Reader Mode
    if mode == "ocr":
        reader = get_ocr_engine()
        # Determine OCR strategy from request
        ocr_strategy = req_data.get('ocr_strategy', 'auto')  # 'auto', 'sign', 'document', 'simple'

        t0 = time.perf_counter()
        ocr_result = reader.extract_text(frame, strategy=ocr_strategy)
        t_infer = (time.perf_counter() - t0) * 1000.0

        voice_summary = reader.summarize_for_voice(ocr_result)
        annotated = draw_ocr_detections(
            frame, ocr_result,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'mode': 'ocr',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': voice_summary['summary'],
            'speech_text': voice_summary['speech_text'],
            'extracted_text': ocr_result['text'],
            'word_count': ocr_result['word_count'],
            'line_count': ocr_result['line_count'],
            'avg_confidence': ocr_result['avg_confidence'],
            'is_sign': voice_summary['is_sign'],
            'has_text': voice_summary['has_text'],
            'relationships': [voice_summary['summary']],
            'object_count': ocr_result['word_count'],
            'strategy': ocr_strategy,
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })

    # 1. QR Code & Barcode Scanner Mode
    elif mode == "scanner":
        scanner = get_code_scanner()
        t0 = time.perf_counter()
        scan_results = scanner.scan(frame)
        t_infer = (time.perf_counter() - t0) * 1000.0

        summary_info = scanner.summarize_scan(scan_results)
        annotated = draw_scan_detections(
            frame, scan_results,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        rel_tags = []
        for item in summary_info['details']:
            rel_tags.append(f"{item['type']}: {item['data']}")

        if not rel_tags:
            rel_tags = ["No codes found"]

        return jsonify({
            'success': True,
            'mode': 'scanner',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': summary_info['summary'],
            'relationships': rel_tags,
            'object_count': summary_info['count'],
            'details': summary_info['details'],
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })

    # 2. Indian Currency Notes Detection Mode
    elif mode == "currency":
        c_detector = get_currency_detector()
        t0 = time.perf_counter()
        detections = c_detector.detect(frame, conf_thresh=conf)
        t_infer = (time.perf_counter() - t0) * 1000.0

        summary_info = c_detector.summarize_currency(detections)
        annotated = draw_currency_detections(
            frame, detections,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer,
            total_value=summary_info['total_value']
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        rel_tags = [f"Total Value: Rs. {summary_info['total_value']}"]
        for name, cnt in summary_info['note_counts'].items():
            rel_tags.append(f"{cnt}x {name}")

        return jsonify({
            'success': True,
            'mode': 'currency',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': summary_info['summary'],
            'relationships': rel_tags,
            'object_count': len(detections),
            'total_value': summary_info['total_value'],
            'counts': summary_info['note_counts'],
            'detections': detections,
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })

    # 3. General Object Detection Mode
    else:
        if object_detector:
            object_detector.conf_thresh = conf
            t0 = time.perf_counter()
            detections = object_detector.detect(frame)
            t_infer = (time.perf_counter() - t0) * 1000.0

            analysis = scene_engine.describe_frame(detections, frame.shape[:2])
            annotated = draw_detections(frame, detections, fps=1000.0/max(1.0, t_infer), inference_time_ms=t_infer)
            
            _, buffer = cv2.imencode('.jpg', annotated)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'mode': 'object',
                'inference_time_ms': round(t_infer, 1),
                'narrative_summary': analysis['summary'],
                'relationships': analysis['relationships'],
                'object_count': analysis['object_count'],
                'counts': analysis['counts'],
                'detections': detections,
                'image_b64': f"data:image/jpeg;base64,{img_b64}"
            })
        return jsonify({'error': 'Object detector not initialized'}), 500


@app.route('/api/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    conf = float(request.form.get('conf', current_settings['conf']))
    mode = request.form.get('mode', current_settings['mode'])

    if mode == "ocr":
        reader = get_ocr_engine()
        ocr_strategy = request.form.get('ocr_strategy', 'auto')

        t0 = time.perf_counter()
        ocr_result = reader.extract_text(cv_img, strategy=ocr_strategy)
        t_infer = (time.perf_counter() - t0) * 1000.0

        voice_summary = reader.summarize_for_voice(ocr_result)
        annotated = draw_ocr_detections(
            cv_img, ocr_result,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'mode': 'ocr',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': voice_summary['summary'],
            'speech_text': voice_summary['speech_text'],
            'extracted_text': ocr_result['text'],
            'word_count': ocr_result['word_count'],
            'line_count': ocr_result['line_count'],
            'avg_confidence': ocr_result['avg_confidence'],
            'is_sign': voice_summary['is_sign'],
            'has_text': voice_summary['has_text'],
            'relationships': [voice_summary['summary']],
            'object_count': ocr_result['word_count'],
            'strategy': ocr_strategy,
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })
    elif mode == "scanner":
        scanner = get_code_scanner()
        t0 = time.perf_counter()
        scan_results = scanner.scan(cv_img)
        t_infer = (time.perf_counter() - t0) * 1000.0

        summary_info = scanner.summarize_scan(scan_results)
        annotated = draw_scan_detections(
            cv_img, scan_results,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        rel_tags = []
        for item in summary_info['details']:
            rel_tags.append(f"{item['type']}: {item['data']}")

        if not rel_tags:
            rel_tags = ["No codes found"]

        return jsonify({
            'success': True,
            'mode': 'scanner',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': summary_info['summary'],
            'relationships': rel_tags,
            'object_count': summary_info['count'],
            'details': summary_info['details'],
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })
    elif mode == "currency":
        c_detector = get_currency_detector()
        t0 = time.perf_counter()
        detections = c_detector.detect(cv_img, conf_thresh=conf)
        t_infer = (time.perf_counter() - t0) * 1000.0

        summary_info = c_detector.summarize_currency(detections)
        annotated = draw_currency_detections(
            cv_img, detections,
            fps=1000.0/max(1.0, t_infer),
            inference_time_ms=t_infer,
            total_value=summary_info['total_value']
        )

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        rel_tags = [f"Total Value: Rs. {summary_info['total_value']}"]
        for name, cnt in summary_info['note_counts'].items():
            rel_tags.append(f"{cnt}x {name}")

        return jsonify({
            'success': True,
            'mode': 'currency',
            'inference_time_ms': round(t_infer, 1),
            'narrative_summary': summary_info['summary'],
            'relationships': rel_tags,
            'object_count': len(detections),
            'total_value': summary_info['total_value'],
            'counts': summary_info['note_counts'],
            'detections': detections,
            'image_b64': f"data:image/jpeg;base64,{img_b64}"
        })
    else:
        if object_detector:
            object_detector.conf_thresh = conf
            t0 = time.perf_counter()
            detections = object_detector.detect(cv_img)
            t_infer = (time.perf_counter() - t0) * 1000.0

            analysis = scene_engine.describe_frame(detections, cv_img.shape[:2])
            annotated = draw_detections(cv_img, detections, fps=1000.0/max(1.0, t_infer), inference_time_ms=t_infer)
            
            _, buffer = cv2.imencode('.jpg', annotated)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'mode': 'object',
                'inference_time_ms': round(t_infer, 1),
                'narrative_summary': analysis['summary'],
                'relationships': analysis['relationships'],
                'object_count': analysis['object_count'],
                'counts': analysis['counts'],
                'detections': detections,
                'image_b64': f"data:image/jpeg;base64,{img_b64}"
            })
        return jsonify({'error': 'Detector not initialized'}), 500


@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json or {}
    if 'conf' in data:
        current_settings['conf'] = float(data['conf'])
    if 'mode' in data:
        current_settings['mode'] = data['mode']
    return jsonify({'status': 'ok', 'settings': current_settings})


@app.route('/api/tts_speak', methods=['POST'])
def tts_speak():
    data = request.json or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'})

    # Clean text and speak via Raspberry Pi hardware audio
    import subprocess
    clean_words = ' '.join(text.split()[:40])
    try:
        subprocess.Popen(['espeak-ng', '-s', '150', '-v', 'en-us', clean_words],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({'success': True, 'engine': 'espeak-ng'})
    except Exception:
        try:
            subprocess.Popen(['espeak', '-s', '150', clean_words],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return jsonify({'success': True, 'engine': 'espeak'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("Starting EdgeSight Server (Object YOLOv8 + Currency + QR/Barcode + OCR Tesseract) at http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
