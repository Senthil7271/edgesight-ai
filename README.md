# 👁️ EdgeSight AI — Edge Assistive Vision Suite for Raspberry Pi 5 & Hailo-8

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A.svg?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![NPU Accelerator](https://img.shields.io/badge/NPU-Hailo--8%20(26%20TOPS)-00C49F.svg)](https://hailo.ai/)
[![Inference Engine](https://img.shields.io/badge/Inference-PyHailoRT%20%7C%20ONNX%20Runtime-orange.svg)](https://hailo.ai/)
[![UI](https://img.shields.io/badge/Interface-Glassmorphic%20HUD-7928CA.svg)](http://localhost:5050)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**EdgeSight** is an edge-native, real-time AI assistive vision suite engineered for visually impaired and low-vision individuals. Running directly on the **Raspberry Pi 5 with Hailo-8 / Hailo-8L AI HAT+ (up to 26 TOPS)**, EdgeSight delivers zero-cloud, privacy-preserving visual awareness with up to 70+ FPS inference throughput and sub-20ms latency.

---

## 🌟 Key Features

### 1. 🎯 Quad AI Vision Modes
- **📦 Spatial Object Detection & Hazard Awareness (`YOLOv8n`)**
  - Identifies 80 COCO classes with spatial relationship reasoning (*e.g., "Person using laptop", "Bottle on dining table"*).
  - Evaluates relative bounding box positions to warn against approaching obstacles.
- **💵 Indian Currency Recognition (`YOLOv5 Custom`)**
  - High-precision detection for all 7 active Indian Rupee denominations: **₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000**.
  - Dynamic sum counter calculates total cash value visible in the camera frame.
- **📱 QR Code & Barcode Scanner (`OpenCV + PyZbar`)**
  - Real-time optical barcode and QR detection with automatic URL/text parsing.
  - Interactive web HUD actions: 1-click **Copy to Clipboard** and **Open URL Link**.
- **📖 Deep Learning OCR Text Reader (`RapidOCR ONNX` + `Tesseract`)**
  - Multi-stage document, book, sign, and label reader powered by deep-learning text detection (DBNet) and recognition (CRNN).
  - Audio text-to-speech reading with speech-rate controls and one-click copy.

### 2. 🎙️ Universal AI Voice Assistant & Speech STT
- Thread-safe voice processing endpoint (`/api/process_voice_audio`) using SpeechRecognition.
- Natural voice commands: *"Describe scene"*, *"Detect rupees"*, *"Read this sign"*, *"Scan QR code"*.
- Built-in encyclopedic Q&A engine with voice audio responses.

### 3. ⚡ Hardware Acceleration (Hailo-8 NPU vs. Pi 5 CPU)
| Accelerator | Runtime | Pure Inference Latency | Pure Throughput | End-to-End Latency | System FPS | CPU Load | Power |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hailo-8 PCIe HAT+ (26 TOPS)** | PyHailoRT (PCIe Gen 3) | **14.1 ms** | **70.9 FPS** | **18.5 ms** | **54.1 FPS** | **12%** | **2.5 W** |
| **Raspberry Pi 5 CPU (Cortex-A76)** | ONNX Runtime (ARM NEON) | 58.4 ms | 17.1 FPS | 68.2 ms | 14.7 FPS | 88% | 5.8 W |

---

## 📁 Repository Structure

```text
edgesight_ai/
├── main.py                     # Main application entry point
├── run.sh                      # 1-click execution launcher for Raspberry Pi
├── setup.sh                    # 1-click environment setup script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Production git ignore configuration
├── LICENSE                     # MIT License
│
├── src/                        # Core application source code
│   ├── app.py                  # Thread-safe camera server, web HUD & API
│   ├── ai_assistant.py         # Universal voice & text AI query assistant
│   ├── currency_detector.py    # Gowtham-369 Indian Currency detection engine
│   ├── hailo_inference.py      # PyHailoRT Hailo-8 (26 TOPS) NPU hardware pipeline
│   ├── ocr_reader.py           # Deep-learning RapidOCR ONNX & Tesseract OCR engine
│   ├── qr_barcode_scanner.py   # Optical QR & 1D barcode scanner
│   ├── scene_descriptor.py     # Spatial relationships & scene reasoning
│   ├── yolo_raspi_detector.py  # YOLOv8 ONNX / Hailo detection engine
│   └── model_utils.py          # Unified model path resolver utility
│
├── models/                     # Categorized neural network models (1 ONNX & 1 HEF per task)
│   ├── object_detection/
│   │   ├── yolov8n.onnx        # Object detection ONNX model
│   │   └── yolov8n.hef         # Hailo-8 compiled YOLOv8n binary (26 TOPS)
│   ├── currency_detection/
│   │   ├── indian_currency_best.onnx # Indian currency notes ONNX model
│   │   └── indian_currency_best.hef  # Hailo-8 compiled Indian currency binary (26 TOPS)
│   └── ocr/
│       ├── ocr_det.onnx        # Text detector ONNX model
│       └── ocr_det.hef         # Hailo-8 compiled text detector binary (26 TOPS)
│
├── templates/
│   └── index.html              # Modern glassmorphic HUD with voice, audio & mode switches
├── static/
│   ├── edgesight_model_performance_academic.png
│   └── edgesight_model_performance_graph.png
│
├── scripts/                    # Hardware verification & maintenance scripts
│   ├── setup_raspi_hailo.sh    # PCIe Gen 3 & driver installer
│   ├── run_hailo_assistant.sh  # Fullscreen monitor launcher
│   ├── check_installation.py   # Hardware & package diagnostic tool
│   ├── verify_hailo_26tops.sh  # Hailo-8 26 TOPS verification script
│   ├── compile_models_to_hailo.py # Hailo DFC compilation script
│   └── check_all.sh            # Complete system verification launcher
│
├── benchmarks/                 # Pi 5 CPU & Hailo-8 NPU benchmark suite
│   ├── benchmark_all_models.py # Multi-model comparative benchmark runner
│   ├── benchmark_cpu.py        # Pi 5 ARM Cortex-A76 CPU benchmark
│   ├── benchmark_hailo.py      # Hailo-8 NPU PCIe benchmark
│   ├── compare_benchmarks.py   # Comparative analyzer
│   ├── run_cpu_benchmark.sh
│   ├── run_hailo_benchmark.sh
│   └── run_full_comparison.sh
│
└── docs/                       # Comprehensive documentation
    ├── DEPLOYMENT_GUIDE_RASPBERRY_PI_HAILO.md
    └── EdgeSight_Project_Report.html
```

---

## 🚀 Deployment on Raspberry Pi 5 + Hailo-8 AI Kit

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/edgesight_ai.git
cd edgesight_ai
```

### Step 2: One-Click Environment Setup
Run the setup script to enable PCIe Gen 3, configure drivers, and install packages:
```bash
chmod +x setup.sh run.sh
./setup.sh
```
*(If prompted to reboot for PCIe Gen 3 enablement, run `sudo reboot` and navigate back to the folder).*

### Step 3: Launch EdgeSight
Connect your camera (Picamera2 or USB Webcam) and HDMI monitor:
```bash
./run.sh
```
The script will verify the Hailo-8 NPU PCIe HAT status, start the server on port **5050**, and open the full-screen browser HUD!

---

## 🔍 Hardware Diagnostics & Verification

Run the diagnostics utility at any time to verify camera, PCIe HAT, models, and libraries:
```bash
python scripts/check_installation.py
```

To run the multi-model CPU vs. Hailo-8 NPU benchmark evaluation:
```bash
python benchmarks/benchmark_all_models.py
```

---

## 🔒 Privacy & Edge Advantage
- **Zero Cloud Latency:** All neural network inference is computed strictly on-device.
- **Privacy Preserving:** No video frames or audio streams leave the local Raspberry Pi.
- **Offline Capable:** Full vision operation functions without internet connectivity.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
