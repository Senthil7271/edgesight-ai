# 🚀 EdgeSight — Raspberry Pi 5 + Hailo-8/8L AI HAT Deployment Guide

Complete step-by-step guide for deploying **EdgeSight** (General YOLO Objects, Gowtham-369 Indian Currency Notes, QR/Barcode Scanner, OCR Text Reader, Web Voice STT & Universal AI) on a **Raspberry Pi 5 with the Raspberry Pi AI Kit (Hailo-8 / Hailo-8L 26 TOPS PCIe HAT)** connected to an HDMI monitor.

---

## 📦 What's Included in This Package

The `Hailo_RaspberryPi_Vision_Assistant` folder contains everything needed:

- **`app.py`**: Thread-safe Flask application supporting Picamera2, Hailo NPU hardware, OpenCV V4L2 fallback, OCR Tesseract, and Web STT + Universal AI Q&A.
- **`hailo_inference.py`**: PyHailoRT (`hailo_platform`) NPU hardware inference engine for Hailo-8/8L PCIe HAT.
- **`yolo_raspi_detector.py`**: Model detection abstraction layer.
- **`currency_detector.py`**: Gowtham-369 Indian Currency Notes detection engine.
- **`qr_barcode_scanner.py`**: QR Code and Barcode scanner engine.
- **`ocr_reader.py`**: Tesseract OCR text extraction engine for books, signs, labels and documents.
- **`scene_descriptor.py`**: Spatial relationship reasoning engine.
- **`ai_assistant.py`**: Universal AI Assistant engine (Llama / Wikipedia API).
- **`templates/index.html`**: EdgeSight Glassmorphic Web UI with mode switcher, OCR panel, copy buttons, prompt chips, and voice TTS.
- **`indian_currency_best.pt`**: Gowtham-369 custom Indian currency model weights.
- **`yolov8n.onnx`**: Optimized 320x320 ONNX ARM NEON model.
- **`setup_raspi_hailo.sh`**: One-click setup script.
- **`run_hailo_assistant.sh`**: One-click execution script.

---

## 🛠️ Step 1: Navigate to Project Folder on Raspberry Pi

Open a Terminal on your Raspberry Pi and enter your `edgesight` project directory:

```bash
# If placed in home directory:
cd ~/edgesight

# Or if placed on Desktop:
cd ~/Desktop/edgesight
```

---

## ⚙️ Step 2: One-Click Environment Setup

Run the setup script to install Hailo AI Kit PCIe drivers, Picamera2 camera support, Tesseract OCR, OpenCV, and Python dependencies:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

*(If prompted to reboot for PCIe Gen 3 enablement, run `sudo reboot` and navigate back to the folder after reboot).*

---

## 🎬 Step 3: Run the Application (HDMI Monitor or Headless)

Connect your camera (Picamera2 Module v2/v3 or USB Webcam) and HDMI monitor, then execute:

```bash
./run.sh
```

### What Happens:
1. Verifies Hailo-8 NPU PCIe HAT status (`hailortcli fw-control identify`).
2. Starts the thread-safe web server on port **`5050`**.
3. Automatically launches the full-screen browser interface on your attached HDMI monitor!

---

## 🔌 Accessing from Laptop / Mobile Browser
You can also connect to the web UI from any laptop or mobile phone connected to the same Wi-Fi network:

```text
http://<RASPBERRY_PI_IP_ADDRESS>:5050
```

---

## ⚡ Optional: Enable Autostart on Boot

To make the AI Vision Assistant launch automatically whenever the Raspberry Pi powers on, run:

```bash
sudo systemctl enable --now hailo-vision.service
```

To stop or disable autostart:
```bash
sudo systemctl disable --now hailo-vision.service
```

---

## 🎯 Usage Summary

- **📦 General Object Mode**: Click **`📸 SNAP & DESCRIBE`** or say *"Capture photo"* for spatial object analysis.
- **💵 Indian Currency Mode**: Click **`💵 DETECT RUPEES`** or say *"Detect rupees"* — runs Gowtham-369 model for ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000 notes and sums the total amount.
- **📱 QR Code & Barcode Mode**: Click **`📱 SCAN QR / BARCODE`** or say *"Scan QR code"* — decodes text/URLs with 1-click `📋 COPY TEXT` and `🔗 OPEN LINK` buttons!
- **📖 OCR Text Reader Mode**: Click **`📖 READ TEXT`** or say *"Read this sign"* / *"Read this page"* — extracts text from books, signs, labels and documents with `📋 COPY TEXT` and `🔊 READ ALOUD` buttons!
- **🤖 Universal AI Q&A**: Ask any question (*"What is a processor?"*, *"DSP"*) by voice or text.
