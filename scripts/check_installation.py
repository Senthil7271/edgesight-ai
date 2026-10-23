#!/usr/bin/env python3
"""
EdgeSight — System & Dependency Verification Utility
Tests all hardware, drivers, Python libraries, camera devices, and AI models
for Raspberry Pi 5 + Hailo-8 deployment.
"""

import sys
import os
import subprocess
import importlib

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ANSI Color codes for clean terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

pass_count = 0
fail_count = 0
warn_count = 0

def report_pass(name, details=""):
    global pass_count
    pass_count += 1
    det = f" ({details})" if details else ""
    print(f"  {GREEN}✔ [PASS]{RESET} {BOLD}{name}{RESET}{det}")

def report_fail(name, advice=""):
    global fail_count
    fail_count += 1
    print(f"  {RED}✘ [FAIL]{RESET} {BOLD}{name}{RESET}")
    if advice:
        print(f"         {YELLOW}Fix: {advice}{RESET}")

def report_warn(name, details="", advice=""):
    global warn_count
    warn_count += 1
    det = f" ({details})" if details else ""
    print(f"  {YELLOW}⚠ [WARN]{RESET} {BOLD}{name}{RESET}{det}")
    if advice:
        print(f"         {CYAN}Note: {advice}{RESET}")

def check_command(cmd):
    try:
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return False, "", str(e)

print(f"\n{BOLD}{CYAN}=============================================================================={RESET}")
print(f"{BOLD}{CYAN}       🔍 EDGESIGHT — RASPBERRY PI 5 & HAILO-8 SYSTEM VERIFICATION           {RESET}")
print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")

# ------------------------------------------------------------------------------
# 1. Hardware & Hailo-8 NPU Check
# ------------------------------------------------------------------------------
print(f"{BOLD}1. Hailo-8 NPU & PCIe Hardware Interface:{RESET}")

# lspci
ok, out, _ = check_command("lspci | grep -i hailo")
if ok and "hailo" in out.lower():
    report_pass("PCIe Bus Detection (lspci)", out.strip())
else:
    report_fail("PCIe Bus Detection (lspci)", "Check PCIe ribbon cable contacts and ensure 'dtparam=pciex1' in /boot/firmware/config.txt")

# /dev/hailo0
if os.path.exists("/dev/hailo0"):
    report_pass("Kernel Device Node (/dev/hailo0)", "Driver active")
else:
    report_fail("Kernel Device Node (/dev/hailo0)", "Run: sudo apt install -y hailo-all && sudo reboot")

# hailortcli
ok, out, _ = check_command("hailortcli fw-control identify")
if ok and ("HAILO8" in out or "Firmware Version" in out):
    report_pass("Hailo Firmware & NPU Communication", "Hailo-8 26 TOPS Ready")
else:
    if check_command("which hailortcli")[0]:
        report_warn("Hailo Firmware Query", "Device did not respond to hailortcli identify", "Check reboot status after driver install")
    else:
        report_fail("HailoRT CLI Utility", "Run: sudo apt install -y hailo-all")

# hailo_platform Python module
try:
    import hailo_platform
    report_pass("PyHailoRT Python Bindings (hailo_platform)", "NPU acceleration active")
except ImportError:
    report_warn("PyHailoRT Python Bindings (hailo_platform)", "Not loaded in current Python", "Installed via 'sudo apt install python3-hailort'")

print()

# ------------------------------------------------------------------------------
# 2. Camera Devices Check
# ------------------------------------------------------------------------------
print(f"{BOLD}2. Camera Input Devices:{RESET}")

# V4L2 USB Devices
ok, out, _ = check_command("ls /dev/video* 2>/dev/null")
if ok and out:
    devices = out.split()
    report_pass("V4L2 Video Devices", f"Found: {', '.join(devices[:3])}")
else:
    report_warn("V4L2 Video Devices", "No /dev/video* devices found", "If using USB webcam, ensure it is plugged into a USB 3.0 port")

# Picamera2
try:
    import picamera2
    report_pass("Picamera2 (Raspberry Pi Camera Module)", "Library installed")
except ImportError:
    report_warn("Picamera2 Library", "Not installed", "Install via: sudo apt install -y python3-picamera2")

# OpenCV VideoCapture probe
try:
    import cv2
    opened = False
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                report_pass("Live Camera Feed Probe", f"OpenCV successfully captured frame from index {idx} ({frame.shape[1]}x{frame.shape[0]})")
                opened = True
                break
            cap.release()
    if not opened:
        report_warn("Live Camera Feed Probe", "No active camera stream returned frames", "Camera will be dynamically acquired when server starts")
except Exception as e:
    report_warn("Live Camera Feed Probe", str(e))

print()

# ------------------------------------------------------------------------------
# 3. Core Computer Vision & AI Python Packages
# ------------------------------------------------------------------------------
print(f"{BOLD}3. Core Computer Vision & AI Python Packages:{RESET}")

packages = [
    ("OpenCV", "cv2", "sudo apt install -y python3-opencv"),
    ("NumPy", "numpy", "sudo apt install -y python3-numpy"),
    ("Pillow", "PIL", "sudo apt install -y python3-pil"),
    ("Flask Web Framework", "flask", "sudo apt install -y python3-flask"),
    ("ONNX Runtime", "onnxruntime", "pip3 install onnxruntime --break-system-packages"),
    ("RapidOCR Engine", "rapidocr_onnxruntime", "pip3 install rapidocr-onnxruntime --break-system-packages"),
    ("PyZbar (Barcodes/QR)", "pyzbar", "pip3 install pyzbar --break-system-packages"),
    ("PyTesseract", "pytesseract", "pip3 install pytesseract --break-system-packages"),
    ("Speech Recognition", "speech_recognition", "pip3 install SpeechRecognition --break-system-packages"),
    ("Requests", "requests", "sudo apt install -y python3-requests")
]

for display_name, mod_name, install_cmd in packages:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "installed")
        report_pass(display_name, f"v{ver}")
    except ImportError:
        report_fail(display_name, install_cmd)

# Optional PyTorch for Indian Currency
try:
    import torch
    report_pass("PyTorch (Indian Currency YOLOv5)", f"v{torch.__version__}")
except ImportError:
    report_warn("PyTorch (Indian Currency YOLOv5)", "Not installed (Optional)", "Run: pip3 install torch torchvision --break-system-packages")

print()

# ------------------------------------------------------------------------------
# 4. System Binary Dependencies
# ------------------------------------------------------------------------------
print(f"{BOLD}4. System Command & Binary Dependencies:{RESET}")

binaries = [
    ("Tesseract OCR Binary", "tesseract --version", "sudo apt install -y tesseract-ocr tesseract-ocr-eng"),
    ("ZBar Tools", "zbarimg --version || which zbarimg", "sudo apt install -y zbar-tools libzbar0"),
    ("FFmpeg Audio/Video", "ffmpeg -version", "sudo apt install -y ffmpeg"),
    ("Chromium Browser", "chromium-browser --version || chromium --version", "sudo apt install -y chromium-browser")
]

for name, cmd, install_cmd in binaries:
    ok, out, _ = check_command(cmd)
    if ok:
        first_line = out.split("\n")[0] if out else "installed"
        report_pass(name, first_line[:40])
    else:
        report_fail(name, install_cmd)

print()

# ------------------------------------------------------------------------------
# 5. Model Files & Application Assets
# ------------------------------------------------------------------------------
print(f"{BOLD}5. Model Files & Web Assets in Current Folder:{RESET}")

model_files = [
    ("models/object_detection/yolov8n.onnx", "YOLOv8 ONNX Object Detection Model", 10_000_000),
    ("models/object_detection/yolov8n.hef", "YOLOv8 Hailo-8 HEF Accelerator Model", 5_000_000),
    ("models/currency_detection/indian_currency_best.onnx", "Indian Currency Notes ONNX Model", 10_000_000),
    ("models/currency_detection/indian_currency_best.hef", "Indian Currency Hailo-8 HEF Accelerator Model", 5_000_000),
    ("models/ocr/ocr_det.onnx", "OCR Text Detector ONNX Model", 1_000_000),
    ("models/ocr/ocr_det.hef", "OCR Text Detector Hailo-8 HEF Model", 1_000_000),
    ("templates/index.html", "EdgeSight Web User Interface Template", 1_000),
    ("src/app.py", "Core Server Application", 5_000),
    ("main.py", "Main Entry Point", 500),
    ("run.sh", "Raspberry Pi Launch Script", 200)
]

for rel_path, desc, min_size in model_files:
    # Check directly, or relative to project root if run from scripts/
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path)
    ]
    resolved_path = None
    for cand in candidates:
        if os.path.exists(cand):
            resolved_path = cand
            break

    if resolved_path:
        sz = os.path.getsize(resolved_path)
        if sz >= min_size:
            report_pass(f"{desc} ({rel_path})", f"{sz / (1024*1024):.1f} MB" if sz > 1024*1024 else f"{sz / 1024:.1f} KB")
        else:
            report_warn(f"{desc} ({rel_path})", f"File size unexpectedly small: {sz} bytes")
    else:
        report_fail(f"{desc} ({rel_path})", "File missing in project directory!")

print()
print(f"{BOLD}{CYAN}=============================================================================={RESET}")
print(f"  Summary: {GREEN}{pass_count} Passed{RESET} | {YELLOW}{warn_count} Warnings{RESET} | {RED}{fail_count} Failed{RESET}")
if fail_count == 0:
    print(f"  {BOLD}{GREEN}🎉 All critical components are INSTALLED and READY to run!{RESET}")
    print(f"  Start EdgeSight now: {BOLD}./run_hailo_assistant.sh{RESET}")
else:
    print(f"  {BOLD}{YELLOW}👉 Run the recommended fix commands above, or re-run: ./setup_raspi_hailo.sh{RESET}")
print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")
