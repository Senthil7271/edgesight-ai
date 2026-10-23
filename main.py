#!/usr/bin/env python3
"""
EdgeSight — Assistive Edge AI Vision Suite (Raspberry Pi 5 + Hailo-8 NPU)
Main Application Entry Point
"""

import os
import sys
import argparse

# Ensure src/ is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.app import app

def main():
    parser = argparse.ArgumentParser(description="EdgeSight AI Vision Server (Raspberry Pi & Hailo-8)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)), help="Port (default: 5050)")
    parser.add_argument("--camera", type=str, default=os.environ.get("CAMERA_INDEX", "0"), help="Camera index or 'picamera'")
    args = parser.parse_args()

    os.environ["CAMERA_INDEX"] = str(args.camera)

    print("==============================================================================")
    print("🚀 Starting EdgeSight AI Vision Server (Raspberry Pi 5 + Hailo-8 NPU)")
    print(f"🌐 Web Interface: http://localhost:{args.port} or http://<PI_IP_ADDRESS>:{args.port}")
    print("==============================================================================")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
