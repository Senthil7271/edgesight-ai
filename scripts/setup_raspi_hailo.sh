#!/usr/bin/env bash
# ==============================================================================
# One-Click Setup Script for Raspberry Pi 5 / 4 with Hailo-8/8L AI HAT & Monitor
# Targeted for Raspberry Pi OS 64-bit (Bookworm)
# ==============================================================================

set -e

echo "=============================================================================="
echo "🚀 Raspberry Pi Hailo-8 AI Vision Assistant Setup"
echo "=============================================================================="

# 1. Update System & Install Hailo AI Kit Drivers, System Python & Camera Support
echo "[1/4] Installing Hailo PCIe drivers, Picamera2, Tesseract OCR & System APT packages..."
sudo apt update
sudo apt install -y hailo-all rpicam-apps python3-pip python3-flask python3-opencv python3-numpy python3-pil python3-requests zbar-tools libzbar0 ffmpeg chromium-browser tesseract-ocr tesseract-ocr-eng pciutils

# 2. Configure Raspberry Pi PCIe Gen 3 Speed for Hailo HAT (Raspberry Pi 5)
echo "[2/4] Checking PCIe settings in /boot/firmware/config.txt..."
CONFIG_FILE="/boot/firmware/config.txt"
if [ -f "$CONFIG_FILE" ]; then
    if ! grep -q "dtparam=pciex1" "$CONFIG_FILE"; then
        echo "Adding dtparam=pciex1 and dtparam=pciex1_gen=3 to config.txt..."
        echo "# Hailo-8 AI HAT PCIe Configuration" | sudo tee -a "$CONFIG_FILE"
        echo "dtparam=pciex1" | sudo tee -a "$CONFIG_FILE"
        echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE"
    fi
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 3. Install Minimal Python Requirements
echo "[3/4] Installing remaining lightweight Python packages..."
pip3 install -r requirements.txt --break-system-packages || pip3 install -r requirements.txt

# 4. Make scripts executable
CURRENT_USER=$(whoami)
chmod +x run.sh setup.sh scripts/*.sh scripts/*.py

# 5. Create Systemd Service for Autostart on Boot (Optional)
SERVICE_FILE="/etc/systemd/system/hailo-vision.service"

echo "[4/4] Creating Autostart systemd service..."
sudo bash -c "cat <<EOF > $SERVICE_FILE
[Unit]
Description=Raspberry Pi Hailo-8 AI Vision Assistant Server
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=\"$DIR\"
ExecStart=/bin/bash \"$DIR/run.sh\"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
echo "Systemd service created: 'hailo-vision.service'"
echo "To enable autostart on boot run: sudo systemctl enable --now hailo-vision.service"

echo "=============================================================================="
echo "✅ SETUP COMPLETE!"
echo "To start the application on your monitor, run:"
echo "   ./run.sh"
echo "=============================================================================="
