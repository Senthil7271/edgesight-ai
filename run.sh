#!/usr/bin/env bash
# ==============================================================================
# EdgeSight AI — Raspberry Pi 5 & Hailo-8 AI Launcher
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=============================================================================="
echo "🎥 Launching EdgeSight AI Vision Suite on Raspberry Pi..."
echo "=============================================================================="

# Check Hailo PCIe HAT state if hailortcli is available
if command -v hailortcli &> /dev/null; then
    echo "⚡ Checking Hailo-8 NPU Status..."
    hailortcli fw-control identify || echo "[NOTE] Hailo HAT initializing or falling back to CPU mode."
fi

# Launch Python Server
python3 main.py "$@" &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Waiting for web interface to become ready on http://localhost:5050..."

for i in {1..15}; do
    if curl -s http://localhost:5050/api/get_live_frame > /dev/null; then
        echo "✅ Web Server is LIVE at http://localhost:5050!"
        break
    fi
    sleep 1
done

# If monitor is attached, launch fullscreen browser
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    echo "📺 Opening interface on HDMI Monitor..."
    chromium-browser --password-store=basic --no-default-browser-check --app=http://localhost:5050 --start-fullscreen || xdg-open http://localhost:5050 || echo "Opened http://localhost:5050"
else
    echo "🌐 No display attached. Access interface from browser at http://$(hostname -I | awk '{print $1}'):5050"
fi

wait $SERVER_PID
