#!/usr/bin/env bash
# ==============================================================================
# One-Click Execution Script for Raspberry Pi Hailo Vision Assistant
# Launches Python Server & Opens Fullscreen Monitor Interface
# ==============================================================================

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=============================================================================="
echo "🎥 Starting Raspberry Pi Hailo AI Vision Assistant..."
echo "=============================================================================="

# Check Hailo PCIe HAT state if hailortcli is present
if command -v hailortcli &> /dev/null; then
    echo "⚡ Checking Hailo NPU Status..."
    hailortcli fw-control identify || echo "[NOTE] Hailo HAT not responding yet, falling back to CPU/ONNX mode."
fi

# Launch Python Server in Background
python3 main.py &
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

# If running with a monitor/display attached (X11 or Wayland), launch browser
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    echo "📺 Opening interface on HDMI Monitor..."
    chromium-browser --password-store=basic --no-default-browser-check --app=http://localhost:5050 --start-fullscreen || xdg-open http://localhost:5050 || echo "Opened http://localhost:5050"
else
    echo "🌐 No display attached. Access interface from laptop browser at http://$(hostname -I | awk '{print $1}'):5050"
fi

# Keep script active
wait $SERVER_PID
