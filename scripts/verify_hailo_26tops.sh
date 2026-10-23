#!/usr/bin/env bash
# ==============================================================================
# Hailo-8 (26 AI TOPS) Hardware & Driver Diagnostic Verification Tool
# For Raspberry Pi 5 + Raspberry Pi AI HAT+ (26 TOPS)
# ==============================================================================

echo "=============================================================================="
echo "🔍 Checking Hailo-8 AI HAT+ (26 TOPS) Hardware & Driver Connection..."
echo "=============================================================================="

ERRORS=0

# 1. Check if lspci is installed
if ! command -v lspci &> /dev/null; then
    echo "Installing pciutils..."
    sudo apt update -qq && sudo apt install -y pciutils
fi

# 2. Layer 1: PCIe Bus Hardware Link
echo -n "[1/4] Checking PCIe Hardware Link (lspci)... "
PCI_INFO=$(lspci | grep -i "Hailo" || true)
if [ -n "$PCI_INFO" ]; then
    echo "✅ PASSED"
    echo "      Found: $PCI_INFO"
else
    echo "❌ FAILED"
    echo "      [ERROR] Hailo PCIe device is NOT detected on the PCIe bus."
    echo "      Common fixes:"
    echo "      1. Check the PCIe ribbon cable! Ensure contacts face the correct direction"
    echo "         and the latch is firmly clamped on BOTH the Pi 5 and the AI HAT+."
    echo "      2. Ensure 'dtparam=pciex1' is in /boot/firmware/config.txt and you rebooted."
    ERRORS=$((ERRORS + 1))
fi

# 3. Layer 2: Kernel Module & Device Node
echo -n "[2/4] Checking Kernel Driver & /dev/hailo0... "
if [ -e "/dev/hailo0" ]; then
    echo "✅ PASSED"
    ls -l /dev/hailo0
else
    echo "❌ FAILED"
    echo "      [ERROR] /dev/hailo0 device node does not exist."
    echo "      Driver status from dmesg:"
    dmesg | grep -i hailo | tail -n 5 || echo "      No Hailo kernel messages found."
    echo "      Run: sudo apt install -y hailo-all && sudo reboot"
    ERRORS=$((ERRORS + 1))
fi

# 4. Layer 3: HailoRT CLI Identification & 26 TOPS Verification
echo -n "[3/4] Querying Hailo Firmware & Architecture... "
if command -v hailortcli &> /dev/null; then
    FW_INFO=$(hailortcli fw-control identify 2>&1 || true)
    if echo "$FW_INFO" | grep -qi "Device Architecture.*HAILO8"; then
        echo "✅ PASSED"
        echo "------------------------------------------------------------------------------"
        echo "$FW_INFO"
        echo "------------------------------------------------------------------------------"
        if echo "$FW_INFO" | grep -qi "HAILO8L"; then
            echo "ℹ️  Device detected as: Hailo-8L (13 TOPS variant)."
        else
            echo "🚀 CONFIRMED: Full Hailo-8 (26 AI TOPS) detected and active!"
        fi
    elif echo "$FW_INFO" | grep -qi "Firmware Version"; then
        echo "✅ PASSED (Device responded)"
        echo "$FW_INFO"
    else
        echo "❌ FAILED to query Hailo firmware."
        echo "      Error response: $FW_INFO"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️ hailortcli not found."
    echo "      Install it via: sudo apt install -y hailo-all"
    ERRORS=$((ERRORS + 1))
fi

# 5. Layer 4: Scan and Functional Test
echo -n "[4/4] Testing NPU Communication (hailortcli scan)... "
if command -v hailortcli &> /dev/null; then
    SCAN_OUTPUT=$(hailortcli scan 2>&1 || true)
    if echo "$SCAN_OUTPUT" | grep -qi "Hailo-8"; then
        echo "✅ PASSED (NPU Communication OK)"
    else
        echo "⚠️ $SCAN_OUTPUT"
    fi
fi

echo "=============================================================================="
if [ $ERRORS -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED! Your Hailo-8 (26 TOPS) NPU is fully operational!"
else
    echo "⚠️ Found $ERRORS issue(s). Please review the steps above."
fi
echo "=============================================================================="
