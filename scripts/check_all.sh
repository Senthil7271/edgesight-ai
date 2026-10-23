#!/usr/bin/env bash
# ==============================================================================
# EdgeSight — Complete System Diagnostic & Verification Launcher
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x check_installation.py
python3 check_installation.py
