#!/usr/bin/env bash
# ==============================================================================
# EdgeSight AI — Raspberry Pi 5 & Hailo-8 AI Setup Script
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x scripts/setup_raspi_hailo.sh
exec ./scripts/setup_raspi_hailo.sh "$@"
