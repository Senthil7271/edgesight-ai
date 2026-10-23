#!/usr/bin/env bash
# ==============================================================================
# EdgeSight — One-Click Hailo-8 NPU Benchmark Runner
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x benchmark_hailo.py
python3 benchmark_hailo.py
