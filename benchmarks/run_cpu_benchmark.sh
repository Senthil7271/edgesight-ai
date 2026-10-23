#!/usr/bin/env bash
# ==============================================================================
# EdgeSight — One-Click CPU Benchmark Runner
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x benchmark_cpu.py
python3 benchmark_cpu.py
