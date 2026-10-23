#!/usr/bin/env bash
# ==============================================================================
# EdgeSight — Full Automated Benchmark Suite (CPU + Hailo-8 + Side-by-Side)
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="$( cd "$DIR/.." >/dev/null 2>&1 && pwd )"

echo "=============================================================================="
echo "🏁 Running EdgeSight Automated Benchmark Suite"
echo "=============================================================================="

# 1. Run CPU Benchmark
echo ""
echo "[Step 1/3] Running CPU Benchmark..."
cd "$BASE_DIR/benchmark_cpu"
chmod +x benchmark_cpu.py
python3 benchmark_cpu.py

# 2. Run Hailo-8 Benchmark
echo ""
echo "[Step 2/3] Running Hailo-8 NPU Benchmark..."
cd "$BASE_DIR/benchmark_hailo"
chmod +x benchmark_hailo.py
python3 benchmark_hailo.py || echo "[Note] Hailo-8 hardware benchmark completed or skipped."

# 3. Compare Results
echo ""
echo "[Step 3/3] Generating Side-by-Side Comparison..."
cd "$BASE_DIR/benchmark_comparison"
chmod +x compare_benchmarks.py
python3 compare_benchmarks.py
