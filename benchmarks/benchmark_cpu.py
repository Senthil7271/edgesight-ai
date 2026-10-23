#!/usr/bin/env python3
"""
EdgeSight — CPU Benchmark Utility (Raspberry Pi 5 ARM Cortex-A76 / Host CPU)
Measures pure CPU inference latency (ms), End-to-End latency, and FPS for YOLOv8 ONNX.
"""

import os
import sys
import time
import json
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Color codes
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def find_model_path():
    candidates = [
        "models/object_detection/yolov8n.onnx",
        "../models/object_detection/yolov8n.onnx",
        os.path.join(os.path.dirname(__file__), "..", "models", "object_detection", "yolov8n.onnx"),
        "yolov8n.onnx",
        "models/yolov8n.onnx",
        "../yolov8n.onnx",
        "../models/yolov8n.onnx",
        os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.onnx")
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def main():
    print(f"\n{BOLD}{CYAN}=============================================================================={RESET}")
    print(f"{BOLD}{CYAN}        ⚡ EDGESIGHT — CPU BENCHMARK RUNNER (Raspberry Pi 5 CPU)              {RESET}")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")

    model_path = find_model_path()
    if not model_path:
        print(f"❌ Error: 'yolov8n.onnx' not found in benchmark_cpu or parent directory.")
        sys.exit(1)

    print(f"📦 Model File: {model_path} ({os.path.getsize(model_path) / (1024*1024):.1f} MB)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ Error: onnxruntime not installed. Run: pip3 install onnxruntime --break-system-packages")
        sys.exit(1)

    # Initialize ONNX Session with optimized CPU threads
    num_threads = max(1, os.cpu_count() - 1)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"⚙️  Initializing ONNX Runtime Session (CPUExecutionProvider, Threads: {num_threads})...")
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    
    # Input shape: usually [1, 3, 320, 320] or [1, 3, 640, 640]
    in_shape = input_meta.shape
    h = in_shape[2] if len(in_shape) == 4 and isinstance(in_shape[2], int) else 320
    w = in_shape[3] if len(in_shape) == 4 and isinstance(in_shape[3], int) else 320
    print(f"📐 Input Resolution: {w}x{h} (Batch Size: 1)")

    # Prepare dummy input frame (representative 720p image)
    raw_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    print("🔥 Warming up CPU cache & clock frequencies (15 iterations)...")
    dummy_blob = np.random.randn(1, 3, h, w).astype(np.float32)
    for _ in range(15):
        session.run([output_name], {input_name: dummy_blob})

    # Benchmark Execution
    NUM_FRAMES = 100
    print(f"🚀 Running CPU Benchmark over {NUM_FRAMES} frames...")

    preprocess_times = []
    inference_times = []
    e2e_times = []

    for i in range(NUM_FRAMES):
        t0 = time.perf_counter()

        # 1. Preprocessing (Resize, normalize, transpose)
        import cv2
        resized = cv2.resize(raw_frame, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        t1 = time.perf_counter()

        # 2. Pure Inference
        outputs = session.run([output_name], {input_name: blob})
        t2 = time.perf_counter()

        # 3. Post-process (Parsing box predictions)
        preds = np.squeeze(outputs[0])
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        _ = preds[preds[:, 4] > 0.25] if preds.shape[0] > 0 else []
        t3 = time.perf_counter()

        p_time = (t1 - t0) * 1000.0
        i_time = (t2 - t1) * 1000.0
        total_time = (t3 - t0) * 1000.0

        preprocess_times.append(p_time)
        inference_times.append(i_time)
        e2e_times.append(total_time)

        if (i + 1) % 25 == 0 or (i + 1) == NUM_FRAMES:
            print(f"   Progress: {i + 1}/{NUM_FRAMES} frames | Current Latency: {i_time:.1f} ms")

    # Metrics Calculation
    avg_infer_ms = float(np.mean(inference_times))
    median_infer_ms = float(np.median(inference_times))
    p95_infer_ms = float(np.percentile(inference_times, 95))
    min_infer_ms = float(np.min(inference_times))
    max_infer_ms = float(np.max(inference_times))

    avg_prep_ms = float(np.mean(preprocess_times))
    avg_e2e_ms = float(np.mean(e2e_times))

    infer_fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0
    e2e_fps = 1000.0 / avg_e2e_ms if avg_e2e_ms > 0 else 0

    results = {
        "device": "Raspberry Pi 5 CPU (ARM Cortex-A76)",
        "model": "YOLOv8 Nano (ONNX Runtime)",
        "input_resolution": f"{w}x{h}",
        "frames_tested": NUM_FRAMES,
        "avg_inference_latency_ms": round(avg_infer_ms, 2),
        "median_inference_latency_ms": round(median_infer_ms, 2),
        "p95_inference_latency_ms": round(p95_infer_ms, 2),
        "min_inference_latency_ms": round(min_infer_ms, 2),
        "max_inference_latency_ms": round(max_infer_ms, 2),
        "avg_preprocess_ms": round(avg_prep_ms, 2),
        "avg_e2e_latency_ms": round(avg_e2e_ms, 2),
        "inference_fps": round(infer_fps, 2),
        "e2e_fps": round(e2e_fps, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON
    out_json = os.path.join(os.path.dirname(__file__), "cpu_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    # Print Summary Box
    print(f"\n{BOLD}{GREEN}=============================================================================={RESET}")
    print(f"{BOLD}{GREEN}                  📊 CPU BENCHMARK RESULTS (Raspberry Pi 5)                   {RESET}")
    print(f"{BOLD}{GREEN}=============================================================================={RESET}")
    print(f"  • Device Tested:            {BOLD}{results['device']}{RESET}")
    print(f"  • Model & Resolution:       {results['model']} ({results['input_resolution']})")
    print(f"  • Frames Evaluated:         {results['frames_tested']} frames")
    print(f"  --------------------------------------------------------------------------")
    print(f"  • {BOLD}Average Inference Latency:{RESET} {CYAN}{avg_infer_ms:.2f} ms{RESET} (Median: {median_infer_ms:.2f} ms)")
    print(f"  • P95 Latency:              {p95_infer_ms:.2f} ms (Min: {min_infer_ms:.1f}ms, Max: {max_infer_ms:.1f}ms)")
    print(f"  • Preprocessing Latency:    {avg_prep_ms:.2f} ms")
    print(f"  • End-to-End Latency:       {avg_e2e_ms:.2f} ms")
    print(f"  --------------------------------------------------------------------------")
    print(f"  • {BOLD}Pure Inference FPS:{RESET}        {YELLOW}{infer_fps:.2f} FPS{RESET}")
    print(f"  • {BOLD}End-to-End System FPS:{RESET}     {YELLOW}{e2e_fps:.2f} FPS{RESET}")
    print(f"  --------------------------------------------------------------------------")
    print(f"  💾 Results saved to: {out_json}")
    print(f"{BOLD}{GREEN}=============================================================================={RESET}\n")

if __name__ == "__main__":
    main()
