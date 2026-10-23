#!/usr/bin/env python3
"""
EdgeSight — Hailo-8 NPU Benchmark Utility (Raspberry Pi 5 + 26 TOPS AI HAT+)
Measures pure NPU inference latency (ms), hardware FPS, and throughput via PyHailoRT / hailortcli.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ANSI Colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

HAILO_HEF_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolov8n.hef"

def find_hef_file():
    candidates = [
        "models/object_detection/yolov8n.hef",
        "../models/object_detection/yolov8n.hef",
        os.path.join(os.path.dirname(__file__), "..", "models", "object_detection", "yolov8n.hef"),
        "yolov8n.hef",
        "models/yolov8n.hef",
        "../yolov8n.hef",
        "../models/yolov8n.hef",
        os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.hef"),
        "/usr/share/hailo-models/yolov8n.hef",
        "/usr/share/rpicam-apps/assets/yolov8n.hef"
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def download_hef_if_needed(target_path="yolov8n.hef"):
    print(f"📥 'yolov8n.hef' not found locally. Downloading official compiled Hailo-8 model from Model Zoo...")
    try:
        import urllib.request
        urllib.request.urlretrieve(HAILO_HEF_URL, target_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
            print(f"✅ Successfully downloaded 'yolov8n.hef' ({os.path.getsize(target_path)/(1024*1024):.1f} MB)!")
            return os.path.abspath(target_path)
    except Exception as e:
        print(f"⚠️ Could not download HEF directly: {e}")
    return None

def run_hailortcli_benchmark(hef_path):
    print(f"🔍 Running official HailoRT CLI hardware benchmark: 'hailortcli run {hef_path}'...")
    try:
        cmd = f"hailortcli run '{hef_path}' --time 3"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            output = res.stdout
            print(output)
            # Parse FPS from hailortcli output if present
            for line in output.split("\n"):
                if "FPS" in line or "fps" in line:
                    print(f"   {GREEN}✔ {line.strip()}{RESET}")
    except Exception as e:
        print(f"Note on hailortcli benchmark: {e}")

def main():
    print(f"\n{BOLD}{CYAN}=============================================================================={RESET}")
    print(f"{BOLD}{CYAN}      🚀 EDGESIGHT — HAILO-8 NPU BENCHMARK RUNNER (26 AI TOPS PCIe HAT+)      {RESET}")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")

    # 1. Check if Hailo PCIe hardware is connected
    try:
        lspci_out = subprocess.check_output("lspci", shell=True, text=True)
        if "Hailo" not in lspci_out:
            print(f"⚠️ Warning: Hailo PCIe device not found in 'lspci'. Ensure the AI HAT+ is firmly clamped.")
    except Exception:
        pass

    # 2. Locate or download HEF model
    hef_path = find_hef_file()
    if not hef_path:
        local_hef = os.path.join(os.path.dirname(__file__), "yolov8n.hef")
        hef_path = download_hef_if_needed(local_hef)

    if not hef_path or not os.path.exists(hef_path):
        print("❌ Error: Could not locate or download 'yolov8n.hef'.")
        print("   On Raspberry Pi 5, ensure 'sudo apt install hailo-all' is completed.")
        sys.exit(1)

    print(f"📦 Hailo HEF Model: {hef_path} ({os.path.getsize(hef_path)/(1024*1024):.1f} MB)")

    # 3. Import PyHailoRT
    try:
        from hailo_platform import (
            HEF,
            VDevice,
            HailoStreamInterface,
            InferVStreams,
            ConfigureParams,
            InputVStreamParams,
            OutputVStreamParams,
            FormatType
        )
    except ImportError:
        print("❌ Error: PyHailoRT ('hailo_platform') not installed.")
        print("   Run on Raspberry Pi: sudo apt install -y hailo-all python3-hailort")
        sys.exit(1)

    print(f"⚡ Initializing Hailo-8 VDevice on PCIe Gen 3 Interface...")
    t_init0 = time.perf_counter()
    target = VDevice()
    hef = HEF(hef_path)
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_infos = hef.get_output_vstream_infos()
    input_shape = input_vstream_info.shape
    h, w, c = input_shape
    t_init = (time.perf_counter() - t_init0) * 1000.0

    print(f"✅ Hailo-8 NPU Initialized in {t_init:.1f} ms!")
    print(f"📐 NPU Tensor Input Shape: {w}x{h}x{c} (UINT8)")

    # 4. Prepare input stream buffers
    raw_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    import cv2
    resized = cv2.resize(raw_frame, (w, h))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_img, axis=0).astype(np.uint8)

    input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    # 5. Warmup
    print("🔥 Warming up Hailo-8 NPU pipeline (15 iterations)...")
    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        input_dict = {input_vstream_info.name: input_data}
        with network_group.activate(network_group_params):
            for _ in range(15):
                _ = infer_pipeline.infer(input_dict)

    # 6. Benchmark Loop
    NUM_FRAMES = 100
    print(f"🚀 Running Hailo-8 26 TOPS NPU Benchmark over {NUM_FRAMES} frames...")

    preprocess_times = []
    inference_times = []
    e2e_times = []

    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        input_dict = {input_vstream_info.name: input_data}
        with network_group.activate(network_group_params):
            for i in range(NUM_FRAMES):
                t0 = time.perf_counter()

                # Preprocessing
                resized_frame = cv2.resize(raw_frame, (w, h))
                rgb_f = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frame_data = np.expand_dims(rgb_f, axis=0).astype(np.uint8)
                t1 = time.perf_counter()

                # Pure NPU Inference over PCIe
                results = infer_pipeline.infer({input_vstream_info.name: frame_data})
                t2 = time.perf_counter()

                # Postprocessing
                _ = results
                t3 = time.perf_counter()

                p_time = (t1 - t0) * 1000.0
                i_time = (t2 - t1) * 1000.0
                total_time = (t3 - t0) * 1000.0

                preprocess_times.append(p_time)
                inference_times.append(i_time)
                e2e_times.append(total_time)

                if (i + 1) % 25 == 0 or (i + 1) == NUM_FRAMES:
                    print(f"   Progress: {i + 1}/{NUM_FRAMES} frames | NPU Latency: {i_time:.2f} ms")

    # Metrics
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
        "device": "Hailo-8 NPU (26 TOPS AI HAT+)",
        "model": "YOLOv8 Nano (Hailo HEF)",
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
    out_json = os.path.join(os.path.dirname(__file__), "hailo_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    # Print Summary Box
    print(f"\n{BOLD}{GREEN}=============================================================================={RESET}")
    print(f"{BOLD}{GREEN}               📊 HAILO-8 NPU BENCHMARK RESULTS (26 TOPS)                    {RESET}")
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
    print(f"  • {BOLD}Pure NPU Inference FPS:{RESET}    {YELLOW}{infer_fps:.2f} FPS{RESET}")
    print(f"  • {BOLD}End-to-End System FPS:{RESET}     {YELLOW}{e2e_fps:.2f} FPS{RESET}")
    print(f"  --------------------------------------------------------------------------")
    print(f"  💾 Results saved to: {out_json}")
    print(f"{BOLD}{GREEN}=============================================================================={RESET}\n")

if __name__ == "__main__":
    main()
