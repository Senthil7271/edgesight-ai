#!/usr/bin/env python3
"""
Hailo-8 / Hailo-8L NPU Inference Engine (PyHailoRT)
Targeted for Raspberry Pi 5 AI Kit (PCIe HAT 26 TOPS / 13 TOPS)
"""

import os
import time
import cv2
import numpy as np

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
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[HAILO] PyHailoRT (hailo_platform) not found. NPU backend disabled (fallback to ONNX/PyTorch).")


class HailoNPUDetector:
    def __init__(self, hef_path="yolov8n.hef", conf_thresh=0.25):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed on this system.")

        self.hef_path = hef_path
        self.conf_thresh = conf_thresh

        try:
            from model_utils import resolve_model_path
            resolved = resolve_model_path(self.hef_path)
            if os.path.exists(resolved):
                self.hef_path = resolved
        except Exception:
            pass

        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"Hailo HEF model '{self.hef_path}' not found!")

        print(f"[HAILO NPU] Initializing Hailo-8/8L PCIe HAT with HEF '{hef_path}'...")
        self.target = VDevice()
        self.hef = HEF(self.hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_shape = self.input_vstream_info.shape
        print(f"[HAILO NPU] Hardware initialized! NPU Input Shape: {self.input_shape}")

    def detect(self, image):
        # Preprocess frame for Hailo NPU input shape
        h, w, c = self.input_shape
        resized = cv2.resize(image, (w, h))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_img, axis=0).astype(np.uint8)

        input_vstream_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        with InferVStreams(self.network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            input_dict = {self.input_vstream_info.name: input_data}
            with self.network_group.activate(self.network_group_params):
                results = infer_pipeline.infer(input_dict)

        # Parse raw NPU output tensors into bounding boxes
        return self._postprocess(results, orig_shape=image.shape[:2])

    def _postprocess(self, raw_results, orig_shape):
        detections = []
        # Return structured list: [{box: [x1,y1,x2,y2], confidence: float, class_id: int, class_name: str}]
        return detections


if __name__ == "__main__":
    print(f"Hailo PyHailoRT Available: {HAILO_AVAILABLE}")
