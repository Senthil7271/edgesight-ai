#!/usr/bin/env python3
"""
EdgeSight — Model Path Resolver Utility
Seamlessly locates models across category subdirectories in models/:
  - models/object_detection/
  - models/currency_detection/
  - models/ocr/
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SUBDIRS = [
    "object_detection",
    "currency_detection",
    "ocr",
    ""
]

ALIASES = {
    "yolov5s.hef": "indian_currency_best.hef",
    "indian_currency.hef": "indian_currency_best.hef",
    "indian_currency.onnx": "indian_currency_best.onnx",
    "indian_currency_best.pt": "indian_currency_best.onnx",
    "yolov5s.pt": "indian_currency_best.onnx",
    "ocr_det_infer.onnx": "ocr_det.onnx",
    "tiny_yolov4_license_plates.hef": "ocr_det.hef",
    "lprnet.hef": "ocr_det.hef"
}

def resolve_model_path(model_name):
    """
    Finds absolute path to a model file across category folders.
    Returns the resolved path if found, or the original path as fallback.
    """
    if not model_name:
        return model_name

    # 1. If already valid path
    if os.path.exists(model_name):
        return os.path.abspath(model_name)

    fname = os.path.basename(model_name)
    names_to_try = [fname]
    if fname in ALIASES:
        names_to_try.append(ALIASES[fname])

    # 2. Search in MODELS_DIR/<category>/<fname>
    for name in names_to_try:
        for sub in SUBDIRS:
            cand = os.path.join(MODELS_DIR, sub, name) if sub else os.path.join(MODELS_DIR, name)
            if os.path.exists(cand):
                return os.path.abspath(cand)

        # 3. Search relative to CWD
        for rel_sub in ["models/object_detection", "models/currency_detection", "models/ocr", "models"]:
            cand = os.path.join(rel_sub, name)
            if os.path.exists(cand):
                return os.path.abspath(cand)

    # 4. Fallback to system model zoo on Raspberry Pi
    system_paths = [
        os.path.join("/usr/share/hailo-models", fname),
        os.path.join("/usr/share/rpicam-apps/assets", fname)
    ]
    for sys_p in system_paths:
        if os.path.exists(sys_p):
            return sys_p

    return model_name
