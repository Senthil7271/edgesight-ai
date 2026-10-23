#!/usr/bin/env python3
"""
EdgeSight OCR Reader Engine
Tesseract-based text extraction for books, signs, labels, and documents.
Optimized for Raspberry Pi with adaptive preprocessing pipeline.

Features:
  - Multi-strategy preprocessing (grayscale, CLAHE, denoise, threshold, deskew)
  - Configurable PSM modes for different text layouts
  - Bounding box extraction for annotated overlays
  - Voice-friendly text summarization
  - Sign/label detection with short-text optimizations
"""

import cv2
import numpy as np
import re
import math

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# Tesseract Page Segmentation Modes (PSM) — selected per use case
PSM_AUTO = 3           # Fully automatic page segmentation (default)
PSM_SINGLE_BLOCK = 6   # Assume a single uniform block of text
PSM_SINGLE_LINE = 7    # Treat the image as a single text line
PSM_SPARSE_TEXT = 11   # Sparse text — find as much text as possible in no order
PSM_RAW_LINE = 13      # Raw line — treat as single text line, no OSD


class OCRReader:
    """
    EdgeSight OCR engine supporting RapidOCR (Deep Learning ONNX) with
    Tesseract fallback. Designed for books, signs, labels, and real-world camera captures.
    """

    def __init__(self, lang="eng", min_confidence=30):
        """
        Args:
            lang: Language code(s), e.g. 'eng'
            min_confidence: Minimum word confidence (0-100) to include in results
        """
        self.lang = lang
        self.min_confidence = min_confidence
        self.engine_type = "none"
        self.rapid_engine = None

        # 1. Initialize RapidOCR (Deep Learning ONNX, zero external C++ dependencies)
        if RAPIDOCR_AVAILABLE:
            try:
                self.rapid_engine = RapidOCR()
                self.engine_type = "rapidocr"
                print("[OCR] RapidOCR ONNX Deep-Learning Engine initialized successfully!")
            except Exception as e:
                print(f"[OCR] RapidOCR initialization warning: {e}")

        # 2. Check Tesseract fallback
        if self.engine_type == "none" and TESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                self.engine_type = "tesseract"
                print(f"[OCR] Tesseract OCR initialized (v{version}, lang='{lang}')")
            except Exception as e:
                print(f"[OCR] Warning: Tesseract binary check failed: {e}")

        if self.engine_type == "none":
            print("[OCR] Note: Running with lightweight fallback parser.")

    # ─── Preprocessing Pipeline ──────────────────────────────────────────

    def _to_grayscale(self, image):
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _enhance_contrast(self, gray):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _denoise(self, gray):
        """Bilateral filter preserves edges while reducing noise."""
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def _adaptive_threshold(self, gray):
        """Adaptive Gaussian threshold for varying lighting conditions."""
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=11
        )

    def _otsu_threshold(self, gray):
        """Otsu's binarization — good for bimodal histograms (printed text)."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _deskew(self, binary):
        """Correct slight rotation of text in camera captures."""
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) < 50:
            return binary

        try:
            # minAreaRect gives the rotation angle of the text block
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]

            # Normalize angle
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = -(angle - 90)
            else:
                angle = -angle

            # Only deskew if rotation is small (< 15 degrees) to avoid false corrections
            if abs(angle) > 15 or abs(angle) < 0.5:
                return binary

            h, w = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        except Exception:
            return binary

    def _add_border(self, image, border_px=15):
        """Add white border padding for Tesseract accuracy."""
        return cv2.copyMakeBorder(
            image, border_px, border_px, border_px, border_px,
            cv2.BORDER_CONSTANT, value=255
        )

    def preprocess(self, image, strategy="auto"):
        """
        Run the full preprocessing pipeline.

        Strategies:
            'auto'      — CLAHE + denoise + adaptive threshold + deskew (default)
            'simple'    — Grayscale + Otsu threshold only (fast)
            'sign'      — Optimized for short text on signs/labels
            'document'  — Optimized for printed documents/books
        """
        gray = self._to_grayscale(image)

        if strategy == "simple":
            binary = self._otsu_threshold(gray)
            return self._add_border(binary)

        if strategy == "sign":
            # Signs: high contrast, large text, short content
            enhanced = self._enhance_contrast(gray)
            # Upscale small text for better recognition
            h, w = enhanced.shape[:2]
            if max(h, w) < 400:
                scale = 400.0 / max(h, w)
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
            binary = self._otsu_threshold(enhanced)
            return self._add_border(binary)

        if strategy == "document":
            # Documents: clean text, consistent lighting
            denoised = self._denoise(gray)
            enhanced = self._enhance_contrast(denoised)
            binary = self._adaptive_threshold(enhanced)
            binary = self._deskew(binary)
            return self._add_border(binary)

        # 'auto' strategy — full pipeline
        denoised = self._denoise(gray)
        enhanced = self._enhance_contrast(denoised)
        binary = self._adaptive_threshold(enhanced)
        binary = self._deskew(binary)
        return self._add_border(binary)

    # ─── OCR Extraction ──────────────────────────────────────────────────

    def extract_text(self, image, strategy="auto", psm=PSM_AUTO):
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image: BGR or grayscale numpy array (camera frame)
            strategy: Preprocessing strategy ('auto', 'simple', 'sign', 'document')
            psm: Tesseract Page Segmentation Mode

        Returns:
            dict with keys:
                'text'       — Full extracted text string
                'words'      — List of word-level detections with confidence and bbox
                'line_count' — Number of text lines detected
                'word_count' — Number of words above confidence threshold
                'avg_confidence' — Average word confidence (0-100)
                'preprocessed_shape' — Shape of the preprocessed image
        """
        # 1. Try RapidOCR (Deep Learning ONNX) first for high-accuracy scene text
        if self.rapid_engine is not None:
            rapid_res = self._extract_with_rapidocr(image)
            if rapid_res and rapid_res.get('word_count', 0) > 0:
                return rapid_res

        # 2. Fallback to Tesseract if available
        if not TESSERACT_AVAILABLE:
            return self._empty_result("No text detected in the image.")

        processed = self.preprocess(image, strategy=strategy)

        # Build Tesseract config string
        config = f"--oem 3 --psm {psm}"

        # Extract full text
        try:
            raw_text = pytesseract.image_to_string(
                processed, lang=self.lang, config=config
            ).strip()
        except Exception as e:
            return self._empty_result(f"OCR extraction failed: {str(e)}")

        # Extract word-level data with bounding boxes and confidence
        words = []
        try:
            data = pytesseract.image_to_data(
                processed, lang=self.lang, config=config,
                output_type=pytesseract.Output.DICT
            )

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                word = data['text'][i].strip()
                conf = int(data['conf'][i])

                if word and conf >= self.min_confidence:
                    words.append({
                        'text': word,
                        'confidence': conf,
                        'box': [
                            int(data['left'][i]),
                            int(data['top'][i]),
                            int(data['width'][i]),
                            int(data['height'][i])
                        ],
                        'line_num': int(data['line_num'][i]),
                        'block_num': int(data['block_num'][i])
                    })
        except Exception:
            pass

        # Clean the extracted text
        cleaned = self._clean_text(raw_text)

        # Calculate statistics
        confidences = [w['confidence'] for w in words]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        # Count unique lines
        line_nums = set(w['line_num'] for w in words)

        return {
            'text': cleaned,
            'raw_text': raw_text,
            'words': words,
            'line_count': len(line_nums),
            'word_count': len(words),
            'avg_confidence': round(avg_conf, 1),
            'preprocessed_shape': processed.shape[:2],
            'strategy': strategy
        }

    def _extract_with_rapidocr(self, image):
        """Extract text using RapidOCR ONNX deep-learning model."""
        try:
            result, _ = self.rapid_engine(image)
            if not result:
                return None

            words = []
            raw_lines = []
            for line_idx, item in enumerate(result, 1):
                box_pts, txt, score = item
                txt_clean = txt.strip()
                conf_pct = int(score * 100)
                if not txt_clean or conf_pct < self.min_confidence:
                    continue

                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bw = max(1, x_max - x_min)
                bh = max(1, y_max - y_min)

                words.append({
                    'text': txt_clean,
                    'confidence': conf_pct,
                    'box': [int(x_min), int(y_min), int(bw), int(bh)],
                    'line_num': line_idx,
                    'block_num': 1
                })
                raw_lines.append(txt_clean)

            if not words:
                return None

            full_raw = '\n'.join(raw_lines)
            cleaned = self._clean_text(full_raw)
            confidences = [w['confidence'] for w in words]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return {
                'text': cleaned if cleaned else full_raw,
                'raw_text': full_raw,
                'words': words,
                'line_count': len(raw_lines),
                'word_count': len(words),
                'avg_confidence': round(avg_conf, 1),
                'preprocessed_shape': image.shape[:2],
                'strategy': 'rapidocr'
            }
        except Exception as e:
            print(f"[OCR] RapidOCR extraction error: {e}")
            return None

    def extract_text_multi_strategy(self, image):
        """
        Try multiple preprocessing strategies and return the best result.
        Useful when the input type (sign vs document) is unknown.
        """
        best_result = None
        best_word_count = 0

        for strategy in ['auto', 'simple', 'sign']:
            result = self.extract_text(image, strategy=strategy)
            if result['word_count'] > best_word_count:
                best_word_count = result['word_count']
                best_result = result

        if best_result is None or best_word_count == 0:
            # Last resort: try with sparse text PSM
            best_result = self.extract_text(image, strategy='auto', psm=PSM_SPARSE_TEXT)

        return best_result

    # ─── Text Cleaning ───────────────────────────────────────────────────

    def _clean_text(self, text):
        """Clean OCR artifacts and normalize whitespace."""
        if not text:
            return ""

        # Remove common OCR noise characters
        text = re.sub(r'[|~`^\\]', '', text)

        # Collapse multiple spaces/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove lines that are only punctuation or single characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 1 or stripped.isalnum():
                cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines).strip()

    # ─── Summarization for Voice Output ──────────────────────────────────

    def summarize_for_voice(self, ocr_result):
        """
        Create a voice-friendly summary of OCR results.

        Returns:
            dict with:
                'speech_text'  — Text to be spoken aloud via TTS
                'summary'      — Short summary string
                'is_sign'      — Whether this looks like a short sign/label
                'has_text'     — Whether any text was detected
        """
        text = ocr_result.get('text', '').strip()
        word_count = ocr_result.get('word_count', 0)
        avg_conf = ocr_result.get('avg_confidence', 0)

        if not text or word_count == 0:
            return {
                'speech_text': "I could not read any text. Please hold the camera steady and point it at the text.",
                'summary': "No text detected in the image.",
                'is_sign': False,
                'has_text': False
            }

        # Determine if this is a short sign/label vs a longer document
        is_sign = word_count <= 8 and ocr_result.get('line_count', 0) <= 3

        if is_sign:
            # For signs: read the text directly
            speech = f"This sign reads: {text}"
            summary = f"Sign detected: \"{text}\" ({word_count} words, {avg_conf:.0f}% confidence)"
        elif word_count <= 30:
            # Short text: read it all
            speech = f"The text reads: {text}"
            summary = f"Text extracted: {word_count} words ({avg_conf:.0f}% confidence)"
        else:
            # Longer document: read first portion, summarize rest
            sentences = re.split(r'(?<=[.!?])\s+', text)
            preview = ' '.join(sentences[:3])
            if len(preview) > 300:
                preview = preview[:297] + "..."

            speech = f"I found {word_count} words of text. Here is the beginning: {preview}"
            summary = f"Document text: {word_count} words across {ocr_result.get('line_count', 0)} lines ({avg_conf:.0f}% confidence)"

        # Add confidence warning if low
        if avg_conf < 50 and avg_conf > 0:
            speech += " Note: the text quality is low, some words may be inaccurate."

        return {
            'speech_text': speech,
            'summary': summary,
            'is_sign': is_sign,
            'has_text': True
        }

    # ─── Drawing / Annotation ────────────────────────────────────────────

    def _empty_result(self, error_msg=""):
        return {
            'text': '',
            'raw_text': '',
            'words': [],
            'line_count': 0,
            'word_count': 0,
            'avg_confidence': 0,
            'preprocessed_shape': (0, 0),
            'strategy': 'none',
            'error': error_msg
        }


def draw_ocr_detections(image, ocr_result, fps=None, inference_time_ms=None):
    """
    Draw OCR word bounding boxes and a telemetry banner on the image.

    Args:
        image: Original BGR image
        ocr_result: Dict from OCRReader.extract_text()
        fps: Optional FPS value for display
        inference_time_ms: Optional latency value

    Returns:
        Annotated BGR image
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    words = ocr_result.get('words', [])

    # Draw word bounding boxes
    for word_info in words:
        box = word_info['box']
        conf = word_info['confidence']
        text = word_info['text']

        x, y, bw, bh = box

        # Color based on confidence: green (high) → yellow (medium) → red (low)
        if conf >= 80:
            color = (0, 255, 170)   # Green
        elif conf >= 60:
            color = (0, 210, 255)   # Amber/Yellow
        else:
            color = (0, 100, 255)   # Orange-Red

        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        # Word label
        label = f"{text} ({conf}%)"
        font_scale = 0.4
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x, max(0, y - th - 4)), (x + tw + 4, y), color, -1)
        cv2.putText(annotated, label, (x + 2, max(th + 1, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Top banner overlay
    word_count = ocr_result.get('word_count', 0)
    avg_conf = ocr_result.get('avg_confidence', 0)
    strategy = ocr_result.get('strategy', 'auto')

    banner_h = 50
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (11, 14, 20), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    engine_tag = "RapidOCR ONNX" if strategy == "rapidocr" else "Tesseract"
    header = f"OCR MODE ({engine_tag}) | {word_count} Words | Confidence: {avg_conf:.0f}%"
    cv2.putText(annotated, header, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 170), 2, cv2.LINE_AA)

    if fps and inference_time_ms:
        stats = f"{fps:.1f} FPS | {inference_time_ms:.1f} ms"
        (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(annotated, stats, (w - tw - 15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 43, 138), 1, cv2.LINE_AA)

    # Bottom text preview bar
    extracted_text = ocr_result.get('text', '')
    if extracted_text:
        preview = extracted_text.replace('\n', ' ')
        if len(preview) > 80:
            preview = preview[:77] + "..."

        bar_h = 40
        bar_y = h - bar_h
        cv2.rectangle(overlay, (0, bar_y), (w, h), (11, 14, 20), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
        cv2.putText(annotated, f"TEXT: {preview}", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 255), 1, cv2.LINE_AA)

    return annotated


if __name__ == "__main__":
    print(f"Tesseract Available: {TESSERACT_AVAILABLE}")
    if TESSERACT_AVAILABLE:
        reader = OCRReader(lang="eng")

        # Test with a blank image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, "HELLO WORLD", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.putText(dummy, "EdgeSight OCR Test", (120, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        result = reader.extract_text(dummy, strategy='auto')
        print(f"Extracted text: '{result['text']}'")
        print(f"Words: {result['word_count']}, Confidence: {result['avg_confidence']}%")

        voice = reader.summarize_for_voice(result)
        print(f"Voice output: {voice['speech_text']}")
