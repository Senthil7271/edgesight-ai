#!/usr/bin/env python3
"""
QR Code & Barcode Scanner Engine
Uses OpenCV built-in QRCodeDetector and BarcodeDetector (with PyZbar fallback if installed)
"""

import cv2
import numpy as np

class QRBarcodeScanner:
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
        self.barcode_detector = None
        try:
            if hasattr(cv2, 'barcode') and hasattr(cv2.barcode, 'BarcodeDetector'):
                self.barcode_detector = cv2.barcode.BarcodeDetector()
        except Exception as e:
            print(f"[SCANNER] BarcodeDetector init note: {e}")

        # Check PyZbar fallback
        self.has_pyzbar = False
        try:
            from pyzbar import pyzbar
            self.pyzbar = pyzbar
            self.has_pyzbar = True
            print("[SCANNER] PyZbar library available!")
        except Exception:
            pass

    def scan(self, image):
        results = []
        h, w = image.shape[:2]

        # 1. Scan via PyZbar if available
        if self.has_pyzbar:
            try:
                barcodes = self.pyzbar.decode(image)
                for code in barcodes:
                    data_str = code.data.decode('utf-8', errors='ignore').strip()
                    code_type = code.type
                    rect = code.rect
                    polygon = code.polygon
                    
                    pts = np.array([[p.x, p.y] for p in polygon], dtype=np.int32) if polygon else None
                    box = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]

                    results.append({
                        'type': code_type,
                        'data': data_str,
                        'polygon': pts,
                        'box': box,
                        'source': 'pyzbar'
                    })
            except Exception as e:
                print(f"[SCANNER] PyZbar error: {e}")

        # 2. Scan QR Codes via OpenCV QRCodeDetector
        if not results:
            try:
                res = self.qr_detector.detectAndDecodeMulti(image)
                retval = res[0]
                decoded_info = res[1]
                points = res[2] if len(res) > 2 else None

                if retval and decoded_info:
                    for i, data in enumerate(decoded_info):
                        data_str = data.strip()
                        if not data_str:
                            continue
                        
                        pts = None
                        box = None
                        if points is not None and i < len(points):
                            pts = points[i].astype(np.int32)
                            x_min = int(np.min(pts[:, 0]))
                            y_min = int(np.min(pts[:, 1]))
                            x_max = int(np.max(pts[:, 0]))
                            y_max = int(np.max(pts[:, 1]))
                            box = [x_min, y_min, x_max, y_max]

                        results.append({
                            'type': 'QRCODE',
                            'data': data_str,
                            'polygon': pts,
                            'box': box,
                            'source': 'opencv_qr'
                        })
            except Exception as e:
                print(f"[SCANNER] OpenCV QR error: {e}")

        # 3. Scan Barcodes via OpenCV BarcodeDetector
        if not results and self.barcode_detector:
            try:
                res = self.barcode_detector.detectAndDecode(image)
                if len(res) == 4:
                    retval, decoded_info, decoded_type, points = res
                elif len(res) == 3:
                    retval, decoded_info, points = res
                    decoded_type = ['BARCODE'] * len(decoded_info) if decoded_info else []
                else:
                    retval = False

                if retval and decoded_info:
                    for i, data in enumerate(decoded_info):
                        data_str = data.strip()
                        if not data_str:
                            continue

                        b_type = decoded_type[i] if (decoded_type and i < len(decoded_type)) else 'BARCODE'
                        pts = None
                        box = None
                        if points is not None and i < len(points):
                            pts = points[i].astype(np.int32)
                            x_min = int(np.min(pts[:, 0]))
                            y_min = int(np.min(pts[:, 1]))
                            x_max = int(np.max(pts[:, 0]))
                            y_max = int(np.max(pts[:, 1]))
                            box = [x_min, y_min, x_max, y_max]

                        results.append({
                            'type': str(b_type),
                            'data': data_str,
                            'polygon': pts,
                            'box': box,
                            'source': 'opencv_barcode'
                        })
            except Exception as e:
                print(f"[SCANNER] OpenCV Barcode error: {e}")

        return results

    def summarize_scan(self, results):
        if not results:
            return {
                'count': 0,
                'summary': "No QR code or barcode detected in the frame. Point camera closer to the code.",
                'details': []
            }

        details = []
        summaries = []

        for item in results:
            code_type = item['type'].upper()
            data = item['data']
            
            is_url = data.startswith('http://') or data.startswith('https://') or data.startswith('www.')
            
            if is_url:
                summary_text = f"Scanned {code_type} URL: {data}"
            elif code_type in ['QRCODE', 'QR']:
                summary_text = f"Scanned QR Code content: {data}"
            else:
                summary_text = f"Scanned {code_type} Barcode: {data}"

            summaries.append(summary_text)
            details.append({
                'type': code_type,
                'data': data,
                'is_url': is_url
            })

        main_summary = ". ".join(summaries)
        return {
            'count': len(results),
            'summary': main_summary,
            'details': details
        }

def draw_scan_detections(image, results, fps=None, inference_time_ms=None):
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for item in results:
        code_type = item['type'].upper()
        data = item['data']
        pts = item.get('polygon')
        box = item.get('box')

        color = (0, 255, 170) if 'QR' in code_type else (0, 210, 255)

        # Polygon box
        if pts is not None and len(pts) >= 4:
            cv2.polylines(annotated, [pts], True, color, 3)
            x1, y1 = pts[0][0], pts[0][1]
        elif box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        else:
            x1, y1 = 50, 50

        display_text = f"{code_type}: {data[:30]}"
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, display_text, (x1 + 5, max(th + 2, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # Top header banner
    banner_h = 45
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (11, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    count_text = f"SCANNER MODE | CODES FOUND: {len(results)}"
    cv2.putText(annotated, count_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 170), 2)

    if fps and inference_time_ms:
        stats = f"{fps:.1f} FPS | {inference_time_ms:.1f} ms"
        (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(annotated, stats, (w - tw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 43, 138), 1)

    return annotated

if __name__ == "__main__":
    scanner = QRBarcodeScanner()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = scanner.scan(dummy)
    print("Scan test output:", res)
    print("Summary:", scanner.summarize_scan(res))
