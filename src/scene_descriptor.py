#!/usr/bin/env python3
"""
Frame-Based Scene Description Engine
Generates detailed narrative frame captions and spatial relationship summaries
for static frame snapshots.
"""

import math

def get_box_geometry(box):
    x, y, w, h = box
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    cx = x + w / 2.0
    cy = y + h / 2.0
    area = max(1.0, w * h)
    return x1, y1, x2, y2, cx, cy, w, h, area

def compute_intersection_over_b(boxA, boxB):
    ax1, ay1, ax2, ay2, _, _, _, _, areaA = get_box_geometry(boxA)
    bx1, by1, bx2, by2, _, _, _, _, areaB = get_box_geometry(boxB)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return inter_area / areaB

def compute_iou(boxA, boxB):
    ax1, ay1, ax2, ay2, _, _, _, _, areaA = get_box_geometry(boxA)
    bx1, by1, bx2, by2, _, _, _, _, areaB = get_box_geometry(boxB)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area

def compute_distance(boxA, boxB):
    _, _, _, _, cxA, cyA, _, _, _ = get_box_geometry(boxA)
    _, _, _, _, cxB, cyB, _, _, _ = get_box_geometry(boxB)
    return math.hypot(cxA - cxB, cyA - cyB)


class FrameSceneDescriptor:
    def describe_frame(self, detections, img_shape=(480, 640)):
        """
        Analyzes a single frame's detections and builds a rich natural language summary sentence
        and structured action relationships.
        """
        if not detections:
            return {
                "summary": "No objects detected in the current frame.",
                "relationships": [],
                "counts": {},
                "object_count": 0
            }

        people = [d for d in detections if d['class_name'] == 'person']
        couches = [d for d in detections if d['class_name'] in ['couch', 'sofa', 'bed']]
        chairs = [d for d in detections if d['class_name'] in ['chair', 'bench']]
        mobiles = [d for d in detections if d['class_name'] in ['cell phone', 'mobile phone', 'phone']]
        laptops = [d for d in detections if d['class_name'] in ['laptop', 'keyboard']]
        cups = [d for d in detections if d['class_name'] in ['cup', 'bottle', 'wine glass']]
        tables = [d for d in detections if d['class_name'] in ['dining table', 'desk', 'table']]
        pets = [d for d in detections if d['class_name'] in ['dog', 'cat']]
        bags = [d for d in detections if d['class_name'] in ['backpack', 'handbag', 'suitcase']]

        relationships = []
        sentence_parts = []

        # 1. Check Person + Mobile Phone
        person_has_phone = False
        for p in people:
            for m in mobiles:
                overlap = compute_intersection_over_b(p['box'], m['box'])
                dist = compute_distance(p['box'], m['box'])
                _, _, _, _, _, _, pw, ph, _ = get_box_geometry(p['box'])

                if overlap > 0.15 or dist < (pw + ph) * 0.5:
                    relationships.append("Person holding a mobile phone")
                    sentence_parts.append("a person holding a mobile phone")
                    person_has_phone = True
                    break

        # 2. Check Person + Couch / Sleeping / Sitting
        person_on_couch = False
        for p in people:
            for c in couches:
                overlap = compute_intersection_over_b(c['box'], p['box'])
                iou = compute_iou(p['box'], c['box'])
                if overlap > 0.25 or iou > 0.15:
                    _, _, _, _, _, _, pw, ph, _ = get_box_geometry(p['box'])
                    if pw / max(1.0, ph) > 0.9:
                        relationships.append(f"Person sleeping or lying on {c['class_name']}")
                        sentence_parts.append(f"a person sleeping on the {c['class_name']}")
                    else:
                        relationships.append(f"Person sitting on {c['class_name']}")
                        sentence_parts.append(f"a person sitting on the {c['class_name']}")
                    person_on_couch = True

        # 3. Check Person + Chair
        if not person_on_couch:
            for p in people:
                for ch in chairs:
                    overlap = compute_intersection_over_b(ch['box'], p['box'])
                    if overlap > 0.2 or compute_iou(p['box'], ch['box']) > 0.1:
                        relationships.append(f"Person sitting on {ch['class_name']}")
                        sentence_parts.append(f"a person sitting on a {ch['class_name']}")
                        break

        # 4. Check Person + Laptop
        for p in people:
            for l in laptops:
                overlap = compute_intersection_over_b(p['box'], l['box'])
                dist = compute_distance(p['box'], l['box'])
                _, _, _, _, _, _, pw, ph, _ = get_box_geometry(p['box'])
                if overlap > 0.15 or dist < (pw + ph) * 0.5:
                    relationships.append("Person using a laptop")
                    sentence_parts.append("a person working on a laptop")
                    break

        # 5. Check Person + Cup / Bottle
        for p in people:
            for cup in cups:
                overlap = compute_intersection_over_b(p['box'], cup['box'])
                dist = compute_distance(p['box'], cup['box'])
                _, _, _, _, _, _, pw, ph, _ = get_box_geometry(p['box'])
                if overlap > 0.2 or dist < (pw + ph) * 0.4:
                    relationships.append(f"Person holding a {cup['class_name']}")
                    sentence_parts.append(f"holding a {cup['class_name']}")

        # 6. Check Pets
        for p in people:
            for pet in pets:
                dist = compute_distance(p['box'], pet['box'])
                _, _, _, _, _, _, pw, ph, _ = get_box_geometry(p['box'])
                if dist < (pw + ph) * 0.6:
                    relationships.append(f"Person with a {pet['class_name']}")
                    sentence_parts.append(f"with a {pet['class_name']} nearby")

        # 7. Check Objects on Table
        for t in tables:
            for obj in cups + laptops + mobiles:
                overlap = compute_intersection_over_b(t['box'], obj['box'])
                if overlap > 0.35:
                    relationships.append(f"{obj['class_name'].capitalize()} placed on {t['class_name']}")

        # Build Object Frequency Summary
        counts = {}
        for d in detections:
            name = d['class_name']
            counts[name] = counts.get(name, 0) + 1

        counts_str = ", ".join([f"{c} {n}{'s' if c > 1 else ''}" for n, c in counts.items()])

        # Compose full paragraph description
        if sentence_parts:
            unique_parts = list(dict.fromkeys(sentence_parts))
            narrative = "The scene shows " + " and ".join(unique_parts) + "."
        elif len(people) > 0:
            narrative = f"The scene depicts {len(people)} person{'s' if len(people) > 1 else ''} present along with {counts_str}."
        else:
            narrative = f"This frame contains {counts_str}."

        return {
            "summary": narrative,
            "relationships": list(dict.fromkeys(relationships)),
            "counts": counts,
            "object_count": len(detections)
        }

    def generate_scene_description(self, detections):
        analysis = self.describe_frame(detections)
        if analysis['relationships']:
            return analysis['relationships']
        elif analysis['summary']:
            return [analysis['summary']]
        return []


SceneDescriptor = FrameSceneDescriptor

