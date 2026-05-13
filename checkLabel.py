import json
import os
import math
from collections import defaultdict

json_path = "Original_Labelled_Data/result.json"
labels_dir = "Original_Labelled_Data/Absolutelabels"
os.makedirs(labels_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

# Build image lookup
images = {img["id"]: img for img in data.get("images", [])}

# Group annotations by image_id
annotations_by_image_id = defaultdict(list)
for ann in data.get("annotations", []):
    annotations_by_image_id[ann["image_id"]].append(ann)

def point_in_bbox(px, py, bbox):
    x, y, w, h = bbox
    return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)

def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w/2.0, y + h/2.0)

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Process each image
for image_id, img_info in images.items():
    img_w, img_h = img_info.get("width"), img_info.get("height")
    raw_name = img_info.get("file_name", "")
    img_name = os.path.basename(raw_name.replace("\\", "/"))
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_name)

    print(f"[INFO] Processing image: {img_name} (id={image_id})")

    current_image_annotations = annotations_by_image_id.get(image_id, [])

    # Collect bbox annotations (category_id == 0) as primary objects
    object_bundles = []
    for ann in current_image_annotations:
        if ann.get("category_id") == 0 and "bbox" in ann:
            object_bundles.append({"bbox_ann": ann, "keypoint_ann": None})

    # Attach keypoints robustly and log whether id-heuristic succeeded
    for kp_ann in current_image_annotations:
        if kp_ann.get("category_id") != 1:
            continue

        matched = False
        # try id-based heuristic first (if ids follow that pattern)
        matching_bbox_ann_id = kp_ann.get("id", -999) - 1
        for bundle in object_bundles:
            if bundle["bbox_ann"].get("id") == matching_bbox_ann_id:
                bundle["keypoint_ann"] = kp_ann
                matched = True
                print(f"[INFO] Matched keypoint id {kp_ann.get('id')} -> bbox id {matching_bbox_ann_id} (id-heuristic)")
                break

        # fallback: spatial containment if id-heuristic failed
        if not matched:
            kps = kp_ann.get("keypoints", [])
            if isinstance(kps, list) and len(kps) >= 2:
                try:
                    kx, ky = float(kps[0]), float(kps[1])
                except Exception:
                    kx, ky = None, None
                if kx is not None and ky is not None:
                    for bundle in object_bundles:
                        bx, by, bw_pix, bh_pix = bundle["bbox_ann"]["bbox"]
                        if (kx >= bx) and (kx <= bx + bw_pix) and (ky >= by) and (ky <= by + bh_pix):
                            bundle["keypoint_ann"] = kp_ann
                            matched = True
                            print(f"[INFO] Matched keypoint id {kp_ann.get('id')} -> bbox id {bundle['bbox_ann'].get('id')} (spatial fallback)")
                            break

        if not matched:
            print(f"[WARN] Could not match keypoint id {kp_ann.get('id')} to any bbox (id-heuristic failed)")

    # Build label lines for this image
    yolo_lines = []
    yolo_class_id = 0  # as per pose.yaml 'nc: 1'

    for bundle in object_bundles:
        bbox_ann = bundle["bbox_ann"]
        keypoint_ann = bundle.get("keypoint_ann")

        # COCO bbox format: [x_min, y_min, width_pix, height_pix] (absolute pixels)
        x_min, y_min, w_pix, h_pix = bbox_ann["bbox"]

        # validate / clamp bbox values
        x_min = int(round(max(0, x_min)))
        y_min = int(round(max(0, y_min)))
        w_pix = int(round(max(0, w_pix)))
        h_pix = int(round(max(0, h_pix)))

        # prepare keypoint defaults and extract if present
        kp_x, kp_y, kp_v = 0, 0, 0
        if keypoint_ann and "keypoints" in keypoint_ann and len(keypoint_ann["keypoints"]) >= 3:
            kps_raw = keypoint_ann["keypoints"]
            try:
                kp_x = int(round(kps_raw[0]))
                kp_y = int(round(kps_raw[1]))
                kp_v = int(kps_raw[2])
            except Exception:
                kp_x, kp_y, kp_v = 0, 0, 0
                print(f"[WARN] Malformed keypoint for image {img_name}, bbox id {bbox_ann.get('id')}; using dummy keypoint.")
            # clamp keypoint to image bounds
            kp_x = max(0, min(kp_x, img_w - 1))
            kp_y = max(0, min(kp_y, img_h - 1))
        else:
            print(f"[WARN] Writing dummy keypoint for image {img_name}, bbox id {bbox_ann.get('id')}")

        # Compose absolute COCO-style line: class x_min y_min width height kp_x kp_y kp_v
        line = f"{yolo_class_id} {x_min} {y_min} {w_pix} {h_pix} {kp_x} {kp_y} {kp_v}"
        yolo_lines.append(line)

    # Write all lines for this image (one file per image)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"[INFO] Generated absolute label file: {txt_path} ({len(yolo_lines)} objects)")
