import numpy as np
import albumentations as A

# (Use your existing augment_pipeline but change KeypointParams and BboxParams there:)
augment_pipeline = A.Compose(
   [
    # 2. MULTI-RESOLUTION RESIZING 
    A.OneOf([
        A.Resize(width=416, height=416, p=1.0),
        A.Resize(width=640, height=640, p=1.0),
        A.Resize(width=1280, height=1280, p=1.0)
    ], p=0.5),

    # 3. GEOMETRIC TRANSFORMS
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Affine(
        scale=(0.8, 1.2),      
        rotate=(-45, 45),      
        shear=(-10, 10), 
        border_mode=cv2.BORDER_CONSTANT,  # FIXED: Changed 'mode' to 'border_mode'
        p=0.6
    ),

    # 4. SURGICAL ARTIFACTS
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.RandomShadow(
        num_shadows_range=(1, 2),        # FIXED: Changed to 'num_shadows_range' tuple
        shadow_dimension=5, 
        p=0.4
    ),
    
    # AdvancedBlur uses blur_range, sigma_x_range, sigma_y_range natively
    A.AdvancedBlur(blur_range=(3, 7), p=0.3), 

    # 5. CAMERA & ENVIRONMENT NOISE
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),       
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),   
        A.GaussNoise(std_range=(0.05, 0.2), p=1.0), # FIXED: Changed 'var_limit' to 'std_range' (fraction of max value)
    ], p=0.6),

    A.ImageCompression(
        quality_range=(40, 100),         # FIXED: Changed lower/upper to 'quality_range' tuple
        p=0.4
    ),
    ],
    # Use COCO absolute-pixel bbox format for augmentation
    bbox_params=A.BboxParams(format='coco', label_fields=['bbox_class_labels'],
                             min_area=0, min_visibility=0),
    # Keypoints must be passed as pixel (x,y)
    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_class_labels', 'keypoint_visibilities'],
                                     remove_invisible=True)
)


def apply_albumentations(image: np.ndarray, absolute_labels: list[list]):
    """
    image: HxWx3 uint8 (BGR or RGB, be consistent)
    absolute_labels: list of [class_id, x_min, y_min, w_pix, h_pix, kp_x_px, kp_y_px, kp_v]
                     all coordinates are in absolute pixels (COCO bbox + pixel keypoints)
    Returns: augmented_image, augmented_absolute_labels (same absolute COCO + pixel keypoint format)
    """
    h, w = image.shape[:2]

    # Build lists in pixel coordinates for Albumentations (COCO format)
    bboxes = []
    keypoints = []
    bbox_class_labels = []
    keypoint_class_labels = []
    keypoint_visibilities = []

    for lab in absolute_labels:
        cls = int(lab[0])
        x_min, y_min, bw_pix, bh_pix = map(float, lab[1:5])
        kx_px, ky_px, kv = map(float, lab[5:8])

        # COCO format expects [x_min, y_min, width, height] in pixels
        bboxes.append([x_min, y_min, bw_pix, bh_pix])
        bbox_class_labels.append(cls)

        # Keypoints as pixel tuples for KeypointParams(format='xy')
        keypoints.append((kx_px, ky_px))
        keypoint_class_labels.append(cls)
        keypoint_visibilities.append(int(kv))

    data = {
        'image': image,
        'bboxes': bboxes,
        'keypoints': keypoints,
        'bbox_class_labels': bbox_class_labels,
        'keypoint_class_labels': keypoint_class_labels,
        'keypoint_visibilities': keypoint_visibilities
    }

    augmented = augment_pipeline(**data)

    aug_img = augmented['image']
    aug_bboxes = augmented.get('bboxes', [])            # COCO pixel format: [x_min, y_min, w, h]
    aug_keypoints_px = augmented.get('keypoints', [])   # pixel coords (x, y)
    aug_bbox_labels = augmented.get('bbox_class_labels', [])
    aug_kp_vis = augmented.get('keypoint_visibilities', [])

    # Reconstruct absolute COCO-style labels and recompute visibility robustly
    aug_h, aug_w = aug_img.shape[:2]
    out_labels = []

    # Iterate over original count to preserve object ordering; Albumentations preserves order for matched annotations
    n = len(bboxes)
    for i in range(n):
        # bbox (COCO pixel)
        if i < len(aug_bboxes):
            x_min_a, y_min_a, w_a, h_a = aug_bboxes[i]
        else:
            x_min_a, y_min_a, w_a, h_a = 0.0, 0.0, 0.0, 0.0

        # keypoint (pixel)
        if i < len(aug_keypoints_px):
            kx_px_a, ky_px_a = aug_keypoints_px[i]
        else:
            kx_px_a, ky_px_a = 0.0, 0.0

        # recompute visibility: 0 outside image, 2 inside bbox, 1 inside image but outside bbox
        vis = 0
        if 0.0 <= kx_px_a <= (aug_w - 1) and 0.0 <= ky_px_a <= (aug_h - 1):
            # inside image bounds
            x1 = x_min_a
            y1 = y_min_a
            x2 = x_min_a + w_a
            y2 = y_min_a + h_a
            if x1 <= kx_px_a <= x2 and y1 <= ky_px_a <= y2:
                vis = 2
            else:
                vis = 1
        else:
            vis = 0

        cls_aug = int(aug_bbox_labels[i]) if i < len(aug_bbox_labels) else 0

        # Clamp and cast to integers for absolute pixel output
        x_min_a_i = int(round(max(0, x_min_a)))
        y_min_a_i = int(round(max(0, y_min_a)))
        w_a_i = int(round(max(0, w_a)))
        h_a_i = int(round(max(0, h_a)))
        kx_px_i = int(round(max(0, min(kx_px_a, aug_w - 1))))
        ky_px_i = int(round(max(0, min(ky_px_a, aug_h - 1))))
        vis_i = int(vis)

        out_labels.append([
            cls_aug,
            x_min_a_i, y_min_a_i, w_a_i, h_a_i,
            kx_px_i, ky_px_i, vis_i
        ])

    return aug_img, out_labels
