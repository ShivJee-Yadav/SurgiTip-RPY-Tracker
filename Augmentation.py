# Define the augmentation pipeline using Albumentations
"""
These transformations are suitable for object detection and pose estimation

# Note: 
    Albumentations can directly handle YOLO bbox format and 'xyv' keypoint format.
    YOLO bbox format: [x_center, y_center, width, height] (normalized)
    Keypoint format: [x, y, visibility] (normalized)
    The 'class_labels' list will contain the class_id for each bbox/keypoint
    Albumentations will use this to correctly associate transformations.
"""
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw # For visualization
import os
import random
from albumentations.core.type_definitions import d4_group_elements

augment_pipeline = A.Compose(
    [

        # A.resize( will be Use Later, Useful to simulate Camera Resolution )

        # Geometric transformations
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),

        A.HorizontalFlip(p=0.5),

        # Affine transform 
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            translate_percent=(0.0625, 0.0625),
            fit_output=False,
            p=0.5
        ),

        # Perspective warp, # mild perspective distortion
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.2),

        # Elastic deformation
        A.ElasticTransform(alpha=100, sigma=8,p=0.2),

        #-------------------------------------#
        #- Pixel-level transformations       -#
        #-------------------------------------#

        # Randomly adjusts brightness and contrast (±20%)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

        # shifts hue, saturation, and value. Mimics color variations due to lighting or camera differences.
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),

        # Adds Gaussian noise with variance between 10–50. Helps the model handle noisy inputs (e.g., low-quality cameras).
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0) ,# Add camera-sensor-like noise scaling with intensity (high ISO), useful for low-light or camera noise simulation.
            A.MotionBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
            ], p=0.8),

        # Converts to grayscale occasionally. Forces the model to learn shape/texture features instead of relying only on color.
        A.ToGray(p=0.5), # Convert to grayscale occasionally

        # Randomly shuffles RGB channels. Prevents overfitting to specific color patterns.
        A.ChannelShuffle(p=0.4), # Shuffle color channels

        # Rotation| Flips -> | e | r90 | r180 | r270 | v | hvt | h | t |
        A.D4(p=0.6),

    ],

    # For keypoints, 'xyv' format is [x, y, visibility]. We pass the class_labels as well.
    bbox_params=A.BboxParams(format='yolo', label_fields=['bbox_class_labels'], min_area=10, min_visibility=0.1),
    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_class_labels', 'keypoint_visibilities'], remove_invisible=True)
)

def apply_albumentations(
    image: np.ndarray,
    yolo_labels: list[list]
    ) -> tuple[np.ndarray, list[list]]:
    """
    Applies Albumentations augmentations to an image and its YOLO-formatted labels.

    Args:
        image (np.ndarray): The input image as a NumPy array (HWC, BGR or RGB).
        yolo_labels (list[list]): A list of YOLO-formatted labels, where each label
                                   is a list: [class_id, xc, yc, w, h, kp_x, kp_y, kp_v].
                                   Coordinates (xc, yc, w, h, kp_x, kp_y) are normalized (0-1).
                                   kp_v is visibility (0, 1, or 2).

    Returns:
        tuple[np.ndarray, list[list]]: A tuple containing the augmented image and
                                         the augmented YOLO-formatted labels.
    """
    if image.shape[-1] == 3 and image.dtype == np.uint8:
        # Albumentations expects RGB. If image is BGR (common with cv2.imread), convert.
        # This function assumes image is already loaded and in appropriate color space.
        pass # No conversion needed if already RGB
    else:
        raise ValueError("Input image must be a 3-channel, uint8 numpy array (RGB).")

    bboxes = [] # For Albumentations, format: [xc, yc, w, h]
    keypoints = [] # For Albumentations, format: (kp_x, kp_y)
    class_labels_list = [] # Temporary list to collect class IDs
    keypoint_visibilities = [] # Temporary list to collect keypoint visibilities

    for label in yolo_labels:
        class_id, xc, yc, w, h = label[0:5]
        kp_x, kp_y, kp_v = label[5:8] # Assuming ONE keypoint per object

        bboxes.append([xc, yc, w, h])
        keypoints.append((kp_x, kp_y)) # Albumentations expects tuples for keypoints (x, y)
        class_labels_list.append(int(class_id)) # Class ID for both bbox and keypoint association
        keypoint_visibilities.append(int(kp_v)) # Store visibility separately

    # Explicitly construct the data dictionary for Albumentations
    # This ensures 'class_labels' is always available as a key for each processor.
    transformation_data = {
        'image': image,
        'bboxes': bboxes,
        'keypoints': keypoints,
        'bbox_class_labels': class_labels_list, # Separate list for bboxes
        'keypoint_class_labels': class_labels_list, # Separate list for keypoints
        'keypoint_visibilities': keypoint_visibilities # Pass keypoint visibilities
    }

    # Apply the augmentations using the explicitly constructed dictionary
    augmented = augment_pipeline(**transformation_data)

    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_keypoints = augmented['keypoints']
    # Albumentations might return the class labels as a list or numpy array; convert to list for consistency
    # We can pick either bbox_class_labels or keypoint_class_labels as they originate from the same list.
    augmented_class_labels = augmented['bbox_class_labels'].tolist() if isinstance(augmented['bbox_class_labels'], np.ndarray) else augmented['bbox_class_labels']
    augmented_keypoint_visibilities = augmented['keypoint_visibilities'].tolist() if isinstance(augmented['keypoint_visibilities'], np.ndarray) else augmented['keypoint_visibilities']

    # Reconstruct YOLO labels from augmented data
    augmented_yolo_labels = []
    for i in range(len(augmented_bboxes)):
        bbox = augmented_bboxes[i]
        # Albumentations returns YOLO format bboxes directly if format='yolo' is set
        xc_aug, yc_aug, w_aug, h_aug = bbox[0:4]

        # Find the corresponding keypoint and class_label using the index `i`.
        # Albumentations ensures the order is preserved for matched bboxes/keypoints *that are retained*.
        # However, if one type of annotation (e.g., keypoint) is dropped while its corresponding
        # bounding box is retained, the lengths of the lists may differ. Safely access with bounds checks.
        class_id_aug = augmented_class_labels[i] if i < len(augmented_class_labels) else 0 # Default to class 0
        kp_aug = augmented_keypoints[i] if i < len(augmented_keypoints) else (0.0, 0.0) # Default to (0,0)
        kp_v_aug = augmented_keypoint_visibilities[i] if i < len(augmented_keypoint_visibilities) else 0 # Default to visibility 0 (not labeled)

        augmented_yolo_labels.append(
            [float(class_id_aug), float(xc_aug), float(yc_aug), float(w_aug), float(h_aug),
             float(kp_aug[0]), float(kp_aug[1]), float(kp_v_aug)]
        )

    return augmented_image, augmented_yolo_labels

print("Function Working Properly")