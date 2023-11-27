import radiomics
import os
import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_local

RAW_IMAGES_DIR = "data/raw/input/imagens_ihq_er/"
RAW_ORGANIZED_DIR = "data/raw/organized/imagens_ihq_er"
CROPPED_IMAGES_DIR = "data/processed/cropped_images/"
FOLDS_DIR = "data/processed/folded_data/"
OTSU_MASKS_DIR = "data/processed/otsu_threshold_masks/"
ADAPTIVE_MASKS_DIR = "data/processed/adaptive_threshold_masks/"
FEATURES_OUTPUT_DIR = "data/processed/extracted_features/"
CROP_SIZE = (40, 30)


def apply_threshold(image_path, method='otsu'):
    """
    Apply thresholding to the image by possible methods.
    """
    image = cv2.imread(image_path, 0)

    if method == 'otsu':
        thresh_value = threshold_otsu(image)
        binary_image = image > thresh_value
    elif method == 'adaptive':
        block_size = 35
        binary_image = threshold_local(image, block_size, offset=10)
        binary_image = image > binary_image
    else:
        raise ValueError("Invalid thresholding method")

    return np.uint8(binary_image * 255)

def process_images():
    """
    Process images for feature extraction from each fold in FOLDS_DIR.
    """
    for fold in os.listdir(FOLDS_DIR):
        fold_path = os.path.join(FOLDS_DIR, fold)

        
        fold_otsu_dir = os.path.join(OTSU_MASKS_DIR, fold)
        fold_adaptive_dir = os.path.join(ADAPTIVE_MASKS_DIR, fold)
        os.makedirs(fold_otsu_dir, exist_ok=True)
        os.makedirs(fold_adaptive_dir, exist_ok=True)
        for image_file in os.listdir(fold_path):
            image_path = os.path.join(fold_path, image_file)

            otsu_mask = apply_threshold(image_path, method='otsu')
            otsu_mask_path = os.path.join(fold_otsu_dir, f"otsu_{image_file}")
            cv2.imwrite(otsu_mask_path, otsu_mask)

            adaptive_mask = apply_threshold(image_path, method='adaptive')
            adaptive_mask_path = os.path.join(fold_adaptive_dir, f"adaptive_{image_file}")
            cv2.imwrite(adaptive_mask_path, adaptive_mask)

if __name__ == "__main__":
    process_images()
