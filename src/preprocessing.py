import os
import sys
import shutil
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

RAW_IMAGES_DIR = "data/raw/input/imagens_ihq_er/"
RAW_ORGANIZED_DIR = "data/raw/organized/imagens_ihq_er"
CROPPED_IMAGES_DIR = "data/processed/cropped_images/"
FOLDS_DIR = "data/processed/folded_data/"
CROP_SIZE = (40, 30)

def organize_images():
    # """
    # Organize images by classification and patients
    # """
    for classification in os.listdir(RAW_IMAGES_DIR):
        classification_path = os.path.join(RAW_IMAGES_DIR, classification)
        for image_name in os.listdir(classification_path):
            patient_id = image_name.split('_')[0]
            patient_dir = os.path.join(RAW_ORGANIZED_DIR, classification, patient_id)            
            os.makedirs(patient_dir, exist_ok=True)
            src = os.path.join(classification_path, image_name)
            dest = os.path.join(patient_dir, image_name)
            shutil.copy(src, dest)

def crop_images():
    # """
    # Crop images by patient
    # """
    for classification in os.listdir(RAW_ORGANIZED_DIR):
        classification_path = os.path.join(RAW_ORGANIZED_DIR, classification)
        for patient_id in os.listdir(classification_path):
            cropped_image_dir = os.path.join(CROPPED_IMAGES_DIR, classification, patient_id)
            os.makedirs(cropped_image_dir, exist_ok=True)
            patient_path = os.path.join(classification_path, patient_id)
            for image_name in os.listdir(patient_path):
                image_path = os.path.join(patient_path, image_name)
                image = cv2.imread(image_path)
                image_name_without_extension = os.path.splitext(image_name)[0]
                for i in range(0, image.shape[0], CROP_SIZE[0]):
                    for j in range(0, image.shape[1], CROP_SIZE[1]):
                        cropped_image = image[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1]]
                        cropped_image_name = f"{image_name_without_extension}_crop_{i}_{j}.png"
                        cv2.imwrite(os.path.join(cropped_image_dir, cropped_image_name), cropped_image)
                        
def prepare_data_for_cross_validation():
    # """
    # Prepare data for 5-fold cross-validation.
    # """
    patient_images = {}
    for classification in os.listdir(CROPPED_IMAGES_DIR):
        classification_path = os.path.join(CROPPED_IMAGES_DIR, classification)
        for patient_id in os.listdir(classification_path):
            patient_path = os.path.join(classification_path, patient_id)
            patient_images[patient_id] = [os.path.join(patient_path, img) for img in os.listdir(patient_path)]
    folds = {i: [] for i in range(5)}
    patient_ids = list(patient_images.keys())
    num_patients_per_fold = len(patient_ids) // 5
    current_fold = 0
    for i, patient_id in enumerate(patient_ids):
        if i != 0 and i % num_patients_per_fold == 0 and current_fold < 4:
            current_fold += 1
        folds[current_fold].extend(patient_images[patient_id])

    for fold_index in folds.keys():
        fold_dir = os.path.join(FOLDS_DIR, f"fold_{fold_index}")
        os.makedirs(fold_dir, exist_ok=True)

        for image_path in folds[fold_index]:
            filename = os.path.basename(image_path)
            dest_path = os.path.join(fold_dir, filename)
            shutil.copy(image_path, dest_path)


# if __name__ == "__main__":
#     organize_images()
#     print('Finished Organizing')
#     crop_images()
#     print('Finished Cropping')
#     prepare_data_for_cross_validation()
#     print('Finished Preparation of data')
    