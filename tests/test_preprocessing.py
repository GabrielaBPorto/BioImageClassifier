import unittest
import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import organize_images, crop_images, prepare_data_for_cross_validation
from preprocessing import RAW_IMAGES_DIR, RAW_ORGANIZED_DIR, CROPPED_IMAGES_DIR, FOLDS_DIR, CROP_SIZE

class TestOrganizeImages(unittest.TestCase):
    def test_directories_created(self):
        organize_images()

        for classification in os.listdir(RAW_IMAGES_DIR):
            classification_path = os.path.join(RAW_IMAGES_DIR, classification)
            if os.path.isdir(classification_path):
                for image_name in os.listdir(classification_path):
                    patient_id = image_name.split('_')[0]
                    patient_dir = os.path.join(RAW_ORGANIZED_DIR, classification, patient_id)
                    self.assertTrue(os.path.isdir(patient_dir))
                    
class TestCropImages(unittest.TestCase):
    def setUp(self):
        organize_images()
        crop_images()
    def test_images_cropped_and_saved(self):
        for classification in os.listdir(RAW_ORGANIZED_DIR):
            for patient_id in os.listdir(os.path.join(RAW_ORGANIZED_DIR, classification)):
                patient_path = os.path.join(RAW_ORGANIZED_DIR, classification, patient_id)
                for image_name in os.listdir(patient_path):
                    image_path = os.path.join(patient_path, image_name)
                    image = cv2.imread(image_path)
                    cropped_image_dir = os.path.join(CROPPED_IMAGES_DIR, classification, patient_id)
                    self.assertTrue(os.path.isdir(cropped_image_dir))
                    image_name_without_extension = os.path.splitext(image_name)[0]
                    for i in range(0, image.shape[0], CROP_SIZE[0]):
                        for j in range(0, image.shape[1], CROP_SIZE[1]):
                            cropped_image_name = f"{image_name_without_extension}_crop_{i}_{j}.png"
                            cropped_image_path = os.path.join(cropped_image_dir, cropped_image_name)
                            self.assertTrue(os.path.exists(cropped_image_path))
                            

class TestPrepareDataForCrossValidation(unittest.TestCase):
    def setUp(self):
        organize_images()
        crop_images()
        prepare_data_for_cross_validation()

    def test_folds_created_and_patient_integrity(self):
        total_patient_images = {}

        for fold in range(5):
            fold_path = os.path.join(FOLDS_DIR, f"fold_{fold}")
            self.assertTrue(os.path.isdir(fold_path))

            for patient_id in os.listdir(fold_path):
                if patient_id not in total_patient_images:
                    total_patient_images[patient_id] = fold
                else:
                    self.assertEqual(total_patient_images[patient_id], fold)

        all_patients = set()
        for fold in os.listdir(FOLDS_DIR):
            fold_path = os.path.join(FOLDS_DIR, fold)
            for patient_id in os.listdir(fold_path):
                all_patients.add(patient_id)

        self.assertEqual(len(all_patients), len(total_patient_images))

if __name__ == '__main__':
    unittest.main()
