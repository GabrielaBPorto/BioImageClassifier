import unittest
import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import organize_images, crop_images, RAW_IMAGES_DIR, RAW_ORGANIZED_DIR, CROPPED_IMAGES_DIR,  CROP_SIZE

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
    def test_images_cropped_and_saved(self):
        crop_images()

        for classification in os.listdir(RAW_ORGANIZED_DIR):
            for patient_id in os.listdir(os.path.join(RAW_ORGANIZED_DIR, classification)):
                patient_path = os.path.join(RAW_ORGANIZED_DIR, classification, patient_id)
                for image_name in os.listdir(patient_path):
                    image_path = os.path.join(patient_path, image_name)
                    image = cv2.imread(image_path)
                    cropped_image_dir = os.path.join(CROPPED_IMAGES_DIR, classification, patient_id)
                    self.assertTrue(os.path.isdir(cropped_image_dir))
                    for i in range(0, image.shape[0], CROP_SIZE[0]):
                        for j in range(0, image.shape[1], CROP_SIZE[1]):
                            cropped_image_name = f"{image_name}_crop_{i}_{j}.png"
                            cropped_image_path = os.path.join(cropped_image_dir, cropped_image_name)
                            self.assertTrue(os.path.exists(cropped_image_path))
                            
         
if __name__ == '__main__':
    unittest.main()