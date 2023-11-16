import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import organize_images, RAW_IMAGES_DIR, RAW_ORGANIZED_DIR

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
                    
if __name__ == '__main__':
    unittest.main()