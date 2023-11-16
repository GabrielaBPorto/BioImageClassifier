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

def organize_images( ):
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
    # Para cada diretório de classificação
    for classification in os.listdir(RAW_ORGANIZED_DIR):
        classification_path = os.path.join(RAW_ORGANIZED_DIR, classification)
        # Para cada diretório de paciente
        for patient_id in os.listdir(classification_path):
            cropped_image_dir = os.path.join(CROPPED_IMAGES_DIR, classification, patient_id)
            os.makedirs(cropped_image_dir, exist_ok=True)
            patient_path = os.path.join(classification_path, patient_id)
            # Para cada imagem no diretório do paciente
            for image_name in os.listdir(patient_path):
                image_path = os.path.join(patient_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Crop and save the image
                    for i in range(0, image.shape[0], CROP_SIZE[0]):
                        for j in range(0, image.shape[1], CROP_SIZE[1]):
                            cropped_image = image[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1]]
                            cropped_image_name = f"{image_name}_crop_{i}_{j}.png"
                            cv2.imwrite(os.path.join(cropped_image_dir, cropped_image_name), cropped_image)
                else:
                    print(f"Error reading image: {image_path}")
                        
# def crop_image(image_path, output_path=CROPPED_IMAGES_DIR, crop_size=CROP_SIZE):
#     """
#     Crop an image into smaller images of specified size and save them.

#     Args:
#         image_path: Path to the image to be cropped.
#         output_path (str): Directory where cropped images will be saved.
#         crop_size (tuple): Size of the cropped images.
#     """
#     img = cv2.imread(image_path)
#     height, width = img.shape[:2]
#     for i in range(0, height, crop_size[1]):
#         for j in range(0, width, crop_size[0]):
#             crop = img[i:i + crop_size[1], j:j + crop_size[0]]
#             cv2.imwrite(os.path.join(output_path, f"{os.path.basename(image_path)}_{i}_{j}.png"), crop)

# def prepare_data_for_cross_validation(cropped_images_directory=CROPPED_IMAGES_DIR, folds_directory=FOLDS_DIR):
    # """
    # Prepare data for 5-fold cross-validation.

    # Args:
    #     cropped_images_directory (str): Directory containing cropped images.
    #     folds_directory (str): Directory where data for each fold will be stored.
    # """
    # images = organize_images(cropped_images_directory)
    # labels = [os.path.basename(img).split('_')[0] for img in images]
    # skf = StratifiedKFold(n_splits=5)
    # for fold, (train_index, test_index) in enumerate(skf.split(images, labels)):
    #     train_images = np.array(images)[train_index]
    #     test_images = np.array(images)[test_index]
    #     np.save(os.path.join(folds_directory, f"train_fold_{fold}.npy"), train_images)
    #     np.save(os.path.join(folds_directory, f"test_fold_{fold}.npy"), test_images)



    # for image_path in organized_images:
    #     crop_image(image_path)

    # prepare_data_for_cross_validation()

if __name__ == "__main__":
    organize_images()
    