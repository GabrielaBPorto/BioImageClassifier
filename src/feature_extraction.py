import os
import cv2
import numpy as np
import radiomics
import csv
import SimpleITK as sitk

RAW_ORGANIZED_DIR = "data/raw_1/organized/imagens_ihq_er"
CROPPED_IMAGES_DIR = "data/processed_1/cropped_images/"
FOLDS_DIR = "data/processed_1/folded_data/"
OTSU_MASKS_DIR = "data/processed_1/otsu_threshold_masks/"
ADAPTIVE_MASKS_DIR = "data/processed_1/adaptive_threshold_masks/"
FEATURES_OUTPUT_DIR = "data/processed_1/extracted_features/"
CROP_SIZE = (40, 30)

def extract_class_from_filename(filename):
    # Assuming the class label is at the end of the filename before the extension
    return filename.split('_')[-1].split('.')[0]


def convert_to_grayscale(image):
    if image.GetNumberOfComponentsPerPixel() == 1:
        return image
    return sitk.VectorIndexSelectionCast(image, 0, sitk.sitkFloat32)


def process_images(output_dir=FEATURES_OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold in os.listdir(FOLDS_DIR):
        fold_path = os.path.join(FOLDS_DIR, fold)
        print(f"Processing fold {fold}... {fold_path}")

        for image_file in os.listdir(fold_path):
            try:
                image_path = os.path.join(fold_path, image_file)
                class_label = extract_class_from_filename(image_file)

                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                otsu_mask = sitk.GetImageFromArray(th3)
                image = sitk.GetImageFromArray(img)

                extractorOtsu = radiomics.featureextractor.RadiomicsFeatureExtractor(os.path.abspath(path='./params.yaml'))
                result_otsu = extractorOtsu.execute(image, otsu_mask, label=255)
                output_file_path_otsu = os.path.join(output_dir, f'otsu_{fold}_{image_file}.csv')
                with open(output_file_path_otsu, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for key, value in result_otsu.items():
                        writer.writerow([key, value])

                sitk_image_array = sitk.GetArrayFromImage(image)
                if len(sitk_image_array.shape) == 3:
                    sitk_image_array = cv2.cvtColor(sitk_image_array, cv2.COLOR_BGR2GRAY)

                adaptive_mask = cv2.adaptiveThreshold(sitk_image_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 9, 2)
                adaptive_mask_sitk = sitk.GetImageFromArray(adaptive_mask)

                
                extractorAdaptive = radiomics.featureextractor.RadiomicsFeatureExtractor(os.path.abspath(path='./params.yaml'))
                result_adaptive = extractorAdaptive.execute(image, adaptive_mask_sitk, label=255)
                output_file_path_adaptive = os.path.join(output_dir, f'adaptive_{fold}_{image_file}.csv')
                with open(output_file_path_adaptive, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for key, value in result_adaptive.items():
                        writer.writerow([key, value])

            except Exception as e:
                print(f"Error processing {image_file} in {fold}: {e}")

if __name__ == "__main__":
    process_images()
