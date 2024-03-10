import cv2
import os
import numpy as np
from skimage import exposure
from tqdm import tqdm

# Directory where the images are stored
img_directory = './menu_match_dataset/foodimages'
processed_img_directory = './preprocessedImages'

# Create the directory for processed images if it does not exist
if not os.path.exists(processed_img_directory):
    os.makedirs(processed_img_directory)

# Get all image file paths
img_files = [f for f in os.listdir(img_directory) if f.endswith(('.jpg', '.jpeg'))]

# Preprocess images
for img_file in tqdm(img_files):
    img_path = os.path.join(img_directory, img_file)
    # Read the image
    image = cv2.imread(img_path)
    
    # Resize the image to 128x128
    resized_image = cv2.resize(image, (128, 128))
    
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    
    # Optionally, convert to grayscale
    # grayscale_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using Adaptive Histogram Equalization
    enhanced_image = exposure.equalize_adapthist(normalized_image, clip_limit=0.03)
    
    # Save the processed image
    save_path = os.path.join(processed_img_directory, img_file)
    # Assuming that enhanced_image is in the range [0, 1], we need to scale it back to [0, 255] for saving
    cv2.imwrite(save_path, np.uint8(enhanced_image * 255))

print('Preprocessing completed.')
