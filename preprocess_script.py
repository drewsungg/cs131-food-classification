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
    resized_image = cv2.resize(image, (500, 500))
    
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(np.uint8(normalized_image * 255), cv2.COLOR_BGR2Lab)
    
    # Split the LAB image into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l_channel = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L channel with the original A and B channel
    enhanced_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    
    # Convert the LAB image back to BGR color space
    clahe_image = cv2.cvtColor(enhanced_image, cv2.COLOR_Lab2BGR)
    
    # Save the processed image
    save_path = os.path.join(processed_img_directory, img_file)
    cv2.imwrite(save_path, clahe_image)

print('Preprocessing completed.')