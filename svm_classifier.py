import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import os

# Define a function to extract HOG features from a color image
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_resized = cv2.resize(image, (500, 500)) 
    
    # Specify channel_axis=-1 for color images to indicate that the color channels are the last axis
    features, _ = hog(image_resized, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), block_norm='L2', visualize=True, channel_axis=-1)
    return features

# Load image paths and labels
def load_dataset(labels_path, images_dir):
    with open(labels_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    image_paths = []
    labels = []
    
    for line in lines:
        parts = line.strip().split(';')
        if not parts or len(parts) < 2:
            continue
        
        image_file = parts[0].strip()
        image_labels = parts[1:]
        primary_label = image_labels[0].strip()
        
        image_path = os.path.join(images_dir, image_file)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(primary_label)
    
    return image_paths, labels


if __name__ == "__main__":
    labels_path = './menu_match_dataset/labels.txt'  
    images_dir = './preprocessedImages'  
 
    # Load the dataset
    image_paths, labels = load_dataset(labels_path, images_dir)
    
    # Extract HOG features for each image
    features = [extract_hog_features(path) for path in image_paths]
    
    # Convert features and labels to numpy arrays for sklearn compatibility
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)
    
    # Initialize and train the SVM classifier
    classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Predict using the trained classifier
    predictions = classifier.predict(X_test)
    
    # Evaluate the classifier
    print("Classification report:\n", metrics.classification_report(y_test, predictions))
    print("Accuracy:", metrics.accuracy_score(y_test, predictions))