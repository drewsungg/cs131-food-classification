from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
import numpy as np
import os
import glob

# Load the labels
with open('/Users/andrewsung/Desktop/andrewFoodProjCS131/menu_match_dataset/labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Load the images
img_files = glob.glob('/path/to/your/images/*.jpg')
images = [cv2.imread(img) for img in img_files]

# Flatten the images
features = [img.flatten() for img in images]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a SVM classifier
clf = svm.SVC()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Print the classification report
print(metrics.classification_report(y_test, y_pred))