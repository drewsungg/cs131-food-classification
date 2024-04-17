import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

def load_dataset(labels_path, images_dir):
    images = []
    labels = []
    with open(labels_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split(';')
        if not parts or len(parts) < 2:
            continue
        image_file, primary_label = parts[0].strip(), parts[1].strip()
        image_path = os.path.join(images_dir, image_file)
        if os.path.exists(image_path):
            img = load_image(image_path)
            images.append(img)
            labels.append(primary_label)
    return np.array(images), np.array(labels)

images_dir = './preprocessedImages'
labels_path = './menu_match_dataset/labels.txt'
images, labels = load_dataset(labels_path, images_dir)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Prediction and Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Find unique labels in predictions and truth
unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
target_names = [label_encoder.classes_[label] for label in unique_labels]

print(classification_report(y_true, y_pred_classes, target_names=target_names, labels=unique_labels))