import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))  # Resize to MobileNetV2's expected input
    img = img / 255.0
    return img

def load_dataset(labels_path, images_dir):
    images, labels = [], []
    with open(labels_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                image_path = os.path.join(images_dir, parts[0])
                if os.path.exists(image_path):
                    images.append(load_image(image_path))
                    labels.append(parts[1])
    return np.array(images), np.array(labels)

# Update these paths according to your dataset
images_dir = './preprocessedImages'
labels_path = './menu_match_dataset/labels.txt'

images, labels = load_dataset(labels_path, images_dir)

# Encoding labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False  # Freeze the base model

inputs = tf.keras.Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(len(label_encoder.classes_), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    epochs=60,
                    validation_data=(X_test, y_test))

# Plotting training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Prediction and generate a classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate an array of indices representing each class
label_indices = np.arange(len(label_encoder.classes_))

# Use the indices as the 'labels' parameter in classification_report
print(classification_report(y_true, y_pred_classes, labels=label_indices, target_names=label_encoder.classes_))
