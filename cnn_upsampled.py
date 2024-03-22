import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

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

def augment_images(X_train, y_train, augment_factor=2):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented_images = []
    augmented_labels = []
    for _ in range(augment_factor):
        for X, y in datagen.flow(X_train, y_train, batch_size=len(y_train), shuffle=False):
            augmented_images.extend(X)
            augmented_labels.extend(y)
            break 
    return np.array(augmented_images), np.array(augmented_labels)

images_dir = './preprocessedImages'
labels_path = './menu_match_dataset/labels.txt'
images, labels = load_dataset(labels_path, images_dir)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

# Upsample/augment data for underrepresented classes
augmented_images, augmented_labels = augment_images(X_train, y_train, augment_factor=3)
X_train = np.concatenate((X_train, augmented_images))
y_train = np.concatenate((y_train, augmented_labels))

# Adjust class weights to focus more on underrepresented classes
class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
class_weights_dict = dict(enumerate(class_weights))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=400, validation_data=(X_test, y_test), class_weight=class_weights_dict)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Adjust the target names to match those that are present in the y_true and y_pred_classes
unique_labels_indices = np.unique(np.concatenate((y_true, y_pred_classes)))
unique_target_names = [label_encoder.classes_[i] for i in unique_labels_indices]

print(classification_report(y_true, y_pred_classes, target_names=unique_target_names, labels=unique_labels_indices))
