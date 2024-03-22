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
    img = cv2.resize(img, (160, 160))  # Resize to MobileNetV2 expected size
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

# Load your dataset
images_dir = './preprocessedImages'  # Update this path
labels_path = './menu_match_dataset/labels.txt'  # Update this path
images, labels = load_dataset(labels_path, images_dir)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

# Load MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Freeze the base model
base_model.trainable = False

# Create a new model on top
inputs = tf.keras.Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Regularization with dropout
outputs = Dense(len(label_encoder.classes_), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Predict and generate a classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Predict and generate a classification report with all classes
all_class_indices = range(len(label_encoder.classes_))
all_class_names = label_encoder.classes_

print(classification_report(y_true, y_pred_classes, labels=all_class_indices, target_names=all_class_names))