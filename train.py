# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_binary_model():
    """Create model for benign/malignant classification"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_tumor_size(image):
    """Extract tumor size using contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.contourArea(largest_contour)
    return 0


def determine_stage(size):
    """Determine cancer stage based on tumor size"""
    if size < 1000:
        return 0  # Stage 1
    elif size < 2000:
        return 1  # Stage 2
    elif size < 3000:
        return 2  # Stage 3
    elif size < 4000:
        return 3  # Stage 4
    else:
        return 4  # Stage 5


def create_stage_model():
    """Create model for stage classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 5 stages
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def prepare_stage_dataset(malignant_folder):
    """Prepare dataset with stage labels based on tumor size"""
    images = []
    stages = []

    for img_name in os.listdir(malignant_folder):
        img_path = os.path.join(malignant_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tumor_size = get_tumor_size(img)
        stage = determine_stage(tumor_size)

        images.append(img)
        stages.append(stage)

    return np.array(images), np.array(stages)


def train_models(data_dir):
    """Train both binary and stage classification models"""
    # Setup data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Binary classification data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Train binary classification model
    print("Training binary classification model...")
    binary_model = create_binary_model()

    binary_checkpoint = ModelCheckpoint(
        'binary_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    binary_history = binary_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=[binary_checkpoint, early_stopping]
    )

    # Prepare and train stage classification model
    print("Preparing stage classification data...")
    malignant_folder = os.path.join(data_dir, 'malignant')
    images, stages = prepare_stage_dataset(malignant_folder)

    # Split the stage classification data
    X_train, X_test, y_train, y_test = train_test_split(
        images, stages, test_size=0.2, random_state=42
    )

    # Train stage classification model
    print("Training stage classification model...")
    stage_model = create_stage_model()

    stage_checkpoint = ModelCheckpoint(
        'stage_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )

    stage_history = stage_model.fit(
        X_train / 255.0,
        y_train,
        validation_data=(X_test / 255.0, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[stage_checkpoint, early_stopping]
    )

    # Plot training histories
    plot_training_history(binary_history, 'Binary Classification')
    plot_training_history(stage_history, 'Stage Classification')


def plot_training_history(history, title):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_history.png')
    plt.close()


if __name__ == "__main__":
    # Set your data directory path
    DATA_DIR = "data/train"  # Should contain 'benign' and 'malignant' folders
    train_models(DATA_DIR)