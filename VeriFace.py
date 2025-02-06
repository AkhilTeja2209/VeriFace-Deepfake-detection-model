import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Optimize GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Optimize input pipeline
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64 

def image_generator(folder):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) 
                  if f.lower().endswith(valid_extensions)]
    
    # Parallel image loading
    for img_path in image_paths:
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            yield img
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            continue

def create_dataset(folder, label, batch_size=BATCH_SIZE):
    try:
        dataset = tf.data.Dataset.from_generator(
            lambda: image_generator(folder),
            output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
        )
        dataset = dataset.map(lambda x: (x, label), num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    except Exception as e:
        print(f"Error creating dataset from {folder}: {str(e)}")
        return None

# Create paths and verify
train_base_path = "C:\\Users\\M V S Akhil Teja\\VeriFace\\Test"
train_real_path = os.path.join(train_base_path, "Real")
train_fake_path = os.path.join(train_base_path, "Fake")

# Create datasets with parallel processing
print("Creating training datasets...")
train_real_dataset = create_dataset(train_real_path, 0)
train_fake_dataset = create_dataset(train_fake_path, 1)

if train_real_dataset is None or train_fake_dataset is None:
    raise ValueError("Failed to create datasets")

# Combine and shuffle datasets
train_dataset = train_real_dataset.concatenate(train_fake_dataset)
train_dataset = train_dataset.shuffle(buffer_size=1000, seed=42)

# Calculate dataset sizes
dataset_size = sum(1 for _ in train_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)

# Split dataset
train_dataset = train_dataset.take(train_size)
remaining_dataset = train_dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Model building with lighter architecture
try:
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),  # Reduced from 256 to 128
        Dropout(0.3),  # Reduced dropout
        Dense(1, activation='sigmoid')
    ])

    # Compile with mixed precision
    optimizer = Adam(learning_rate=1e-3)  # Increased learning rate
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

except Exception as e:
    print(f"Error building model: {str(e)}")
    raise

# Efficient callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Training with reduced epochs
try:
    history = model.fit(
        train_dataset,
        epochs=20,  
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise

# Quick evaluation
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Final Test Accuracy: {test_acc*100:.2f}%')

# Save model
model.save('fake_face_detection_model.keras')

# Print final metrics
print("\nFinal Results:")
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")