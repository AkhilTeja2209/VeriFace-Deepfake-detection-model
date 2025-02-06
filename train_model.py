import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

def image_generator(folder):
    valid_extensions = ('.jpg', '.jpeg', '.png')  # Added more extensions
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            try:
                img_path = os.path.join(folder, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize here
                yield img_array
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

def create_dataset(folder, label, batch_size=32):
    try:
        dataset = tf.data.Dataset.from_generator(
            lambda: image_generator(folder),
            output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
        )
        dataset = dataset.map(lambda x: (x, label))
        dataset = dataset.batch(batch_size)
        return dataset
    except Exception as e:
        print(f"Error creating dataset from {folder}: {str(e)}")
        return None

# Create paths and verify
train_base_path = "C:\\Users\\M V S Akhil Teja\\VeriFace\\Test"
train_real_path = os.path.join(train_base_path, "Real")
train_fake_path = os.path.join(train_base_path, "Fake")

# Verify directories
for path in [train_real_path, train_fake_path]:
    if not os.path.exists(path):
        raise ValueError(f"Directory not found: {path}")

# Create datasets with error handling
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

# Improved preprocessing function
def preprocess_data(image, label):
    try:
        image = tf.cast(image, tf.float32)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_data)
val_dataset = val_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

# Model building with error handling
try:
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile with gradient clipping
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

except Exception as e:
    print(f"Error building model: {str(e)}")
    raise

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
]

# Training with error handling
try:
    history = model.fit(
        train_dataset,
        epochs=2,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise

# Save model
try:
    model.save('fake_face_detection_model.keras')
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Evaluation and visualization
try:
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Final Test Accuracy: {test_acc*100:.2f}%')

    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Print final metrics
    print("\nFinal Results:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

except Exception as e:
    print(f"Error in evaluation and visualization: {str(e)}")