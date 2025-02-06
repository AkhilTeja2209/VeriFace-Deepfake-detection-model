import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

def image_generator(folder):
    valid_extensions = ('.jpg')
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            try:
                img_path = os.path.join(folder, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                yield img_array
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

def create_dataset(folder, label, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(folder),
        output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )
    dataset = dataset.map(lambda x: (x, label))
    dataset = dataset.batch(batch_size)
    return dataset

#Create paths
# Set up paths
train_base_path = "C:\\Users\\M V S Akhil Teja\\VeriFace\\Test"

# Create all necessary paths
train_real_path = os.path.join(train_base_path, "Real")
train_fake_path = os.path.join(train_base_path, "Fake")

# Verify all directories exist
all_paths = [train_real_path, train_fake_path]
for path in all_paths:
    if not os.path.exists(path):
        raise ValueError(f"Directory not found: {path}")

# Create datasets
print("Creating training datasets...")
train_real_dataset = create_dataset(train_real_path, 0)
train_fake_dataset = create_dataset(train_fake_path, 1)

# Combine datasets
train_real_dataset = train_real_dataset.unbatch()
train_fake_dataset = train_fake_dataset.unbatch()

# Combine datasets
train_dataset = train_real_dataset.concatenate(train_fake_dataset)

# Shuffle the combined dataset
train_dataset = train_dataset.shuffle(buffer_size=1000)

# Split the dataset into train, validation, and test sets
dataset_size = sum(1 for _ in train_dataset)
train_size = int(0.7 * dataset_size)  # 70% for training
val_size = int(0.15 * dataset_size)   # 15% for validation
test_size = dataset_size - train_size - val_size  # 15% for testing

train_dataset = train_dataset.take(train_size)
val_dataset = train_dataset.skip(train_size).take(val_size)
test_dataset = train_dataset.skip(train_size + val_size).take(test_size)

# Apply data preprocessing
def preprocess_data(image, label):
    image=tf.cast(image,tf.float32)
    image=tf.clip_by_value(image,0,255)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess_data)
val_dataset = val_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

train_dataset = train_dataset.batch(32)
val_dataset = val_dataset.batch(32)
test_dataset = test_dataset.batch(32)

# Cache and prefetch for better performance
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

# Build model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256),
    BatchNormalization(),  
    tf.keras.layers.ReLU(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile with gradient clipping
optimizer=Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]



# Train model
try:
    history = model.fit(
        train_dataset,
        epochs=1,
        validation_data=val_dataset,
        callbacks=callbacks,  
        verbose=1
    )
except Exception as e:
    print(f"Error training model: {str(e)}")
    raise

# Final evaluation on test set
test_loss, test_acc = model.evaluate(train_dataset)
print(f'Final Test Accuracy: {test_acc*100:.2f}%')

# Save model
model.save('fake_face_detection_model.keras')

# Plot training history
plt.figure(figsize=(15, 5))

# Plot training & validation accuracy values
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Print final metrics
print("\nFinal Results:")
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
