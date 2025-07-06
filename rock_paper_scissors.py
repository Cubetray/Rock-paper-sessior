# Install required packages if needed (uncomment if running in Colab)
# !pip install tensorflow tensorflow_datasets matplotlib

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Set image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Load rock_paper_scissors dataset (split into training and validation sets)
(train_ds, val_ds), info = tfds.load('rock_paper_scissors',
                                     split=['train[:80%]', 'train[80%:]'],
                                     as_supervised=True,
                                     with_info=True)

# Preprocessing function: resize and normalize images
def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

# Apply preprocessing and batching
train_ds = train_ds.map(format_image).shuffle(1000).batch(BATCH_SIZE)
val_ds = val_ds.map(format_image).batch(BATCH_SIZE)

# Load pre-trained MobileNetV2 model (without the top layer)
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=5)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
Added transfer learning project code
