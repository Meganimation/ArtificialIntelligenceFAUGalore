#Classifying Cifar10 images using CNN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility (optional)
tf.keras.utils.set_random_seed(0)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Show the .shape of (train_images, train_labels), (test_images, test_labels)
# use .shape


# Preprocess (normalize)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# change to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

# Try data augmentation
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomRotation(0.05),
])

# Build a simple CNN
inputs = keras.Input(shape=(32, 32, 3))
x = data_aug(inputs)

# Block 1
x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)

# Block 2
x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)

# Block 3
x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
x = layers.ReLU()(x)

# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# YOUR WORK HERE
# Show the network structure using model.summary()

# Compile
# Using model.compile, compile the model with Adam, categorical_crossentropy


# Train
# Using model.fit, train the model with x_train, y_train data
# Use validation_split=0.1
# Use your own values of epochs and batch_size

# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
