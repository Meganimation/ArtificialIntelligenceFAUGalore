import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Download CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Show dataset shapes
print("train_images shape:", train_images.shape)
print("train_labels shape:", train_labels.shape)
print("test_images shape :", test_images.shape)
print("test_labels shape :", test_labels.shape)

# Normalize images to [0, 1]
X_train = train_images.astype("float32") / 255.0
X_test = test_images.astype("float32") / 255.0

# One-hot encode labels for categorical_crossentropy
Y_train = to_categorical(train_labels, num_classes=10)
Y_test = to_categorical(test_labels, num_classes=10)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# Display i-th image
idx = 3
image = train_images[idx]
label = train_labels[idx][0]
plt.figure()
plt.imshow(image)
plt.title(f"Label {label}")
plt.colorbar()
plt.grid(False)
plt.show()

# Display a different image
idx = 25
image = train_images[idx]
label = train_labels[idx][0]
plt.figure()
plt.imshow(image)
plt.title(f"Label {label}")
plt.colorbar()
plt.grid(False)
plt.show()

# Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
base_model.trainable = False

base_model.summary()

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation="relu")
dense_layer_2 = Dense(20, activation="relu")
prediction_layer = Dense(10, activation="softmax")

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer,
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

es = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])

# Show accuracy on test_images
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Bonus: Add BatchNormalization layer right after dense_layer_1 and compare results.

