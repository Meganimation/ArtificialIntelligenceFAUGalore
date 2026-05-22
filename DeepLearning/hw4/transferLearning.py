# ‘’’
# YOUR WORK HERE
# Download cifar10 dataset.
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Show the shape of (train_images, train_labels), (test_images, test_labels)
#  ‘’’

X_train=train_images
X_test=test_images
Y_train=train_labels
Y_test=test_labels

print(X_train.shape)
print(X_test.shape)


# display i-th image
idx=3
image = X_train[idx]
label = Y_train[idx]
image = image.numpy()
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title('Label {}'.format(label))
plt.colorbar()
plt.grid(False)
plt.show()
# ‘’’
# YOUR WORK HERE
# Show give a diffent number of idx and display the image
#  ‘’’

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
base_model.trainable = False  ## Not trainable weights

base_model.summary()

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(10, activation='softmax')

model = models.Sequential([
    OOO
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    OOO
])
# Making the Predictions
‘’’
12. 1)-3 (4 pts)
YOUR WORK HERE
Complete OOO parts above
‘’’

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])

# Making the Predictions
# ‘’’
# YOUR WORK HERE
# Show accuracy of test_images
# ‘’’

# Bonus: Add BatchNormalization layer right after every dense_layer_1 layer. Compare the results.

