from keras.datasets import fashion_mnist

# TensorFlow and tf.keras
import tensorflow as tf
from keras import datasets, layers, models
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# 加载数据集会返回四个 NumPy 数组
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# check dataset info
train_images.shape # (60000, 28, 28)
len(train_labels) # 60000
train_labels # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
test_images.shape # 

# pre-processing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalize images
train_images = train_images / 255.0

test_images = test_images / 255.0

# check data format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# build model here
# TODO: need further modification
# Current: 8 layers: Input * 1 + Conv2D * 3 + Pooling * 2 + Dense * 2
model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation='relu'))

# .summary() will print model structure and details
model.summary()

# add Dense Layer???
model.add(layers.Flatten())
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# compile and train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# Assessment of model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# plot loss and accuracy curve
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# accuracy
print(test_acc)