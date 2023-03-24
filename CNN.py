from keras.datasets import fashion_mnist

# TensorFlow and tf.keras
import tensorflow as tf
from keras import datasets, layers, models
# keras optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print(tf.__version__)

# load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# return 4 numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


### create the validation dataset for validation purposes:
### split the train dataset into train + validation dataset
validate_train_ratio = 0.2
# validation_images = np.split(train_images, )
train_images, validate_images, train_labels, validate_labels = train_test_split(train_images,
                                                                                train_labels,
                                                                                test_size = validate_train_ratio,
                                                                                random_state = 12345)
# check dataset info
print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(validate_images.shape)
print(len(validate_labels))
print(test_images.shape) # (10000, 28, 28)


# check 1st image: visualization
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# pre-processing
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

# add Dense Layer
model.add(layers.Flatten())
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# compile and train
# opt = SGD(lr=0.01, momentum=0.9)
# model.compile(optimizer=opt,
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
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