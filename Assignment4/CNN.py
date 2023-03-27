from keras.datasets import fashion_mnist

# TensorFlow and tf.keras
import tensorflow as tf
from keras import datasets, layers, models
from keras.models import Sequential, model_from_json
from keras.models import load_model
# keras optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import to_categorical
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# from keras.utils import plot_model
import datetime
import pydot 
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split

print(tf.__version__)

# load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# return 4 numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

num_classes = 10
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
print(train_images.shape) # (60000*(1-ratio), 28, 28)
print(len(train_labels)) # 60000*(1-ratio)
print(train_labels) # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(validate_images.shape) # 60000*ratio
print(len(validate_labels)) # 60000*ratio
print(test_images.shape) # (10000, 28, 28)


# # check 1st image: visualization
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# pre-processing
# normalize images
train_images = train_images / 255.0
validate_images = validate_images / 255.0

test_images = test_images / 255.0

# # check data format
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# One hot encoding for categorical loss
train_labels = to_categorical(train_labels, num_classes)
validate_labels = to_categorical(validate_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)



# build model here
# TODO: need further modification
# Current: 8 layers: Input * 1 + Conv2D * 3 + Pooling * 2 + Dense * 2
# Baseline:
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Add dropouts to the model
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Add dropouts to the model
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# Add dropouts to the model
model.add(layers.Dropout(0.4))
# add Dense Layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# Add dropouts to the model
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

# .summary() will print model structure and details
model.summary()
# plot the model
tf.keras.utils.plot_model(model, to_file='model.png')
tf.keras.utils.plot_model(model, show_shapes = True, to_file='model1.png')


# compile and train
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# opt = SGD(lr=0.01, momentum=0.9)
# model.compile(optimizer=opt,
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# Optimizer
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])



history = model.fit(train_images, 
                    train_labels, 
                    epochs=15, 
                    validation_data=(validate_images, validate_labels))


# save model and weights
model_json = model.to_json()
with open("Baseline.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Baseline_weights.h5")
print("Model saved")


# load json and create model
json_file = open('Baseline.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Baseline_weights.h5")
print("Loaded model from disk")



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# accuracy
print('Test Loss : {:.4f}'.format(test_loss))
print('Test Accuracy : {:.4f}'.format(test_acc))



# plot training and validation accuracy, loss.

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Make predictions



# # Classification Report
# #Get the predictions for the test data
# predicted_classes = model.predict_classes(test_images)
# #Get the indices to be plotted
# y_true = test_images.iloc[:, 0]
# correct = np.nonzero(predicted_classes==y_true)[0]
# incorrect = np.nonzero(predicted_classes!=y_true)[0]
# from sklearn.metrics import classification_report
# target_names = ["Class {}".format(i) for i in range(num_classes)]
# print(classification_report(y_true, predicted_classes, target_names=target_names))

