from keras.datasets import fashion_mnist

# TensorFlow and tf.keras
import tensorflow as tf
from keras import datasets, layers, models
from keras.models import Sequential, model_from_json
from keras.models import load_model
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
# Baseline:
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
                    validation_data=(validate_images, validate_labels))

# model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
# history = model.fit(train_images, train_labels, epochs=10, batch_size=4096, verbose=1,
#                     validation_data=(validate_images, validate_labels))


# save model and weights
model_json = model.to_json()
with open("CNN1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("weights.h5")
print("Model saved")


# load json and create model
json_file = open('CNN1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded model from disk")




# Assessment of model
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# plot loss and accuracy curve
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# accuracy
print('Test Loss : {:.4f}'.format(test_loss))
print('Test Accuracy : {:.4f}'.format(test_acc))



# plot training and validation accuracy, loss.
import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
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

