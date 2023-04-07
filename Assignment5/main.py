from read_stanford40 import train_dataset, train_labels, \
                            test_dataset, test_labels, \
                            validation_dataset, validation_labels, \
                            num_classes, keep_stanford40
import tensorflow as tf
from keras.applications import vgg16
from keras.applications import ResNet50
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
                         Flatten, Dense, GlobalAveragePooling2D
# optimizers
from tensorflow.python.keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam

import numpy as np
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

def define_model():
    # VGG16
    # model = vgg16.VGG16(include_top=False,
    #                  weights='imagenet',
    #                  input_tensor=None,
    #                  input_shape=input_shape,
    #                  pooling=None,
    #                  classes=num_classes, # 12
    #                  classifier_activation='softmax')
    # model.add(Dense(num_classes, activation="softmax"))
    # VGG 16
    model = Sequential()     
    # Input Conv1 Conv2 MaxPooling1
    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))

    # Conv3 Conv4 MaxPooling2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))

    # Conv5 Conv6 Conv7 MaxPooling3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))

    # Conv8 Conv9 Conv10 MaxPooling4
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))

    # Conv11 Conv12 Conv13 MaxPooling5
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))

    # Flatten1 Dense1 Dense2 Dense3 Output
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    
    # # ResNet18
    # resNet18 = Sequential()
    # # ResNet50
    # resnet50 = ResNet50(weights='imagenet', include_top=False)
    # resnet_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    # resnet_model = ResNet50(weights=resnet_weights)
    
    
    return model


def plot_curves(history, model_name):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model_name + 'Training and Validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name + 'Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model = define_model()

    # print model structure and details
    model.summary() 

    # plot the model
    tf.keras.utils.plot_model(model, to_file='model.png')
    tf.keras.utils.plot_model(model, show_shapes = True, to_file='model_shape.png')

    # Compile
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])


    history = model.fit(train_dataset, 
                        train_labels, 
                        validation_data=(validation_dataset, validation_labels), 
                        epochs=10)
    model.save('Model/Stanford40_Frames.h5')


    # evaluate test accuracy & loss
    test_loss, test_acc = model.evaluate(test_dataset, test_labels, verbose=2)
    print('Test Loss : {:.4f}'.format(test_loss))
    print('Test Accuracy : {:.4f}'.format(test_acc))


    # plot training and validation accuracy, loss.
    plot_curves(history=history, 
                model_name = "Stanford40_Frames")
    

    # Confusion Matrix

    from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix

    #Make predictions
    y_probs = model.predict(test_dataset)

    #Convert prediction probabilities into integers
    y_preds = y_probs.argmax(axis=1)
    # # convert one hot encoded labels to single-digit ones
    # rounded_labels = np.argmax(test_labels, axis=1)
    #Confusion matrix
    # cm=confusion_matrix(y_preds, rounded_labels)
    cm=confusion_matrix(y_preds, test_labels)
    #Plot
    disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = keep_stanford40)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax)
    plt.show()
    


