"""## Read the train and test splits, combine them and make better splits to help training networks easier."""
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", 
                   "riding_a_horse", "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]

num_classes = len(keep_stanford40)

with open(r'Assignment5\Stanford40\ImageSplits\train.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

with open(r'Assignment5\Stanford40\ImageSplits\test.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

# Combine the splits and split for keeping more images in the training set than the test set.
all_files = train_files + test_files
all_labels = train_labels + test_labels
train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

# print(f'Train files ({len(train_files)}):\n\t{train_files}')
# print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
#       f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
# print(f'Test files ({len(test_files)}):\n\t{test_files}')
# print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
#       f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
# action_categories = sorted(list(set(train_labels)))
# print(f'Action categories ({len(action_categories)}):\n{action_categories}')


img_path = 'Assignment5/Stanford40/JPEGImages/'
# preprocessing
def preprocessing(files):
    dataset = []
    for file in files:
        img = cv2.imread(img_path + file)
        # either 112 or 224 should work really well
        # img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA) 
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) 
        img = img/255.0
        # img = img.astype('float32')/255.0
        dataset.append(np.array(img))
    dataset = np.array(dataset)
    return dataset

# use 10% of train files as validation set
train_files, validation_files, train_labels, validation_labels = train_test_split(train_files, 
                                                                                  train_labels, 
                                                                                  test_size=0.1,
                                                                                  stratify= train_labels)

train_dataset = preprocessing(train_files)
test_dataset = preprocessing(test_files)
validation_dataset = preprocessing(validation_files)

# transform class label into number code
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)
validation_labels = le.transform(validation_labels)
print("\n\n after label encoder:")
print("train labels: ", train_labels) # 2733 = 2459 + 274
print("validation labels: \n", len(validation_labels), validation_labels) # 274
print("test labels: ", test_labels) # 304



# # if use categorical_crossentropy in compile(loss), one hot encoding is needed
# from keras.utils import to_categorical
# train_labels = to_categorical(train_labels, num_classes)
# test_labels = to_categorical(test_labels, num_classes)
# validation_labels = to_categorical(validation_labels, num_classes)
# # validate_labels = to_categorical(validate_labels, num_classes)
# print("\n\n after one hot encoding:")
# print("train labels: \n", len(train_labels), train_labels)
# print("validation labels: \n", len(validation_labels), validation_labels) 
# print("test labels: \n", len( test_labels), test_labels)












# """### Visualize a photo from the training files and also print its label"""

# # from google.colab.patches import cv2_imshow

# image_no = 234  # change this to a number between [0, 1200] and you can see a different training image
# img = cv2.imread(f'Assignment5\Stanford40\JPEGImages\{train_files[image_no]}')
# print(f'An image with the label - {train_labels[image_no]}')
# cv2.imshow("img_example", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()