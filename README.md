### [Assignment 4: Convolutional neural networks](https://uu.blackboard.com/webapps/assignment/uploadAssignment?content_id=_4411485_1&course_id=_140663_1&group_id=&mode=view)

**Overall goal**\
In this assignment, you will get hands-on experience in training and validating CNN models, and how to report the performance of the model. It acts as the foundation for the final assignment.

**Software and installation**\
We will program in Python. We recommend any version above Python 3.6. We will use [Tensorflow 2](https://www.tensorflow.org/versions). In this version, Keras is provided as a package for Tensorflow. Keras will greatly facilitate the development and validation of CNNs. Once you have Python installed, you can simply install TensorFlow with this pip command: $pip install tensorflow. See also for GPU support in Tensorflow: <https://www.tensorflow.org/install/gpu> 

**Data**\
The data for this assignment is the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset from Keras. The dataset is already available as part of TensorFlow/Keras, you can import it by: from keras.datasets import fashion_mnist. It contains 60k training and 10k test images. Each image has one out of ten possible class labels: t-shirts/top, trousers/pants, pullover shirt, dress, coat, sandal, shirt, sneaker, bag or ankle boot. Images are greyscale (single channel) of sizes 28x28.

**Running experiments**\
If your computer has a GPU with CUDA enabled, you can benefit from significant speed-up, especially when training your models. You can make use of computation services such as [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). Google Cloud will also give you some free allowance on sign-in. Microsoft offers free [Azure](https://azure.microsoft.com/en-us/developer/students/) access for students. There are also free cloud GPU services from Kaggle as well. **Important**: Don't forget to shut down your notebooks when you have finished running experiments.

**TensorFlow + Keras vs. OpenCV**\
We mainly use TensorFlow and Keras instead of OpenCV. But OpenCV might still be your go-to option to address some of the choice tasks, such as preprocessing the data of another dataset. In this case, you can create a separate program/script (in C++ or Python) that does the job and saves the output (images). While the integration of OpenCV in Python is straightforward, you don't need to do this necessarily.

**Tasks**\
Your tasks are to develop the scripts to:

1.  Import the Fashion MNIST dataset from Keras including the data labels. This would import two sets (training set and test set). Create a third set (validation set) by splitting the training set into two (training set and validation set) for validation purposes. Decide what a good ratio of training/validation is, and motivate your choice. You should use the validation set to evaluate the different choices you make when building your CNNs. Keep in mind that the test set will be used at the very final stage and will not be included in the validation step.
2.  Create **one baseline **CNN architecture and **four variants**. Each model takes as input a greyscale image of size 28x28x1 and has 10 outputs, one for each class. Each model should have 7 to 10 layers, including the input and output layer (layer information is added in the next section). The four variants should differ on exactly property from your baseline model. Choose any meaningful property, e.g., different layer type (convolution, pooling, etc.), use of dropout, activation function, number of kernels, kernel size, stride length, learning rate, etc. **Your variants should be aimed at getting a better performance. **We keep the batch sizes fixed so choose a number and keep it constant for all models. You need to motivate (1) your choice of baseline model and (2) the choice of property (how do you think the change will matter?). **Important**: change only one property of the baseline for each variant, otherwise you can't make pair-wise comparisons.
3.  Train each of your models on the **training set **and validate them on the **validation set **of Fashion MNIST. Train and validate each of the five models for up to 15 epochs. In each epoch, store your training and validation loss and accuracy. From the five trained networks, we choose the best architecture based on the validation set. Draw a training/validation loss per epoch graph for your each of your five CNNs. Here is [an example ](https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-a-Training-Dataset-the-May-be-too-Small-Relative-to-the-Validation-Dataset.png)for this graph. (Note that this is a completely random graph from the web for you to see the axes and the lines).
4.  Test your best-performing and second best-performing CNNs on the **test set**. You can now use all the original training set (training set + validation set) to train these models since you have already selected your best model setting.
5.  Report your findings (see under submission).

**Layers**

-   "7 to 10 layers" should be composed of either input/output, convolutional, pooling, and dense layers.
-   You can use any additional layers such as reshaping (flatten), regularization, normalization, activation layers but these are not counted towards the "7 to 10 layers" constraint.
-   No use of any merging, attention, recurrent or locally-connected layers.
-   An example of a 7-layer network is the following (underlined layers show the layers we count):\
    Input -> Convolutional -> BatchNormalization -> Pooling -> Dropout -> Convolutional -> ReLU Activation -> Pooling -> Flatten -> Dense -> Dense (Output)

**Report**

Reporting is important in this assignment, so pay attention to the motivation of your choices, and the discussion of your results. You can use  [this template](https://uu.blackboard.com/bbcswebdav/pid-4411485-dt-content-rid-59886861_2/xid-59886861_2) , but feel free to deviate from the suggested lengths. Your report should contain:

1.  For your baseline model a **description and motivation **of the architecture and parameters (which layers, which dimensions, how connected, etc.) (use model.summary()), and for all four variants a description of **which property differs **from the baseline model, and why this choice was made
2.  For each of the five models (1) a **graph **with the training/validation loss on the y-axis and epochs on the x-axis and (2) a **link **to your model's weights (publicly accessible).
3.  A **table **with the train/validation top-1 accuracy for all five models.
4.  A **discussion** of your results in terms of your model (e.g. complexity, type of layers, etc.). Make **pair-wise comparisons **between the four variants and the baseline model.
5.  A discussion of the differences between the two models evaluated on the **test set **in terms of architecture and comparison with validation performance. How can differences be explained?
6.  A list of the **choice tasks **you implemented. For each task, look at what additional reporting we expect.

**Submission**\
Through Blackboard, hand in the following two deliverables:

1.  A zip of your code (**no binaries, no libraries, no data/model weight files, no images**). Your scripts should be .py files, or notebooks.
2.  A report (**2-5 pages**), see above.

**Grading**

The maximum score for this assignment is 100 (grade 10). The assignment counts for 15% of the total grade. You can get 70 regular points and max 30 points for chosen tasks. Fixed tasks:

1.  Create five CNNs: 15
2.  Train and validate five models on Fashion MNIST training set: 20
3.  Test your best two models on Fashion MNIST test set: 5
4.  Reporting (architectures motivated, results correctly presented, discussion of results, etc.): 30

Choice tasks also include a reporting aspect. When applicable, choose one of your top two architectures and report the performance both with and without the novel functionality.

1.  CHOICE 1: Provide and explain a confusion matrix for the results on the test set of one of the models: 10
2.  CHOICE 2: Create and apply a function to decrease the learning rate at a 1/2 of the value every 5 epochs: 10
3.  CHOICE 3: Instead of having a fixed validation set, implement k-fold cross-validation: 10 (Note that this significantly increases the running time)
4.  CHOICE 4: Create output layers at different parts of the network for additional feedback. Show and explain some outputs of a fully trained network: 20
5.  CHOICE 5: Perform data augmentation techniques (at least 3): 10. Explain how they affect your performance. Make sure to choose appropriate ones to the data and the problem.
6.  CHOICE 6: Download [this dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small), pre-process the images so they fit your network, choose the relevant classes and report the test performance on one of your models: 30
7.  CHOICE 7: Use your creativity. Check with [Metehan](mailto:m.doyran@uu.nl) for eligibility.

**Contact**\
Should you have any questions about the assignment, post them in the Assignment channel of the INFOMCV 2023 Teams. If you need help with your code, do not post your code completely. Rather post the chunks to which the question should refer.

**Frequently Asked Questions**\
*Q: Are we allowed to use tutorials such as [this one](https://www.tensorflow.org/tutorials/keras/classification)?*\
A: You can use these tutorials as a guideline but we expect you to be able to implement all tasks yourself eventually. Make sure you understand what happens.

*Q: Am I going to be marked down for not having great results?*\
A: No. We mostly want to see that you understand the basic principles of how CNNs work and how you implemented cross-validation and testing. This is not meant to be a benchmarking assignment, but you should try your best to get decent results (around 85 to 90%). You will need to motivate differences in performance between the two top-performing models.

*Q: How large should my batch sizes be?*\
A: Standard batch sizes in literature have been: 32, 64, 128, 256 -- larger than these are mostly for significantly more complex networks. Keep in mind large batch size => large learning rates, small batch size => small learning rates. Keep the batch size fixed across all experiments.
