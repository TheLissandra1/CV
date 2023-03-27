# CV
## assignment 5

### [Assignment 5: Action recognition](https://uu.blackboard.com/webapps/assignment/uploadAssignment?content_id=_4418000_1&course_id=_140663_1&group_id=&mode=view)

**Overall goal**\
The aim of this assignment is to classify actions (1) in still images and (2) in videos with CNNs. The focus is on transfer learning, with the additional use of optical flow and combination of CNN outputs. There are no constraints in terms of the number of layers or the type of the architecture. Again, we're not looking for the best performance, although you should aim at passing the baseline accuracy scores for each task. **Make informed choices and reflect on those.** Explaining your choices and results in the report will be important. We will use TensorFlow with Keras, but you are free to use PyTorch as well. To get you started you can find the skeleton code [here](https://colab.research.google.com/drive/1a7sp8k4Zr3uc-cuDfuodSsmM1LgyhDFq?usp=sharing) that downloads the two datasets and creates lists for you to use. You should click Copy to Drive to create a copy of it and start editing.

**Data **\
We will use two datasets: 

-   **Stanford 40**: An image-based dataset for action recognition. It includes 40 action classes with 180-300 images per class (total of 9532 images). The dataset can be downloaded [[here](http://vision.stanford.edu/Datasets/40actions.html)]. Use the train/test split in the [skeleton code](https://colab.research.google.com/drive/1a7sp8k4Zr3uc-cuDfuodSsmM1LgyhDFq?usp=sharing) (*train_files* and *test_files* variables). When using a validation set, sample from the training set (10% of the training data with stratification). Make all your choices and experiments on the training set. The test set must be used only once for reporting the final performance result of your model.
-   **HMDB51: **Includes short video clips 51 classes with a total of 6849 images, approximately 100-150 per class. Videos are sourced from the web and contain a large amount of variation. Each video is a couple of seconds long. The dataset can be found [[here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview)]. We are using the prespecified training set (ID: 1) and test set (ID: 2). The videos per set can be downloaded [here](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar), with documentation [here](https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt). When using a validation set, sample from the training set (10% of the training data). The sets have an equal number of videos per class so stratification is not needed.

We will focus on the following 12 overlapping classes (HMDB51 - Stanford 40):

-   applauding - clap hands
-   climbing - climb
-   drinking - drink
-   jumping - jump
-   pouring liquid - pour
-   riding a bike - ride bike
-   riding a horse - ride horse
-   running - run
-   shooting an arrow - shoot bow
-   smoking - smoke
-   throwing frisbee - throw
-   waving hands - wave

The other actions (images and videos) can be ignored. This effectively reduces the complexity of the assignment.

**Tasks\
**

You will train four different networks, with different inputs and with different training strategies. For all tasks, select your hyperparameters on the validation set you created with stratification and report the final performance on the test set (see under "Submission"). Develop (build, train and evaluate on validation set) the following networks: 

1.  **Stanford 40 -- Frames:** Create a CNN and train it on the images in Stanford 40. Naturally, you will have 12 output classes. 
2.  **HMDB51 -- Frames (transfer learning):** Use your pretrained CNN (same architecture/weights) and fine-tune it on the middle frame of videos of the HMDB51 dataset (you can use the middle frame of each video, for example). You can use a different learning rate than for the Stanford 40 network training.
3.  **HMDB51 ****-- Optical flow:** Create a new CNN and train it on the optical flow of videos in HMBD51. You can use the middle frame (max 5 points) or stack a fixed number (e.g., 16) of optical flow frames together (max 10 points).
4.  **HMDB51**** -- Two-stream:** Finally, create a two-stream CNN with one stream for the frames and one stream for the optical flow. Use your pre-trained CNNs. Think about how to fuse the two streams and motivate this in your report. Look at the Q&A at the end of this assignment. Fine-tune the network.

**Network architecture**

You are free in developing your own network, without any constraints. Make sure you motivate your choices well. If you base yor choices on existing network architectures, provide literature references. Your networks can be as complex as you like, but bear in mind that (1) training large models is time-consuming and (2) with the modest amount of data is more prone to overfitting. You are also free to reduce the input image size (for all networks) to reduce computation time. You can experiment with various architectures and hyperparameters. Based on your validation set, select your best network. You don't need to describe all networks that didn't make the cut.

**Baselines\
**Although there is not a strict accuracy score you should get, the more informed choices you make the better accuracy scores you should get in general. We will keep a leaderboard on Teams, just for reference and to understand which choices have a positive effect. Since we have 12 classes and we use stratitifaction, the baseline is ~8%. We expect scores well above 8%, but much lower compared to the previous assignment.

**Submission**\
Through Blackboard, hand in the following two deliverables. 

1.  A zip with your **code** (no binaries, no libraries, no data/model weight files!)
2.  A **report** (5 pages max). You can use  [infomcv_assignment_5_report_template.docx](https://uu.blackboard.com/bbcswebdav/pid-4418000-dt-content-rid-60053312_2/xid-60053312_2)  if you want. Make sure you include:
    1.  A brief **description and motivation** of your frame CNN and optical flow CNN, detailing the architecture and parameter values. Also motivate how you combined the two networks into a two-stream network.
    2.  **One** **table** and **eight graphs**. In the table, include the best rates achieved on each of the datasets, the models chosen and the top-1 accuracy and loss on the train and test sets (see example below). Provide **train-validation accuracy** and **train-validation loss** graphs for: (1-2) your frame CNN on the Stanford 40 dataset, (3-4) your transfer learned frame CNN on HMDB51, (5-6) your optical flow CNN on HMDB51, and (7-8) your two-stream model on HMDB51. In all cases, report only the runs with the selected hyperparameters. All experimentation with architectures and hyperparameters doesn't have be included in the report. But make sure you motivate your final networks well (see previous point). Use different colors for different models and different color shades for train/validation.
    3.  **Explain** your results in terms of your architecture and the training procedure.
    4.  Add a **link** to your model weights (in Dropbox or Google/One Drive). Make sure m.doyran@uu.nl and r.w.poppe@uu.nl have access.
    5.  Clearly mention which **choice tasks** you implemented. Make sure you discuss how you implemented the task and, when applicable, how much improvement you gained by implementing the task.

| **Dataset** | **Model** | **Top-1 acc.** | **Top-1 loss** | **Model size (GB)** |
| Stanford40 | Frames | W% | 0.www | w.ww |
| HMDB51 | Frames | X% | 0.xxx | x.xx |
| HMDB51 | Optical flow | Y% | 0.yyy | y.yy |
| HMDB51 | Two-stream | Z% | 0.zzz | z.zz |

**Grading **\
The maximum score for this assignment is 100 (grade 10). The assignment counts for 15% of the total grade. You can get 75 regular points and max 25 points for chosen tasks: 

-   Create custom CNN and train/validate/test it on Stanford 40: 10
-   Apply transfer learning and re-train (fine-tune) your CNN on HMDB51: 10
-   Calculate the optical flow for HMDB51: 10
-   Train/validate/test a CNN using optical flow on HMDB51: 10
-   Train a Two-Stream CNN and train/validate/test it on HMDB51: 15 
-   Reporting: 20 

The options below are eligible for choice points. Check with Metehan if you have other ideas. For each task, report the change in performance for one model (frames/optical flow/two-stream).

-   CHOICE 1: Do data augmentation for Stanford-40: 5
-   CHOICE 2: Do data augmentation for HMDB51: 5  (random crop is already in the skeleton code)
-   CHOICE 3: Create a Cyclical Learning rate schedule: 5
-   CHOICE 4: Use 1x1 convolutions to connect activations between the two branches: 10
-   CHOICE 5: Systematically investigate how your results change when using other frames than the middle frame in your HMDB51 frame model: 10
-   CHOICE 6: Show class activations (e.g., using [Grad-CAM](https://keras.io/examples/vision/grad_cam/)) for two classes in the HMDB51 frame, optical flow and two-stream models: 10
-   CHOICE 7: Experiment with at least three types of cooperation for the two networks (average, mean, addition etc.): max 10
-   CHOICE 8: Present and discuss a confusion matrix for your (1) Stanford 50 Frames and (2) HMDB51 Optical flow models: 5

**Contact **\
Should you have any questions about the assignment, post them on Teams in the Assignments channel. If you need help with your code, **do not post your code entirely**. Rather post the **chunks** which the question should refer to.

**Frequently Asked Questions and Notes **\
*Q: What accuracy should I expect to get? *\
A: Depends on the architecture. Random guessing would be ~8% (100/12%). Of course, you should have scores well above that but the complexity of the images and videos is high. We will try to post intermediate leaderboards on Teams.

*Q: Am I supposed to perform data augmentation myself? *\
A: Changes in the data will be reflected on the optical flow as well. You are advised to only do color-based data augmentation. For convenience, you can use the [imgaug](https://github.com/aleju/imgaug) repo or the Keras ImageDataGenerator class.

*Q: How should I connect the two models in a two-stream network? *\
A: There are plenty of ways to fuse information from two (or more) networks. For example, the figure below is a very solid point to start from.

 ![](https://uu.blackboard.com/bbcswebdav/pid-4418000-dt-content-rid-51878880_2/xid-51878880_2)

Figure 1. Examples of fusion locations in Two-Stream CNNs.

*Q: Do I need to re-calculate the flow if I do data augmentation with cropping? *\
A: Yes, because the image will be altered. You could simply perform the same type of cropping that you applied in your image to the flow image as well.

*Q: How can we have two sources for input to our network?*\
A: Check Keras Functional API.

*Q: What is the recommended image size?*\
A: Either 112x112 or 224x224 should work quite well. You can also decrease the image size if you either want to increase the batch size (if the data don't fit in the GPU) or if the model underfits the data.
