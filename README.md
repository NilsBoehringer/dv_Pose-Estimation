# Hand Segmentation and Handpose Estimation
#### Deep Vision SS2019 Project

###### Setup:

 - Clone the repository
 - Download the Rendered Handpose Dataset used for this project [here](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html "Rendered HandPose Dataset")
 - Unzip the dataset in the root folder of the git repository
 - Rename the folder *RHD_published_v2* to *data*

###### How to Use
The project contains two networks:
 - HandSegNet: A Unet to segment the input image into hand and not-hand
 - PoseNet: A Unet to estimate 21 keypoints in the hand


Each Neural Network has 4 files:
 - process.*: The python script to prepare the dataset for training
 - train.*: The python script to train the network
 - eval.*: The python script to evaluate the performance
 - checkpoint_*.pth.tar: The saved state of the Neural network. Used for evauation.


###### Info:
The two Jupyter Notebook files included in the repository are there to substitute if
one of the *HandSegNet* files fails.
