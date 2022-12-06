# AI_Project_Group_K
Artificial Intelligence (COMP 6721) Fall 2022 Project Group K

# Project Description
The following project aims to perform a multiclass image classification using Convolutional Neural Networks(CNN), being applied to facial recognition database of images imported from Kaggle. The code is purely written in Jupyter notebook through the use of existing pytorch libraries. The final results are available in the 'Later Trainings' Section with certain pre-trained models available for download via the 'Models' Section. A set of 3 models namely, ShuffleNetV2, MnasNet and MobilenetV2 were chosen for the purposes of the project. The default parameters for the training are as follows:

Optimizer: Adam  
Learning Rate: 0.001  
Batch Size: 32  
No_of_Epochs: 100  
Image Size: 224*224 pixels  
Loss Function: Cross Entropy  
Decaying LR: Step LR with step size=20 and gamma=0.5

# Code Requirements
Jupyter Notebook can be installed using ```pip install notebook``` and then executed using ```run notebook```
The following libraries need to be imported afterwards:
1. Pytorch : ```pip install torch```, Used for importing all the various functions
2. Torchvision : ```pip install torchvision```, Used for executing the dataloader commands
3. Matplotlib : ```pip install matplotlib``` , Used for displaying and storing the results in the form of graphs
4. 
