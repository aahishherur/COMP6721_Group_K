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
4. Torchsummary/Pthflops : ```pip install torchsummary/pip install pthflops```, Used for calculating a models's learnable parameters and flops respectively

# Training/Validating Procedure
The code makes use of a single folder consisting of the various image multiclasses.
![image](https://user-images.githubusercontent.com/52701687/206001948-81cc9b38-728c-47f7-a612-67cfff575dd3.png)

The path to these folders can be modified with the train_path list, in conjuction with modifiying the path for storing the trained models and their graphical results.
```python
train_paths=['D:/Datasets/Natural-Faces/train/','D:/Datasets/Tiny/train/','D:/Datasets/FerMasked/train/']
mdl_path='D:/Datasets/Models/'
img_path='D:/Datasets/Images/'
```
The default parameters can be changed as per the system's compuatational capability and project requirements
```python
batch_size=32
image_size=(224,224)
num_epochs=100
learning_rate=0.001
weight_dec=0.001
```
Define the models within the mdls list, with ```weights=True``` for the purposes of trasnfer Learning. 
The respective labels to be displayed within the results can also be altered.
```python
mdls=[torchvision.models.shufflenet_v2_x0_5(),torchvision.models.mnasnet0_5(),torchvision.models.mobilenet_v2(weights=True)]
dtaname=['Natural-Faces','Tiny','Fer-Masked']
mdlname=['Shufflenet','MnasNet','MobileNetV2_TL'] 
```
Run all the cells to initiate the training process.
![image](https://user-images.githubusercontent.com/52701687/206004048-dae4617f-dfbe-45fa-b479-d3f6743e1770.png)

# Testing on a Sample Dataset
A total of 3 models trained on the Fer-Masked Dataset have been provided within the 'Models' Section.
The Sample Dataset of 500 randomly selected images is available within the 'Sample Set' Section.

The following code can thus be used to test it on any of the models preferred.

