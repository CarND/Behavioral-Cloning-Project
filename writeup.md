# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

[image1]: ./examples/model_mean_squared_error_loss_graph.png "Model Mean Squared error Loss Graph"

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* fun_run.mp4 contains the video of the car being driven by model.h5
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used to train the self-driving car is an exact copy of NVIDIA's pipeline and the keras code used to create that pipeline was found and copied from the internet.

The network consists of 10 layers, including a cropping layer, a normalization layer, 5 convolutional layers and 3 fully connected layers. 

The first layer of the network crops the image to remove pixels above the horizon, allowing the training to be faster and learning to focus on the road data.

The second layer of the network performs image normalization, which is hard-coded and is not adjusted in the learning process. 

The five convolutional layers are designed to perform feature extraction. The
first three convolutional layers have a 2×2 stride and a 5×5 kernel whereas the last two convolutional layers are non-strided convolutions with a 3×3 kernel size.

The five convolutional layers are followed by three fully connected layers leading to an output control value which is the inverse turning radius - our goal. 

The fully connected layers are designed to function as a controller for steering. 
Nvidia notes that by training the system end-to-end, it is not possible to make a 
clean break between which parts of the network function primarily as feature 
extractor and which serve as controller.

#### 2. Attempts to reduce overfitting in the model

There were attempts to reduce overfitting in our model by adding dropout layers but that led to our car running off the road.  So instead of adding dropout layers to combat overfitting, the model was trained for less epochs, 3 in fact, since it was observed that training accuracy but not validation accuracy improved around epoch 4 and beyond, an indication of overfitting.  

#### 3. Model parameter tuning

The model used an adamax optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was prepared by using the provided data and incorporating all of the data from the left, right, and center cameras.  I augmented the data by flipping every datapoint (image and steer angle) and that seemed to be enough data to train the car to complete the easy track without having to collect more driving data. However, the car failed the more challenging track, which leads me to believe that it may be required to collect extra driving data to improve our model's ability to drive in more challenging terrain.   
### Model Architecture and Training Strategy

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================        
cropping2d (Cropping2D)      (None, 65, 320, 3)        0
_________________________________________________________________        
lambda (Lambda)              (None, 65, 320, 3)        0
_________________________________________________________________        
conv2d (Conv2D)              (None, 31, 158, 24)       1824
_________________________________________________________________        
conv2d_1 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________        
conv2d_2 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________        
conv2d_3 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________        
conv2d_4 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________        
flatten (Flatten)            (None, 2112)              0
_________________________________________________________________        
dense (Dense)                (None, 100)               211300
_________________________________________________________________        
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________        
dense_2 (Dense)              (None, 10)                510
_________________________________________________________________        
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

