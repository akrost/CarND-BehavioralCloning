# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_cnn]: ./examples/images/nvidia-cnn-architecture.png "CNN architecture by NVIDIA"
[center_lane]: ./examples/images/center_lane_driving.jpg "Center lane driving"
[recover]: ./examples/gifs/recover.gif "Driving maneuver recovering from left side"
[preflip]: ./examples/images/pre-flip.jpg "Original image"
[postflip]: ./examples/images/post-flip.jpg "Flipped image"
[left]: ./examples/images/left.jpg "Image taken by the left camera"
[center]: ./examples/images/center.jpg "Image taken by the center camera"
[right]: ./examples/images/right.jpg "Image taken by the right camera"
[track1]: ./examples/gifs/track1.gif "Performance on track 1"
[track2]: ./examples/gifs/track2.gif "Performance on track 2"

---
**Requirements**

* [Anaconda 3](https://www.anaconda.com/download/) is installed on your machine.
* Download the simulator:
  *  [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
  * [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
  * [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
* Download the udacity [data set] (optional)(https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

---
## **Getting started**

1. Clone repository:<br/>
```sh
git clone https://github.com/akrost/CarND-BehavioralCloning.git
cd carnd-behavioralcloning
```

2. Create and activate Anaconda environment:
```sh
conda create --name carnd-p3
source activate carnd-p3
```
Activating the environment may vary for your OS.

3. Install packages:
```sh
pip install -r requirements.txt
```

4. 
   1. To run the project, first start the script
    ```sh
    python drive.py model.h5
    ```
   2. Start the simulator in autonomous mode
  

## **Project**

### Model Architecture and Training Strategy

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### 1. Model architecture

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 178-184) 

The model includes ReLU layers to introduce nonlinearity (code line 178-189). The model uses Keras Lambda layers to transform the color space of the input image from RGB to HSV (code line 175) and to normalize the data (code line 176). 

#### 2. Reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 174 et seqq). The dropout layers are applied after both convolutional and fully connected layers. The dropout rate is different, though (conv: 0.2, fc: 0.5).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22-59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 195).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used multiple different training data sets:
* Center lane driving on both tracks in both directions (model.py line 22)
* Special driving maneuvers (recovery, etc.) (model.py line 28)
* Udacity default data set (model.py line 36)
* Center lane driving track 1, using an analog joystick (model.py line 43)
* Center lane driving track 2, using an analog joystick (model.py line 49)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Final Model Architecture

Various different CNNs were tested. The final model architecture (model.py lines 174-193) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Description | Param# |
|---|---|--:|
| Input | 160x320x3 RGB image | 0 |
| Lambda: Color representation conv| Converts RGB image to HSV | 0 |
| Lambda: Normalization| Normalizes pixel values | 0 |
| Cropping2D | Input size 160x320x3, output size 75x320x3 | 0 |
|Conv2D | 2x2 strides, 5x5 kernel, SAME padding, outputs 36x158x24, ReLU, dropout with .8 keep rate| 1,824 |
|Conv2D | 2x2 strides, 5x5 kernel, SAME padding, outputs 16x77x36, ReLU, dropout with .8 keep rate | 21,636 |                         |
|Conv2D | 2x2 strides, 5x5 kernel, SAME padding, outputs 6x37x48, ReLU, dropout with .8 keep rate | 43,248 |
|Conv2D| 3x3 kernel, SAME padding, outputs 4x35x64, ReLU, dropout with .8 keep rate | 27,712 |
| Flatten | 8,960 nodes | 0 |
| Dense | 8,960 inputs, 100 outputs, ReLU, dropout with  .5 keep rate | 896,100 |
| Dense | 100 inputs, 50 outputs, ReLU | 5,050 |
| Dense | 50 inputs, 1 output, linear activation | 51 |

In total the model has  995,621 parameters.


The CNN used in this project was the network architecture [published by NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/):

![CNN architecture by NVIDIA][nvidia_cnn]

There are two differences between the NVIDIA architecture and the model shown in the table above:
* NVIDIA used images in YUV representation to feed into the CNN, the simulator used in this project only outputs RGB images. It was found, that transforming the RGB images to HSV representation improved the performance of the network.
* The input size of the images used by NVIDIA was 66x200x3. In the project the input size of the images was 160x320x3. Later the images were cropped to a size of 60x320x3. NVIDIA did not use any cropping, but they didn't mention whether the image was already cropped. 

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on each track using center lane driving. Here is an example image of center lane driving:

![Center lane driving][center_lane]

After that I recorded another lap on each track using center lane driving, but this time I drove in the opposite direction, to avoid the model from getting biased to certain curves. I also recorded some laps using an analog joystick to get better quality input.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to deal with situations that differ from the regular driving situation. This images shows what a recovery looks like starting from the left side of the lane:

![Driving maneuver recovering from left side][recover]


To augment the data set, I also flipped images and steering angles thinking that this would improve the model since with only one curve in the data set it would also see a similar mirrored curve. For example, here is an image that has then been flipped:

![Original image][preflip]
![Flipped image][postflip]

I also made use of the "left" and "right" image that the simulator provides. It is an image taken from a camera that is not in the center of the wind shield. Instead it is a little off center to the left or the right. This way, even though the car drives on the center of the road, the image looks like the car is off center. Using those images, the steering angle has to be adjusted as well. I used a parameter to adjust the steering angle. Here are three images, a left a center and a right image of the same point in time:

![Left image][left]
![Center image][center]
![Right image][right]

After the collection process, I had 21,596 number of data points. Using left, center and right images I had a 64,788 images. By flipping all of them I doubled the input data set to a total amount of 129,576 images.

I used a generator function (line 69, model.py) to generate data in batches. It will return six times the amount of images than what the parameter "batch_size" says. So if the batch_size is 10, the function will return 60 data points (1 original and 5 augmented).

The data the generator returns is randomly shuffled.
I put 20 % of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. After very few epochs the model was not able to improve very much anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 3. Model performance

With the architecture and the training data described above, the model is able to perform decently:

![Performance on track 1][track1]
![Performance on track 2][track2]

