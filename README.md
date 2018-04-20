## Project: Build a german traffic sign classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

In this project ,I have built a traffic sign classifier for german traffic signs.The dataset that was used consists of 39000 and 12500 traini and and test images respectively.All the images are in RGB colorspace with the dimensions 32x32x3.I used the very famous Lenet neural network architecture as baseline to modify my own network which is somewhat different from Lenet.The accuracy on validation set is 99.4% and 94.6% on test set correspondingly.On the other random images downloaded from the web,the accuracy came out to be 100%.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset

1. [Download the dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads).
2. Clone the project and start the notebook.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
3. Follow the instructions in the `Traffic_Sign_Recognition.ipynb` notebook.
