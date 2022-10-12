# PSNet:Parallel symmetric network for RGB-T salient object detection
![image](figs/overall.jpg)
    
Figure.1 The overall architecture of the proposed PSNet model.

# 1.Requirements
Python 3.7, Pytorch 0.4.0+, Cuda 10.0, TensorboardX 2.0, opencv-python

# 2.Data Preparation
Download the raw data from . Then put them under the following directory:
    -Dataset\ 
       -train\  
       -test\
       -test_in_train\
       
# 3.Training & Testing
**Training the PSNet**
python Train.py

**Testing the PSNet**
python Test.py
Then the test maps will be saved to './Salmaps/'

**Evaluate the result maps**
You can evaluate the result maps using the tool in





Test maps of vgg16 and vgg19 backbones of our model can be download from [here](https://pan.baidu.com/s/1aVHjW0WdIDIDvbHeC1ypNg) [code:NEPU] 
