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

# 4.Results
# 4.1 Qualitative Comparison  
![image](figs/Qualitative_comparison.jpg)  
Figure.2 Qualitative comparison of our proposed method with ﬁfteen SOTA methods. 

![image](figs/results.jpg)  
Table.1 Quantitative comparison with ﬁfteen SOTA models on three public RGB-T benchmark datasets.  

# 4.3 Download  
The salmaps of the above datasets (Res2Net backbone) can be download from [here](https://pan.baidu.com/s/1u5kDvmTsSAQHuHk077Beng) [code:NEPU]  
The salmaps of the above datasets (VGG16 backbone) can be download from [here](https://pan.baidu.com/s/11ibTBy0VUE17Lp5FxsNd9w) [code:NEPU]  
The salmaps of the above datasets (Swintransformer) can be download from [hrer](https://pan.baidu.com/s/1_uY9a8cEBfPoIRAZD7Xvwg) [code:NEPU]
 
