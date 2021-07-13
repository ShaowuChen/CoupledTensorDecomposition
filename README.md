Code for [**Deep Convolutional Neural Network Compression via Coupled Tensor Decomposition**](https://ieeexplore.ieee.org/document/9261106)


Feel free to ask me questions, and please cite our work if it help:
```
@ARTICLE{9261106,
  author={Sun, Weize and Chen, Shaowu and Huang, Lei and So, Hing Cheung and Xie, Min},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Deep Convolutional Neural Network Compression via Coupled Tensor Decomposition}, 
  year={2021},
  volume={15},
  number={3},
  pages={603-616},
  doi={10.1109/JSTSP.2020.3038227}}
```



***
# ISTA+

## Baseline model

The Baseline model and training data is from the paper "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing".

 Run the '**Train_Code_for_ISTA_Net_plus.py**' to train the original ISTA plus net with parameter sharing and get ' **para_dict.npy** '  file which including the parameters.

## TT-compression

'**ISTA_Net_TT_compression.py**' uses the independent TT decomposition.

## NC-CTD

 '**5_25_joint_TT_compression.py**'  uses the NC-CTD algorithm.

# Resnet18

## Baseline model

Run the '**resnet18.py'** using the cifar10 dataset to get the weights parameters file '**parameter.npy**'

## Single compression

The independent TT and SVD compression methods are showed in '**resnet18_single_tt.py**' and '**resnet18_svd.py**' respectively.

## NC-PCTD

 **'algrithm1_para_npy'** includes the weights by using the algrithm1, and **'algrithm2_npy'** includes the weights by using the algrithm2.

**'joint_net.py'** uses the NC-PCTD approach.







#### 
