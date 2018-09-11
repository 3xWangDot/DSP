# DSP
The code for ECCV18 paper: [Learning Discriminative Video Representations Using Adversarial Perturbations](https://arxiv.org/pdf/1807.09380.pdf)
## How to run the Demo
Please refer the demo for generating the DSP descriptor. As DSP descriptors are from stiefel manifold, non-linear kernelized SVM should be used as the classifier. Please refer to [Lib-SVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) in terms of kernelized SVM.
Two-stream ResNet-152 feature for a subset (10 classes) of HMDB-51 split-1 is provided here: [Training Data](https://drive.google.com/open?id=1hLWDpq0v0r29s1Pd_-3xSXtLgVD9tkCT) and [Testing Data](https://drive.google.com/file/d/1wD4gv0or4nxlkm68DNYu9wvAdRAOmD7t/view?usp=sharing). Features from Spatial and Temporal stream are concatenated for each sequence, which becomes a n by d matrix ( n is the number of samples in each sequence, and d is the feature dimension which is 4096 here). 

For generating the UAP, please refer to the https://github.com/LTS4/universal. We modified their code for generating the UAP for high-level CNN features.
