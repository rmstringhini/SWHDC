# **Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing**

**Abstract**

Traditional convolutional neural networks (CNNs) face significant challenges when applied to omnidirectional images due to the non-uniform sampling inherent in equirectangular projection (ERP). This projection type leads to distortions, particularly near the poles of the ERP image, and fixed-size kernels in planar CNNs are not designed to address this issue. This paper introduces a convolutional block called Spherically-Weighted Horizontally Dilated Convolutions (SWHDC). Our block mitigates distortions during the feature extraction phase by properly weighting dilated convolutions according to the optimal support for each row in the ERP, thus enhancing the ability of a network to process omnidirectional images. We replace planar convolutions of well-known backbones with our SWHDC block and test its effectiveness in the 3D object classification task using ERP images as a case study. We considered standard benchmarks and compared the results with state-of-the-art methods that convert 3D objects to single 2D images. The results show that our SWHDC block improves the classification performance of planar CNNs when dealing with ERP images without increasing the number of parameters, outperforming peering methods.


The swhdc.py file contains the complete module proposed in Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing (https://ieeexplore.ieee.org/document/10716273).
