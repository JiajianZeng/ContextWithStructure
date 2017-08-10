# ContextWithStructure

ContextWithStructure is a deep learning based facial landmark detection framework, which jointly considers the context as well as the intrinsic geometric structure of facial landmarks. This project hosts the full source code for our paper **Deep Context-Sensitive Facial landmark Detection with Tree-Structured Modeling** (under review of TIP). 

# Overview of source code 

This project utilizes convolutional neural network for facial landmark detection and is written under [Caffe](https://github.com/BVLC/caffe) framework. To be specific, we develop training related code in C++ via the **Layer** interface provided by Caffe and write data processing, evaluation code in Python, this part of Python code is in the **experiments** sub-directory.


