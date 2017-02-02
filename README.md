# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

## Layers related to our landmark project

There exists three layers related to our landmark project: yolo_loss_layer, perceptual_loss_layer and patch_semantic_layer. I finished the first two layers at present (source codes for patch_semantic_layer is also finished), both source codes under 'src/caffe/layers' directory and test codes (test whether Forward and Backward are correct) under 'src/caffe/test' directory. And the only one header file is 'deep_landmark_layers.hpp' under 'include/caffe' directory. 

The reference paper for yolo_loss_layer is: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
The reference paper for perceptual_loss_layer is: [UNSUPERVISED CONVOLUTIONAL NEURAL NETWORKS FOR MOTION ESTIMATION](https://arxiv.org/abs/1601.06087) and [Determining optical flow](http://image.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf). For the paper [Determining optical flow](http://image.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf), you just need to figure out the computation of the three partial derivatives (this is analogously what patch_semantic_layer does). If you got any problems, feel free to let me know.
