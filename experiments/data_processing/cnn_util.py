import caffe
import numpy as np
from util import log
import sys

class CNN(object):
    """
    Convolutional neural networks for evaluation.
    """

    def __init__(self, net, model, mode="GPU", gpu_id=0):
        self.net = net
        self.model = model
        self.mode = mode
        self.gpu_id = gpu_id
        
        if self.mode == "GPU":
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            caffe.set_mode_cpu()
        # get specified CNN
        try:
            self.cnn = caffe.Net(str(self.net), str(self.model), caffe.TEST)
        except:
            log("Can not open [%s, %s], plz check if these two files exist or not." % (self.net, self.model))
            sys.exit(1)

    def forward(self, data, input_layer='data', output_layer='fc8', reshape=True):
        """
        Do a CNN forward and return a specified result.

        param:
        -data, 1 * C * H * W
        """
        # set input data
        self.cnn.blobs[input_layer].data[...] = data.astype(np.float32)
        self.cnn.forward()
        result = self.cnn.blobs[output_layer].data[0]
        # 2N vector -> (N, 2)
        if reshape:
            r = lambda x: np.asarray([np.asarray([x[2 * i], x[2 * i + 1]]) for i in range(len(x) / 2)])
            result = r(result)
        return result
