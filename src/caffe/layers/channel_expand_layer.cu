#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void ChannelExpandLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  for (int b = 0; b < bottom[0]->num(); ++b){    
    for (int n = 0; n < num_channel_; ++n){
      for(int h = 0; h < bottom[0]->height(); ++h){
        for(int w = 0; w < bottom[0]->width(); ++w){
          top[0]->mutable_cpu_data()[top[0]->offset(b,n,h,w)] = bottom[0]->cpu_data()[bottom[0]->offset(b,0,h,w)];
        }
      }
    }
  }
  
 
}

template <typename Dtype>
void ChannelExpandLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
   // backward do nothing
}




INSTANTIATE_LAYER_GPU_FUNCS(ChannelExpandLayer);
} // namespace caffe
