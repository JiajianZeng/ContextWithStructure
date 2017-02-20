#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void ChannelExpandLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  for (int b = 0; b < bottom[0]->num(); ++b){    
    for (int n = 0; n < scale_; ++n){
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


template <typename Dtype>
void PrintBlob(const Blob<Dtype> &blob, string name){
  std::cout << "#################################" << std::endl;
  std::cout << "name:" << name << std::endl;
  std::cout << "num:" << blob.num() << std::endl;
  std::cout << "channel:" << blob.channels() << std::endl;
  std::cout << "height:" << blob.height() << std::endl;
  std::cout << "width:" << blob.width() << std::endl;
  std::cout << "data:" << std::endl;
  std::cout << "[";
  for (int n = 0; n < blob.num(); ++n){
    for (int c = 0; c < blob.channels(); ++c){
      for (int h = 0; h < blob.height(); ++h){
        for (int w = 0; w < blob.width(); ++w){
          std::cout << blob.data_at(n,c,h,w) << ",";
        }
      }
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "#################################" << std::endl;
   
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelExpandLayer);
} // namespace caffe
