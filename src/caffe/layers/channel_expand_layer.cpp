#include "caffe/deep_landmark_layers.hpp"

namespace caffe{

template <typename Dtype>
void ChannelExpandLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
  ChannelExpandParameter ce_param = this->layer_param_.channel_expand_param();
  num_channel_ = ce_param.num_channel();
}

template <typename Dtype>
void ChannelExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
  CHECK_EQ(bottom[0]->channels(), 1) 
      << "Input blob must be single channel blob";
  bottom[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  
  // top[0] stores the expanded channel of bottom[0]
  top[0]->Reshape(bottom[0]->num(), num_channel_, bottom[0]->height(), bottom[0]->width()); 
  
}

template <typename Dtype>
void ChannelExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
  NOT_IMPLEMENTED;
} 
 
template <typename Dtype>
void ChannelExpandLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& down){
  NOT_IMPLEMENTED;
}
     
#ifdef CPU_ONLY
STUB_GPU(ChannelExpandLayer);
#endif

INSTANTIATE_CLASS(ChannelExpandLayer);
REGISTER_LAYER_CLASS(ChannelExpand);

} // namespace caffe
