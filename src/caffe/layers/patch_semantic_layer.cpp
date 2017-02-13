#include "caffe/deep_landmark_layers.hpp"

namespace caffe{

template <typename Dtype>
void PatchSemanticLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
  PatchSemanticParameter ps_param = this->layer_param_.patch_semantic_param();
  num_landmark_ = ps_param.num_landmark();
  image_.first = ps_param.image_w();
  image_.second = ps_param.image_h();
  patch_.first = ps_param.patch_w();
  patch_.second = ps_param.patch_h();
  
  // push back all spatial operations in an order
  // by spatial operation we mean the meta data of "Convolution" or "Pooling" layer
  int num_spatial_operations = ps_param.spatial_operation_param_size();
  int count = 0;
  arr = new SpatialOperation<Dtype>[num_spatial_operations];
  for (int i = 0; i < num_spatial_operations; ++i) {
    SpatialOperationParameter sp_param = ps_param.spatial_operation_param(i);
    SpatialOperation<Dtype> sp(sp_param);
    if (sp_param.used()) {
      count += 1;
      index_sp_used_vec_.push_back(i);
    }
    arr[i] = sp;
    sp_vec_.push_back(sp);
    
    
  }
 
  for (int i = 0; i < num_spatial_operations; ++i) {
     sp_vec_[i] = arr[i];
  }

  // bottom[0] and bottom[1] are ground truth and prediction for landmark, respectively.
  // and other bottom(s) are used to calculate partial derivatives 
  CHECK_EQ(bottom.size() - 2, count) 
      << "The number of blobs to calculate the three partial derivatives must equal the number of the used spatial operation.";
  // following these spatial operations one by one
  std::cout << sp_vec_[0].Weight() << std::endl;
  std::cout << sp_vec_[1].Weight() << std::endl;

  for (int i = 1; i < sp_vec_.size(); ++i) {
    sp_vec_.at(i).Following(sp_vec_.at(i - 1)); 
  }
  // for a certain layer, we compute a corresponding partial derivative for each Fx, Fy or Ft 
  if (ps_param.cross_channel() == PatchSemanticParameter_CrossChannelMethod_MAX) {
    cross_channel_max_ = true;
    cross_channel_ave_ = false;
  } else {
    cross_channel_ave_ = true;
    cross_channel_max_ = false;
  }
  
}

template <typename Dtype>
void PatchSemanticLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
  CHECK_EQ(bottom[0]->count(1), 2 * num_landmark_) 
      << "Current landmark localization must be of dimension 2 * num_landmark";
  CHECK_EQ(bottom[1]->count(1), 2 * num_landmark_)
      << "Current landmark shift must be of dimension 2 * num_landmark";
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_, 2, 1);
  bottom[1]->Reshape(bottom[1]->num(), num_landmark_, 2, 1);
  
  // top[0] stores a triple (Fx, Fy, Ft) for a landmark
  top[0]->Reshape(bottom[0]->num(), num_landmark_, 3, 1); 
  
}

template <typename Dtype>
void PatchSemanticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
  NOT_IMPLEMENTED;
} 
 
template <typename Dtype>
void PatchSemanticLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& down){
  NOT_IMPLEMENTED;
}
     
#ifdef CPU_ONLY
STUB_GPU(PatchSemanticLayer);
#endif

INSTANTIATE_CLASS(PatchSemanticLayer);
REGISTER_LAYER_CLASS(PatchSemantic);

} // namespace caffe
