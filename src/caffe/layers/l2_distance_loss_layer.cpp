#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  L2DistanceLossParameter param = this->layer_param_.l2_distance_loss_param();
  num_landmark_ = param.num_landmark();
  error_threshold_ = param.error_threshold();
}

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), 2 * num_landmark_)
      << "Landmark predictions must have dimension num_landmark * 2 .";
  CHECK_EQ(bottom[1]->count(1), 2 * num_landmark_)
<<<<<<< HEAD
      << "Predictions must have dimension num_landmark * 2 .";
=======
      << "Landmark ground truth must have dimension num_landmark * 2 .";
  CHECK_EQ(bottom[2]->count(1), 1)
      << "Each image must have one normalizer (bi-ocular distance or width of facial bounding box).";
>>>>>>> 6c25c223680f5ec93f7cd50019196b13a8192bd5
  // reshape bottom[0] to (batch, num_landmark, 2, 1)
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_, 2, 1);
  // reshape bottom[1] to (batch, num_landmark, 2, 1)
  bottom[1]->Reshape(bottom[1]->num(), num_landmark_, 2, 1);
  // reshape bottom[2] to (batch, 1, 1, 1);
  
  // top[0] stores the l2distance error for landmarks 
  top[0]->Reshape(1, 1, 1, 1);
  diff_x_.Reshape(bottom[0]->num(), num_landmark_, 1, 1);
  diff_y_.Reshape(diff_x_.shape());
  sum_.Reshape(diff_x_.shape());
  
}

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(L2DistanceLossLayer);
#endif

INSTANTIATE_CLASS(L2DistanceLossLayer);
REGISTER_LAYER_CLASS(L2DistanceLoss);

} // namespace caffe
