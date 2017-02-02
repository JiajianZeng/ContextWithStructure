#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void PerceptualLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  PerceptualLossParameter loss_param = this->layer_param_.perceptual_loss_param();

  num_landmark_ = loss_param.num_landmark();
  epsilon_ = loss_param.epsilon();

}

template <typename Dtype>
void PerceptualLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), 2 * num_landmark_)
      << "Predictions must have dimension num_landmark * 2 .";
  CHECK_EQ(bottom[1]->count(1), 3 * num_landmark_)
      << "Ground truth must have dimension num_landmark * 3 .";
  // reshape bottom[0] to (batch, num_landmark, 2, 1)
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_, 2, 1);
  // reshape bottom[1] to (batch, num_landmark, 3, 1)
  bottom[1]->Reshape(bottom[1]->num(), num_landmark_, 3, 1);
  
  derivative_x_.Reshape(bottom[0]->num(), num_landmark_, 1, 1);
  derivative_y_.Reshape(derivative_x_.shape());
  derivative_t_.Reshape(derivative_x_.shape());
  ones_.Reshape(derivative_x_.shape());
  sum_.Reshape(derivative_x_.shape());
  errors_.Reshape(derivative_x_.shape());
  
  for (int i = 0; i < ones_.count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
  
}

template <typename Dtype>
void PerceptualLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PerceptualLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(PerceptualLossLayer);
#endif

INSTANTIATE_CLASS(PerceptualLossLayer);
REGISTER_LAYER_CLASS(PerceptualLoss);

} // namespace caffe
