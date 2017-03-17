#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void StructualLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  StructualLossParameter loss_param = this->layer_param_.structual_loss_param();

  w0_ = loss_param.w0();
  w1_ = loss_param.w1();
  w2_ = loss_param.w2();
  w3_ = loss_param.w3();
  num_landmark_ = loss_param.num_landmark();
  
}

template <typename Dtype>
void StructualLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Two input blobs must have equal dimension";

  // reshape bottom[0] to (batch, num_landmark + 1, 2, 1)
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_ + 1, 2, 1);
  // reshape bottom[1] to (batch, num_landmark + 1, 2, 1)
  bottom[1]->Reshape(bottom[1]->num(), num_landmark_ + 1, 2, 1);
  
  diff_x_pre_.Reshape(bottom[0]->num(),num_landmark_,1,1);
  diff_y_pre_.Reshape(bottom[1]->num(),num_landmark_,1,1);
  diff_x_gt_.Reshape(diff_x_pre_.shape());
  diff_y_gt_.Reshape(diff_x_pre_.shape());
  diff_.Reshape(bottom[0]->num(),num_landmark_,1,1);
 
 
}

template <typename Dtype>
void StructualLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void StructualLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(StructualLossLayer);
#endif

INSTANTIATE_CLASS(StructualLossLayer);
REGISTER_LAYER_CLASS(StructualLoss);

} // namespace caffe
