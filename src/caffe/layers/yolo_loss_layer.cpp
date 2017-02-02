#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  YoloLossParameter loss_param = this->layer_param_.yolo_loss_param();
  classes_ = loss_param.classes();
  side_ = loss_param.side();
  boxes_ = loss_param.boxes();
  coords_ = loss_param.coords();
  
  object_scale_ = loss_param.object_scale();
  noobject_scale_ = loss_param.noobject_scale();
  class_scale_ = loss_param.class_scale();
  coord_scale_ = loss_param.coord_scale();

  softmax_ = loss_param.softmax(); 
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), side_ * side_ * ((1 + coords_) * boxes_ + classes_))
      << "Predictions must have dimension S * S * ((1 + 4) * B + C) .";
  CHECK_EQ(bottom[1]->count(1), side_ * side_ * (1 + coords_ + classes_))
      << "Ground truth must have dimension S * S * (1 + 4 + C) .";
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}  

#ifdef CPU_ONLY
STUB_GPU(YoloLossLayer);
#endif

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);

} // namespace caffe
