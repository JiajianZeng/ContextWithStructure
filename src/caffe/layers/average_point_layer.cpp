#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void AveragePointLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
  AveragePointParameter param = this->layer_param_.average_point_param();
  num_landmark_ = param.num_landmark();
}

template <typename Dtype>
void AveragePointLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), 2 * num_landmark_)
      << "Landmark input must have dimension num_landmark * 2 .";

  // reshape bottom[0] to (batch, num_landmark, 2, 1)
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_, 2, 1);

  
  // top[0] stores the average of all landmarks 
  top[0]->Reshape(bottom[0]->num(), 2, 1, 1);
  
}

template <typename Dtype>
void AveragePointLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void AveragePointLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(AveragePointLayer);
#endif

INSTANTIATE_CLASS(AveragePointLayer);
REGISTER_LAYER_CLASS(AveragePoint);

} // namespace caffe
