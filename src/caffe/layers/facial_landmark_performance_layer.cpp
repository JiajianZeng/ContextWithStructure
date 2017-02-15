#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void FacialLandmarkPerformanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
  FacialLandmarkPerformanceParameter param = this->layer_param_.facial_landmark_performance_param();
  num_landmark_ = param.num_landmark();
  error_threshold_ = param.error_threshold();
}

template <typename Dtype>
void FacialLandmarkPerformanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), 2 * num_landmark_)
      << "Ground truth must have dimension num_landmark * 2 .";
  CHECK_EQ(bottom[1]->count(1), 2 * num_landmark_)
      << "Predictions must have dimension num_landmark * 2 .";
  CHECK_EQ(bottom[2]->count(1), 1)
      << "Each image must have one normalizer (bi-ocular distance or width of facial bounding box).";
  // reshape bottom[0] to (batch, num_landmark, 2, 1)
  bottom[0]->Reshape(bottom[0]->num(), num_landmark_, 2, 1);
  // reshape bottom[1] to (batch, num_landmark, 2, 1)
  bottom[1]->Reshape(bottom[1]->num(), num_landmark_, 2, 1);
  // reshape bottom[2] to (batch, 1, 1, 1);
  bottom[2]->Reshape(bottom[2]->num(), 1, 1, 1);
  
  // top[0] stores one average error for each landmark 
  top[0]->Reshape(num_landmark_, 1, 1, 1);
  // top[1] stores one false rate for each landmark
  top[1]->Reshape(num_landmark_, 1, 1, 1);
  
}

template <typename Dtype>
void FacialLandmarkPerformanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FacialLandmarkPerformanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FacialLandmarkPerformanceLayer);
#endif

INSTANTIATE_CLASS(FacialLandmarkPerformanceLayer);
REGISTER_LAYER_CLASS(FacialLandmarkPerformance);

} // namespace caffe
