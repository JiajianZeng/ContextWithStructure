#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
void AveragePointLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {

  for (int b = 0; b < bottom[0]->num(); ++b){ 
    Dtype sumx = Dtype(0);
    Dtype sumy = Dtype(0);   
    for (int n = 0; n < num_landmark_; ++n){
      sumx = sumx + bottom[0]->cpu_data()[bottom[0]->offset(b, n)];
      sumy = sumy + bottom[0]->cpu_data()[bottom[0]->offset(b, n, 1)];
    }
    top[0]->mutable_cpu_data()[top[0]->offset(b, 0)] = sumx / num_landmark_;
    top[0]->mutable_cpu_data()[top[0]->offset(b, 1)] = sumy / num_landmark_;
  }
  
}

template <typename Dtype>
void AveragePointLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]){
    for (int b = 0; b < bottom[0]->num(); ++b){    
      const Dtype alphax = top[0]->cpu_diff()[top[0]->offset(b, 0)];
      const Dtype alphay = top[0]->cpu_diff()[top[0]->offset(b, 1)];
      for (int n = 0; n < num_landmark_; ++n){
        bottom[0]->mutable_cpu_diff()[bottom[0]->offset(b, n)] = alphax/num_landmark_;
        bottom[0]->mutable_cpu_diff()[bottom[0]->offset(b, n, 1)] = alphay/num_landmark_;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AveragePointLayer);
} // namespace caffe
