#include "caffe/deep_landmark_layers.hpp"

namespace caffe {
  
template <typename Dtype>
__global__ void ForwardSqrt(int n, const Dtype* in, Dtype* out) {
  // f(x) = sqrt(x)
  CUDA_KERNEL_LOOP(index, n){
    Dtype val = in[index];
    out[index] = sqrt(val);
  }
}

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
  // bottom[0] stores ground truth of the shape (batch, num_landmark, 2, 1)
  // bottom[1] stores predictions of the shape (batch, num_landmark, 2, 1)
  // diff_x_ = x - x'
  // diff_y_ = y - y'
  for (int b = 0; b < bottom[0]->num(); ++b){    
    for (int n = 0; n < num_landmark_; ++n){
      diff_x_.mutable_cpu_data()[diff_x_.offset(b) + n] = 
          bottom[0]->cpu_data()[bottom[0]->offset(b, n)] - 
          bottom[1]->cpu_data()[bottom[1]->offset(b, n)];
      
      diff_y_.mutable_cpu_data()[diff_y_.offset(b) + n] = 
          bottom[0]->cpu_data()[bottom[0]->offset(b, n, 1)] -
          bottom[1]->cpu_data()[bottom[1]->offset(b, n, 1)];
    }
  }
  Blob<Dtype> diff_x2, diff_y2;
  diff_x2.Reshape(diff_x_.shape());
  diff_y2.Reshape(diff_x_.shape());
  // diff_x2 = (x - x')^2
  // diff_y2 = (y - y')^2
  int count = diff_x_.count();
  caffe_gpu_mul(
    count,
    diff_x_.gpu_data(),
    diff_x_.gpu_data(),
    diff_x2.mutable_gpu_data());

  caffe_gpu_mul(
    count,
    diff_y_.gpu_data(),
    diff_y_.gpu_data(),
    diff_y2.mutable_gpu_data());
  
  // sum_ = (x - x')^2 + (y - y')^2
  caffe_gpu_add(
    count,
    diff_x2.gpu_data(),
    diff_y2.gpu_data(),
    sum_.mutable_gpu_data());
  
  // sum_ = sqrt((x - x')^2 + (y - y')^2)
  ForwardSqrt<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, sum_.gpu_data(), sum_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  Dtype sum_error = Dtype(0);

  for (int n = 0; n < num_landmark_; ++n) {
    for (int b = 0; b < sum_.num(); ++b) {
      sum_error += sum_.cpu_data()[sum_.offset(b, n)];
    }
    
  }
  top[0]->mutable_cpu_data()[0] = sum_error / sum_.num();
}

template <typename Dtype>
void L2DistanceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                         const vector<bool>& propagate_down,
                                                         const vector<Blob<Dtype>*>& bottom) {
   // backward
   Blob<Dtype> deriative_x,deriative_y, diff;
   deriative_x.Reshape(diff_x_.shape());
   deriative_y.Reshape(diff_x_.shape());
   diff.Reshape(bottom[0]->shape());
   int count = diff_x_.count();
   caffe_gpu_div(
      count,
      diff_x_.gpu_data(),
      sum_.gpu_data(),
      deriative_x.mutable_gpu_data());
   caffe_gpu_div(
      count,
      diff_y_.gpu_data(),
      sum_.gpu_data(),
      deriative_y.mutable_gpu_data());
   for(int b=0;b<bottom[0]->num();b++){
      for(int n=0;n<num_landmark_;n++){
         diff.mutable_cpu_data()[diff.offset(b,n)] = deriative_x.cpu_data()[deriative_x.offset(b,n)];
         diff.mutable_cpu_data()[diff.offset(b,n,1)] = deriative_y.cpu_data()[deriative_y.offset(b,n)];
      }
   }
   int i;
   for(i=0;i < 2;i++){
     int sign = (i == 0) ? 1 : -1;
     const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
     caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
   }

   
}



INSTANTIATE_LAYER_GPU_FUNCS(L2DistanceLossLayer);
} // namespace caffe
