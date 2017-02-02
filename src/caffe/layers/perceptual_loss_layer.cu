#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardPerceptual(int n, const Dtype* in, Dtype* out, Dtype epsilon) {
  // f(x) = sqrt(x * x + epsilon)
  CUDA_KERNEL_LOOP(index, n){
    Dtype val = in[index];
    out[index] = sqrt(val * val + epsilon);
  }
}

template <typename Dtype>
void PerceptualLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> delta_x, delta_y;
  delta_x.Reshape(derivative_x_.shape());
  delta_y.Reshape(delta_x.shape());
  // bottom[0] stores predictions of the shape (batch, num_landmark, 2, 1)
  for (int b = 0; b < bottom[0]->num(); ++b){    
    for (int n = 0; n < num_landmark_; ++n){
      delta_x.mutable_cpu_data()[delta_x.offset(b) + n] = 
          bottom[0]->cpu_data()[bottom[0]->offset(b, n)];
      delta_y.mutable_cpu_data()[delta_y.offset(b) + n] = 
          bottom[0]->cpu_data()[bottom[0]->offset(b, n, 1)];
    }
  }
  
  // bottom[1] stores ground truth of the shape (batch, num_landmark, 3, 1)
  for (int b = 0; b < bottom[1]->num(); ++b){
    for (int n = 0; n < num_landmark_; ++n){
      derivative_x_.mutable_cpu_data()[derivative_x_.offset(b) + n] = 
          bottom[1]->cpu_data()[bottom[1]->offset(b, n)];
      derivative_y_.mutable_cpu_data()[derivative_y_.offset(b) + n] = 
          bottom[1]->cpu_data()[bottom[1]->offset(b, n, 1)];
      derivative_t_.mutable_cpu_data()[derivative_t_.offset(b) + n] = 
          bottom[1]->cpu_data()[bottom[1]->offset(b, n, 2)];
    }
  }

  int count = derivative_x_.count();
  caffe_gpu_mul(
    count,
    delta_x.gpu_data(),
    derivative_x_.gpu_data(),
    delta_x.mutable_gpu_data());

  caffe_gpu_mul(
    count,
    delta_y.gpu_data(),
    derivative_y_.gpu_data(),
    delta_y.mutable_gpu_data());
  
  // delta_x * derivative_x + delta_y * derivative_y + derivative_t
  caffe_gpu_add(
    count,
    delta_x.gpu_data(),
    delta_y.gpu_data(),
    sum_.mutable_gpu_data());
  
  caffe_gpu_add(
    count,
    derivative_t_.gpu_data(),
    sum_.gpu_data(),
    sum_.mutable_gpu_data());
  
  ForwardPerceptual<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, sum_.gpu_data(), errors_.mutable_gpu_data(), epsilon_);
  CUDA_POST_KERNEL_CHECK;
  
  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  
}

template <typename Dtype>
void PerceptualLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> diff_x, diff_y, diff;
  diff_x.Reshape(derivative_x_.shape());
  diff_y.Reshape(derivative_y_.shape());
  diff.Reshape(bottom[0]->shape());
  
  // compute diff for delta x
  int count = derivative_x_.count();
  caffe_gpu_mul(
    count,
    sum_.gpu_data(),
    derivative_x_.gpu_data(),
    diff_x.mutable_gpu_data());
  
  caffe_gpu_div(
    count,
    diff_x.gpu_data(),
    errors_.gpu_data(),
    diff_x.mutable_gpu_data());
  
  // compute diff for delta y
  caffe_gpu_mul(
    count,
    sum_.gpu_data(),
    derivative_y_.gpu_data(),
    diff_y.mutable_gpu_data());
  
  caffe_gpu_div(
    count,
    diff_y.gpu_data(),
    errors_.gpu_data(),
    diff_y.mutable_gpu_data());

  for (int b = 0; b < bottom[0]->num(); ++b) {
    for (int n = 0; n < num_landmark_; ++n) {
      diff.mutable_cpu_data()[diff.offset(b, n)] = 
          diff_x.cpu_data()[diff_x.offset(b) + n];
      diff.mutable_cpu_data()[diff.offset(b, n, 1)] = 
          diff_y.cpu_data()[diff_y.offset(b) + n];
    }
  }
  
  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
  caffe_gpu_axpby(
    2 * count,
    alpha,
    diff.gpu_data(),
    Dtype(0),
    bottom[0]->mutable_gpu_diff());
   
}

template <typename Dtype>
void PerceptualPrintBlobDiff(const Blob<Dtype>* blob, string name){
  std::cout << "#################################" << std::endl;
  std::cout << "name:" << name << std::endl;
  std::cout << "num:" << blob->num() << std::endl;
  std::cout << "channel:" << blob->channels() << std::endl;
  std::cout << "height:" << blob->height() << std::endl;
  std::cout << "width:" << blob->width() << std::endl;
  std::cout << "diff:" << std::endl;
  std::cout << "[";
  for (int n = 0; n < blob->num(); ++n){
    for (int c = 0; c < blob->channels(); ++c){
      for (int h = 0; h < blob->height(); ++h){
        for (int w = 0; w < blob->width(); ++w){
          std::cout << blob->diff_at(n,c,h,w) << ",";
        }
      }
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "#################################" << std::endl;
   
}

template <typename Dtype>
void PrintBlob(const Blob<Dtype> &blob, string name){
  std::cout << "#################################" << std::endl;
  std::cout << "name:" << name << std::endl;
  std::cout << "num:" << blob.num() << std::endl;
  std::cout << "channel:" << blob.channels() << std::endl;
  std::cout << "height:" << blob.height() << std::endl;
  std::cout << "width:" << blob.width() << std::endl;
  std::cout << "data:" << std::endl;
  std::cout << "[";
  for (int n = 0; n < blob.num(); ++n){
    for (int c = 0; c < blob.channels(); ++c){
      for (int h = 0; h < blob.height(); ++h){
        for (int w = 0; w < blob.width(); ++w){
          std::cout << blob.data_at(n,c,h,w) << ",";
        }
      }
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "#################################" << std::endl;
   
}

INSTANTIATE_LAYER_GPU_FUNCS(PerceptualLossLayer);
} // namespace caffe
