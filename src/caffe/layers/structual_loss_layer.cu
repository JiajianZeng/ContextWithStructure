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
void StructualLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
/*
  std::cout << "Count: "<< bottom[0]->count() << std::endl;
  std::cout << "Num: "<< bottom[0]->num() << std::endl;
  std::cout << "Channels: "<< bottom[0]->channels() << std::endl;
  std::cout << "Height: "<< bottom[0]->height() << std::endl;
  std::cout << "Width: "<< bottom[0]->width() << std::endl;
  for(int i=0;i < bottom[1]->count();i++){
     std::cout << bottom[1]->cpu_data()[i] << std::endl;

  }
*/
  

  Dtype root_x, root_y;
  Blob<Dtype> pre_struct;
  Blob<Dtype> gt_struct;
  Blob<Dtype> diff_x_2;
  Blob<Dtype> diff_y_2;
  Blob<Dtype> sum;
  Blob<Dtype> ones;
  Blob<Dtype> diffx_tmp;
  Blob<Dtype> diffy_tmp;
  int count;
  diff_x_2.Reshape(diff_x_pre_.shape());
  diff_y_2.Reshape(diff_x_pre_.shape());
  sum.Reshape(diff_x_pre_.shape());
  ones.Reshape(diff_x_pre_.shape());
  pre_struct.Reshape(bottom[0]->num(),1,1,1);
  gt_struct.Reshape(bottom[1]->num(),1,1,1);
 
  for (int i = 0; i < 2 ; i++){

    for (int b = 0; b < bottom[0]->num(); ++b){ 
      
      root_x = bottom[i]->cpu_data()[bottom[i]->offset(b,num_landmark_ )];
      root_y = bottom[i]->cpu_data()[bottom[i]->offset(b,num_landmark_ ,1)];
      //std::cout<<  root_x <<std::endl;
      //std::cout<<  root_y <<std::endl;
      for (int n = 0; n < num_landmark_; ++n){
        if(i==0) {
          diff_x_pre_.mutable_cpu_data()[diff_x_pre_.offset(b,n)] = bottom[i]->cpu_data()[bottom[i]->offset(b, n)] - root_x;
          diff_y_pre_.mutable_cpu_data()[diff_y_pre_.offset(b,n)] = bottom[i]->cpu_data()[bottom[i]->offset(b, n, 1)] - root_y;
        }
        else if(i==1) {
          diff_x_gt_.mutable_cpu_data()[diff_x_gt_.offset(b,n)] = bottom[i]->cpu_data()[bottom[i]->offset(b, n)] - root_x;
          diff_y_gt_.mutable_cpu_data()[diff_y_gt_.offset(b,n)] = bottom[i]->cpu_data()[bottom[i]->offset(b, n, 1)] - root_y;
        }
        //std::cout <<  bottom[i]->cpu_data()[bottom[i]->offset(b, n)] - root_x<<std::endl; 
        //std::cout <<  bottom[i]->cpu_data()[bottom[i]->offset(b, n, 1)] - root_y<<std::endl; 
      }
    }
    count = diff_x_pre_.count();

    caffe_gpu_set(
      count,
      Dtype(1),
      ones.mutable_gpu_data());

    if(i == 0) {
      caffe_gpu_mul(
        count,
        diff_x_pre_.gpu_data(),
        diff_x_pre_.gpu_data(),
        diff_x_2.mutable_gpu_data());

      caffe_gpu_mul(
        count,
        diff_y_pre_.gpu_data(),
        diff_y_pre_.gpu_data(),
        diff_y_2.mutable_gpu_data());

      caffe_gpu_axpby(
        count,              // count
        w0_,                              // alpha
        diff_x_pre_.gpu_data(),                   // a
        Dtype(0),                           // beta
        sum.mutable_gpu_data());  // b

      caffe_gpu_axpby(
        count,              // count
        w2_,                              // alpha
        diff_y_pre_.gpu_data(),                   // a
        Dtype(1),                           // beta
        sum.mutable_gpu_data());  // b
    }
    else if(i==1){
      caffe_gpu_mul(
        count,
        diff_x_gt_.gpu_data(),
        diff_x_gt_.gpu_data(),
        diff_x_2.mutable_gpu_data());

      caffe_gpu_mul(
        count,
        diff_y_gt_.gpu_data(),
        diff_y_gt_.gpu_data(),
        diff_y_2.mutable_gpu_data());

      caffe_gpu_axpby(
        count,              // count
        w0_,                              // alpha
        diff_x_gt_.gpu_data(),                   // a
        Dtype(0),                           // beta
        sum.mutable_gpu_data());  // b

      caffe_gpu_axpby(
        count,              // count
        w2_,                              // alpha
        diff_y_gt_.gpu_data(),                   // a
        Dtype(1),                           // beta
        sum.mutable_gpu_data());  // b
    }

    caffe_gpu_axpby(
      count,              // count
      w1_,                              // alpha
      diff_x_2.gpu_data(),                   // a
      Dtype(1),                           // beta
      sum.mutable_gpu_data());  // b

    caffe_gpu_axpby(
      count,              // count
      w3_,                              // alpha
      diff_y_2.gpu_data(),                   // a
      Dtype(1),                           // beta
      sum.mutable_gpu_data());  // b 
/*
    for(int j=0;j < count;j++){
//      std::cout << diff_x_.cpu_data()[i] << std::endl;
//      std::cout << diff_y_.cpu_data()[i] << std::endl;
      std::cout << sum.cpu_data()[j] << std::endl;
    }*/

    Dtype sign = i ? Dtype(-1) : Dtype(1);
    Dtype w = i ? Dtype(1) : Dtype(0);
    caffe_gpu_axpby(
      count,              // count
      sign,                              // alpha
      sum.gpu_data(),                   // a
      w,                           // beta
      diff_.mutable_gpu_data());  // b 
 /*
     if(i == 0) {
       for (int b = 0; b < bottom[0]->num(); ++b){ 
         Dtype struct_sum = Dtype(0);
         for (int n = 0; n < num_landmark_; ++n){
           struct_sum = struct_sum + sum.cpu_data()[sum.offset(b,n)];
         }
         //std::cout<<  struct_sum <<std::endl;
         pre_struct.mutable_cpu_data()[pre_struct.offset(b)] = struct_sum;
       }
       
     }
     else if(i == 1) {
       for (int b = 0; b < bottom[0]->num(); ++b){ 
         Dtype struct_sum = Dtype(0);
         for (int n = 0; n < num_landmark_; ++n){
           struct_sum = struct_sum + sum.cpu_data()[sum.offset(b,n)];
         }
         //std::cout<<  struct_sum <<std::endl;
         gt_struct.mutable_cpu_data()[gt_struct.offset(b)] = struct_sum;
       }
     }
*/
  }
/*
  count = gt_struct.count();
  caffe_gpu_sub(
      count,
      pre_struct.gpu_data(),
      gt_struct.gpu_data(),
      diff_.mutable_gpu_data());
*/
  Dtype loss;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / 2;

}

template <typename Dtype>
void StructualLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> diff_x, diff_y, diff;
  Blob<Dtype> ones;
  diff_x.Reshape(diff_x_pre_.shape());
  diff_y.Reshape(diff_x_pre_.shape());
  diff.Reshape(bottom[0]->shape());

  ones.Reshape(diff_x_pre_.shape());
  // compute diff for delta x
  int count = diff_x_pre_.count();

  for(int i = 0; i < 2; i++){
    caffe_gpu_set(
       count,
       Dtype(-1),
       ones.mutable_gpu_data());
    if(i==0) {
      caffe_gpu_axpby(
        count,              // count
        2*w1_,                              // alpha
        diff_x_pre_.gpu_data(),                   // a
        Dtype(0),                           // beta
        diff_x.mutable_gpu_data());  // b
      caffe_gpu_axpby(
        count,              // count
        2*w3_,                              // alpha
        diff_y_pre_.gpu_data(),                   // a
        Dtype(0),                           // beta
        diff_y.mutable_gpu_data());  // b
    }
    else if(i==1){
      caffe_gpu_axpby(
        count,              // count
        2*w1_,                              // alpha
        diff_x_gt_.gpu_data(),                   // a
        Dtype(0),                           // beta
        diff_x.mutable_gpu_data());  // b
      caffe_gpu_axpby(
        count,              // count
        2*w3_,                              // alpha
        diff_y_gt_.gpu_data(),                   // a
        Dtype(0),                           // beta
        diff_y.mutable_gpu_data());  // b
    }
    
  
    caffe_gpu_add_scalar(
      count,
      w0_,
      diff_x.mutable_gpu_data());
  
    caffe_gpu_add_scalar(
      count,
      w2_,
      diff_y.mutable_gpu_data());

  //Dtype root_x_diff, root_y_diff;
  //caffe_gpu_dot(count, ones.gpu_data(), diff_x, &root_x_diff);
  //caffe_gpu_dot(count, ones.gpu_data(), diff_y, &root_y_diff);
    int sign = i ? -1 : 1; 
    for (int b = 0; b < bottom[0]->num(); ++b) {
      Dtype struct_sumx = Dtype(0);
      Dtype struct_sumy = Dtype(0);
      
      for (int n = 0; n < num_landmark_ ; ++n) {
        Dtype weight = sign*diff_.cpu_data()[diff_.offset(b,n)];
        diff.mutable_cpu_data()[diff.offset(b, n)] = weight*diff_x.cpu_data()[diff_x.offset(b) + n];
        struct_sumx = struct_sumx + weight*diff_x.cpu_data()[diff_x.offset(b) + n];
        diff.mutable_cpu_data()[diff.offset(b, n, 1)] = weight*diff_y.cpu_data()[diff_y.offset(b) + n];
        struct_sumy = struct_sumy + weight*diff_y.cpu_data()[diff_y.offset(b) + n];
      }
      diff.mutable_cpu_data()[diff.offset(b, num_landmark_)] = -struct_sumx;
      diff.mutable_cpu_data()[diff.offset(b, num_landmark_, 1)] = -struct_sumy;
    }

    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num();
    caffe_gpu_axpby(
      bottom[0]->count(),
      alpha,
      diff.gpu_data(),
      Dtype(0),
      bottom[i]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(StructualLossLayer);
} // namespace caffe
