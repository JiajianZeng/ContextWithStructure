#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void YoloForwardClassification(int n, const Dtype* in, Dtype* out, const Dtype* cls_lambda){
  // f(x) = 0.5 * x^2 * cls_lambda
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    out[index] = 0.5 * val * val * cls_lambda[index];
  }
}

template <typename Dtype>
__global__ void YoloForwardConfidence(int n, const Dtype* in, Dtype* out, const Dtype* conf_lambda) {
  // f(x) = 0.5 * x^2 * conf_lambda
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    out[index] = 0.5 * val * val * conf_lambda[index];
  }
}

template <typename Dtype>
__global__ void YoloForwardLocalization(int n, const Dtype* in, Dtype* out, const Dtype* loc_lambda) {
  // f(x) = 0.5 * x^2 * loc_lambda
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    out[index] = 0.5 * val * val * loc_lambda[index];
  }
}

template <typename Dtype>
void YoloPrintBlob(const Blob<Dtype> &blob, string name){
  std::cout << "#################################" << std::endl;
  std::cout << "name:" << name << std::endl;
  std::cout << "num:" << blob.num() << std::endl;
  std::cout << "channel:" << blob.channels() << std::endl;
  std::cout << "width:" << blob.width() << std::endl;
  std::cout << "height:" << blob.height() << std::endl;
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

// assign one predictor to be responsible for predicting an object based on which prediction has the highest current IOU with the ground truth
// label.x = object.x / w_image * side_ - col
// label.y = object.y / h_image * side_ - row
// label.w = sqrt(object.w / w_image)   
// label.h = sqrt(object.h / h_image)
template <typename Dtype>
int YoloLossLayer<Dtype>::AdjustLambda(int batch, int grid, const vector<Blob<Dtype>*>& bottom, Blob<Dtype>& conf_lambda, Dtype* highest_iou){
  int gt_offset = batch * side_ * side_ * (1 + coords_ + classes_) + grid * (1 + coords_ + classes_);
  int predicted_offset = batch * side_ * side_ * ((1 + coords_) * boxes_ + classes_) + side_ * side_ * (classes_ + boxes_) + grid * boxes_ * coords_;
  
  BoundingBox<Dtype> gt_box(bottom[1]->cpu_data()[gt_offset + 1 + classes_],
                            bottom[1]->cpu_data()[gt_offset + 1 + classes_ + 1],
                            bottom[1]->cpu_data()[gt_offset + 1 + classes_ + 2],
                            bottom[1]->cpu_data()[gt_offset + 1 + classes_ + 3]);
  gt_box.SetX((gt_box.GetX() + grid % side_) / side_);
  gt_box.SetY((gt_box.GetY() + grid / side_) / side_);
  gt_box.SetW(pow(gt_box.GetW(), 2));
  gt_box.SetH(pow(gt_box.GetH(), 2));
  
  int best_index = -1;
  Dtype max_iou = 0;
  Dtype min_rmse = 20;
  for (int i = 0; i < boxes_; ++i){
    BoundingBox<Dtype> predicted_box(bottom[0]->cpu_data()[predicted_offset + i * coords_],
                                     bottom[0]->cpu_data()[predicted_offset + i * coords_ + 1], 
                                     bottom[0]->cpu_data()[predicted_offset + i * coords_ + 2],
                                     bottom[0]->cpu_data()[predicted_offset + i * coords_ + 3]);
    predicted_box.SetX((predicted_box.GetX() + grid % side_) / side_);
    predicted_box.SetY((predicted_box.GetY() + grid / side_) / side_);
    predicted_box.SetW(pow(predicted_box.GetW(), 2));
    predicted_box.SetH(pow(predicted_box.GetH(), 2));
   
    // intersection over union
    Dtype iou = predicted_box.BoxIou(gt_box);
    // root-mean-square error
    Dtype rmse = predicted_box.BoxRmse(gt_box);
    
    if (max_iou > 0 || iou > 0){
      if (iou > max_iou){
        best_index = i;
        max_iou = iou;
      }
    } else {
      if (rmse < min_rmse){
        best_index = i;
        min_rmse = rmse;
      }
    }
  }
    
  // still can't find a best index based on IOU and RMSE
  if (best_index < 0) {
    best_index = 0;
  }
  // calculate IOU between best box and ground truth
  BoundingBox<Dtype> best_box(bottom[0]->cpu_data()[predicted_offset + best_index * coords_],
                              bottom[0]->cpu_data()[predicted_offset + best_index * coords_ + 1],
                              bottom[0]->cpu_data()[predicted_offset + best_index * coords_ + 2],
                              bottom[0]->cpu_data()[predicted_offset + best_index * coords_ + 3]);
  
  best_box.SetX((best_box.GetX() + grid % side_) / side_);
  best_box.SetY((best_box.GetY() + grid / side_) / side_);
  best_box.SetW(pow(best_box.GetW(), 2));
  best_box.SetH(pow(best_box.GetH(), 2));

  *highest_iou = best_box.BoxIou(gt_box);
  
  conf_lambda.mutable_cpu_data()[conf_lambda.offset(batch, grid, best_index)] = object_scale_;
  return best_index;
}


template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // definitions for computing classification loss
  
  int num_classes_per_image = side_ * side_ * classes_;
  Blob<Dtype> cls_predicted, cls_gt, cls_errors;

  cls_predicted.Reshape(bottom[0]->num(), side_ * side_, classes_, 1);
  cls_gt.Reshape(cls_predicted.shape());
  cls_diff_.Reshape(cls_predicted.shape());
  cls_errors.Reshape(cls_predicted.shape());
  cls_lambda_.Reshape(cls_predicted.shape());

  for (int i = 0; i < cls_lambda_.count(); ++i){ 
    cls_lambda_.mutable_cpu_data()[i] = Dtype(0);
  }

  // definitions for computing confidence loss
  int num_confidences_per_image = side_ * side_ * boxes_;
  Blob<Dtype> conf_predicted, conf_gt, conf_errors;
  
  conf_predicted.Reshape(bottom[0]->num(), side_ * side_, boxes_, 1);
  conf_gt.Reshape(conf_predicted.shape());
  conf_diff_.Reshape(conf_predicted.shape());
  conf_errors.Reshape(conf_predicted.shape());
  conf_lambda_.Reshape(conf_predicted.shape());

  for (int i = 0; i < conf_lambda_.count(); ++i){
    conf_lambda_.mutable_cpu_data()[i] = noobject_scale_;
  }
  
  // definitions for computing localization loss
  int num_coords_per_image = side_ * side_ * boxes_ * coords_;
  Blob<Dtype> loc_predicted, loc_gt, loc_errors;
  
  loc_predicted.Reshape(bottom[0]->num(), side_ * side_, boxes_, coords_);
  loc_gt.Reshape(loc_predicted.shape());
  loc_diff_.Reshape(loc_predicted.shape());
  loc_errors.Reshape(loc_predicted.shape());
  loc_lambda_.Reshape(loc_predicted.shape());

  for (int i = 0; i < loc_lambda_.count(); ++i){
    loc_lambda_.mutable_cpu_data()[i] = Dtype(0);
  }

  // bottom[0] stores B bounding boxes per grid cell, each with 1 confidence and 4 coordinates,
  // the input image is divided into S * S grid cells,
  // and for each grid cell, bottom[0] also predicts C class probabilities.
  // hence, the total number of data is batch * S * S * ((1 + 4) * B + C)
  // (S * S * classes, S * S * B, S * S * B *4) for image 1, image 2, and so on.

  // bottom[1] stores one ground truth bounding box that each grid cell is responsible,
  // the ground truth bounding box has 4 coordinates and 1 confidence,
  // and each grid cell has C class probabilities.
  // hence, the total number of data is batch * S * S * (1 + 4 + C)
  // S * S * (1 confidence + C classes probabilities + 4 coordinates) for image 1 , image 2, and so on.
  for (int b = 0; b < bottom[0]->num(); ++b){    // batch
    for (int i = 0; i < side_ * side_; ++i){    // grid

      int grid_offset = i * (1 + coords_ + classes_); 
      for (int j = 0; j < classes_; ++j){    // class
        // collect classification information
        int index = cls_predicted.offset(b, i, j);
        int offset = cls_predicted.offset(0, i, j);
        cls_predicted.mutable_cpu_data()[index] = bottom[0]->cpu_data()[bottom[0]->offset(b) + offset];
        cls_gt.mutable_cpu_data()[index] = bottom[1]->cpu_data()[bottom[1]->offset(b) + grid_offset + 1 + j];
        cls_lambda_.mutable_cpu_data()[index] = 
            class_scale_ * bottom[1]->cpu_data()[bottom[1]->offset(b) + grid_offset];   // means class_scale_ * the indicator funation
      }
 
      for (int k = 0; k < boxes_; ++k){
        // collect confidence information
        int index = conf_predicted.offset(b, i, k);
        int offset = conf_predicted.offset(0, i, k);
        conf_predicted.mutable_cpu_data()[index] = bottom[0]->cpu_data()[bottom[0]->offset(b) + num_classes_per_image + offset];
        conf_gt.mutable_cpu_data()[index] = bottom[1]->cpu_data()[bottom[1]->offset(b) + grid_offset];

        // collect localization information
        for (int l = 0; l < coords_; ++l){
          int index = loc_predicted.offset(b, i, k, l);
          int offset = loc_predicted.offset(0, i, k, l);
          loc_predicted.mutable_cpu_data()[index] = 
              bottom[0]->cpu_data()[bottom[0]->offset(b) + 
                                    num_classes_per_image + 
                                    num_confidences_per_image + 
                                    offset];
          loc_gt.mutable_cpu_data()[index] = 
              bottom[1]->cpu_data()[bottom[1]->offset(b) + 
                                    grid_offset + 
                                    1 + classes_ + 
                                    l];
          // sqrt is applied to w and h 
          /*if (sqrt_ && (l == 2 || l == 3)){
            loc_gt.mutable_cpu_data()[index] = sqrt(loc_gt.mutable_cpu_data()[index]);
          }*/
        }
      }
      // current grid is responsible for predicting object
      // determine one predictor within that grid to be responsible for predicting that object
      if (bottom[1]->cpu_data()[bottom[1]->offset(b) + grid_offset] == 1){
        Dtype max_iou;
        int box_index = AdjustLambda(b, i, bottom, conf_lambda_, &max_iou);
        // actually, rescore may not be a good strategy
        /*if (rescore_) {
          conf_gt.mutable_cpu_data()[conf_gt.offset(b, i, box_index)] = max_iou;
        }
        */
        for (int l = 0; l < coords_; ++l){
          loc_lambda_.mutable_cpu_data()[loc_lambda_.offset(b, i, box_index, l)] = coord_scale_;
        }
      }
    }
  }
  // forward classification error
  int num_classes_total_batch = bottom[0]->num() * num_classes_per_image;
  caffe_gpu_sub(
      num_classes_total_batch,
      cls_predicted.gpu_data(),
      cls_gt.gpu_data(),
      cls_diff_.mutable_gpu_data());
  YoloForwardClassification<Dtype><<<CAFFE_GET_BLOCKS(num_classes_total_batch), CAFFE_CUDA_NUM_THREADS>>>(
      num_classes_total_batch, cls_diff_.gpu_data(), cls_errors.mutable_gpu_data(), cls_lambda_.gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  // forward confidence error
  int num_confidences_total_batch = bottom[0]->num() * num_confidences_per_image;
  caffe_gpu_sub(
      num_confidences_total_batch,
      conf_predicted.gpu_data(),
      conf_gt.gpu_data(),
      conf_diff_.mutable_gpu_data());
  YoloForwardConfidence<Dtype><<<CAFFE_GET_BLOCKS(num_confidences_total_batch), CAFFE_CUDA_NUM_THREADS>>>(
      num_confidences_total_batch, conf_diff_.gpu_data(), conf_errors.mutable_gpu_data(), conf_lambda_.gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  // forward localization error
  int num_localizations_total_batch = bottom[0]->num() * num_coords_per_image;
  caffe_gpu_sub(
      num_localizations_total_batch,
      loc_predicted.gpu_data(),
      loc_gt.gpu_data(),
      loc_diff_.mutable_gpu_data());
  YoloForwardLocalization<Dtype><<<CAFFE_GET_BLOCKS(num_localizations_total_batch), CAFFE_CUDA_NUM_THREADS>>>(
      num_localizations_total_batch, loc_diff_.gpu_data(), loc_errors.mutable_gpu_data(), loc_lambda_.gpu_data());
  CUDA_POST_KERNEL_CHECK; 
  
  Blob<Dtype> ones;
  Dtype loss;
  // start computing classification loss
  ones.Reshape(bottom[0]->num(), num_classes_per_image, 1, 1);
  for(int i = 0;i < num_classes_total_batch;++i){
    ones.mutable_cpu_data()[i] = Dtype(1);
  }
  caffe_gpu_dot(num_classes_total_batch, ones.gpu_data(), cls_errors.gpu_data(), &loss);
  loss /= bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
  // finish computing classification loss

  // start computing confidence loss
  ones.Reshape(bottom[0]->num(), num_confidences_per_image, 1, 1);
  for(int i = 0;i < num_confidences_total_batch; ++i){
    ones.mutable_cpu_data()[i] = Dtype(1);
  }
  caffe_gpu_dot(num_confidences_total_batch, ones.gpu_data(), conf_errors.gpu_data(), &loss);
  loss /= bottom[0]->num();
  top[0]->mutable_cpu_data()[0] += loss;
  // finish computing confidence loss
  
  // start computing localization loss
  ones.Reshape(bottom[0]->num(), side_ * side_, boxes_, coords_);
  for (int i = 0;i < num_localizations_total_batch; ++i){
    ones.mutable_cpu_data()[i] = Dtype(1);
  }
  caffe_gpu_dot(num_localizations_total_batch, ones.gpu_data(), loc_errors.gpu_data(), &loss);
  loss /= bottom[0]->num();
  top[0]->mutable_cpu_data()[0] += loss;
  // finish computing localization loss
}

template <typename Dtype>
__global__ void YoloBackwardClassification(int n, const Dtype* in, const Dtype* lambda, Dtype* out){
  // f'(x) = x * lambda
  CUDA_KERNEL_LOOP(index, n){
     out[index] = in[index] * lambda[index];
  }
}

template <typename Dtype>
__global__ void YoloBackwardConfidence(int n, const Dtype* in, const Dtype* lambda, Dtype* out){
  // f'(x) = x * lambda
  CUDA_KERNEL_LOOP(index, n){
    out[index] = in[index] * lambda[index];    
  }
}

template <typename Dtype>
__global__ void YoloBackwardLocalization(int n, const Dtype* in, const Dtype* lambda, Dtype* out){
  // f'(x) = x * lambda
  CUDA_KERNEL_LOOP(index, n){
    out[index] = in[index] * lambda[index];
  }
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  // backward classification
  int cls_count = cls_diff_.count();
  YoloBackwardClassification<Dtype><<<CAFFE_GET_BLOCKS(cls_count),CAFFE_CUDA_NUM_THREADS>>>(
      cls_count, cls_diff_.gpu_data(), cls_lambda_.gpu_data(), cls_diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  // backward confidence
  int conf_count = conf_diff_.count();
  YoloBackwardConfidence<Dtype><<<CAFFE_GET_BLOCKS(conf_count), CAFFE_CUDA_NUM_THREADS>>>(
      conf_count, conf_diff_.gpu_data(), conf_lambda_.gpu_data(), conf_diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  // backward localization
  int loc_count = loc_diff_.count();
  YoloBackwardLocalization<Dtype><<<CAFFE_GET_BLOCKS(loc_count), CAFFE_CUDA_NUM_THREADS>>>(
      loc_count, loc_diff_.gpu_data(), loc_lambda_.gpu_data(), loc_diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  int num_classes_per_image = side_ * side_ * classes_;
  int num_confidences_per_image = side_ * side_ * boxes_;
  for (int b = 0; b < bottom[0]->num(); ++b) {
    for (int i = 0; i < side_ * side_; ++i) {
      // classification
      for (int c = 0; c < classes_; ++c){
        int index = cls_diff_.offset(b, i, c);
        int offset = cls_diff_.offset(0, i, c);
        diff_.mutable_cpu_data()[bottom[0]->offset(b) + offset] = cls_diff_.cpu_data()[index];
      }
      for (int j = 0; j < boxes_; ++j){
        // confidence
        int index = conf_diff_.offset(b, i, j);
        int offset = conf_diff_.offset(0, i, j);
        diff_.mutable_cpu_data()[bottom[0]->offset(b) + num_classes_per_image + offset] = conf_diff_.cpu_data()[index];
        for (int l = 0; l < coords_; ++l){
          int index = loc_diff_.offset(b, i, j, l);
          int offset = loc_diff_.offset(0, i, j, l);
          diff_.mutable_cpu_data()[bottom[0]->offset(b) + num_classes_per_image + num_confidences_per_image + offset] = 
              loc_diff_.cpu_data()[index];
        }
      }
    }
  }

  Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
  caffe_gpu_axpby(
    bottom[0]->count(),                 // count
    alpha,                              // alpha
    diff_.gpu_data(),                  // x
    Dtype(0),                           // beta
    bottom[0]->mutable_gpu_diff());     // y
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloLossLayer);

} //namespace caffe
