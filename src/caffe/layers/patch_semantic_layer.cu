#include "caffe/deep_landmark_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype Distance_xy(Dtype x1, Dtype y1, Dtype x2, Dtype y2){
  return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}
template <typename Dtype>
Dtype GE_and_LE(Dtype x, int min, int max) {
  x = x < min ? min : x;
  x = x > max ? max : x;
  return x;
}

void Right_and_Down(const pair<int, int>& fiducial, int width, int height, 
                    bool* right, bool* down){
  int x = fiducial.first, y = fiducial.second;
  *right = (x + 1 < width);
  *down = (y + 1 < height);
}

template <typename Dtype>
void best_fiducial(const pair<Dtype, Dtype>& landmark, const pair<int, int>& image,
                   const pair<int, int>& patch, const SpatialOperation<Dtype>& sp,
                   const Blob<Dtype>* feature, pair<int, int>& fiducial){
  // image width and height
  Dtype img_w = image.first, img_h = image.second;
  // patch width and height
  Dtype patch_w = patch.first, patch_h = patch.second;
  // tweak landmark that crosses image boundary
  Dtype landmark_x = ceil(landmark.first), landmark_y = ceil(landmark.second);
  landmark_x = GE_and_LE(landmark_x, 0, img_w); 
  landmark_y = GE_and_LE(landmark_y, 0, img_h);
  // create a small patch around the landmark 
  // that does not cross image boundary
  Dtype x1 = landmark_x - patch_w / 2, x2 = landmark_x + patch_w / 2;
  Dtype y1 = landmark_y - patch_h / 2, y2 = landmark_y + patch_h / 2;
  x1 = GE_and_LE(x1, 0, img_w);
  x2 = GE_and_LE(x2, 0, img_w);
  y1 = GE_and_LE(y1, 0, img_h);
  y2 = GE_and_LE(y2, 0, img_h);
  //BoundingBox<Dtype> around_patch(landmark_x, landmark_y, x2 - x1, y2 - y1);
  BoundingBox<Dtype> around_patch(x1, y1, x2 - x1, y2 - y1);
  
  Dtype s_h = sp.AccumStrideH();
  Dtype s_w = sp.AccumStrideW();
  Dtype r_h = sp.ReceptiveFieldH();
  Dtype r_w = sp.ReceptiveFieldW();

  Dtype max_iou = 0;
  Dtype mdist = 0;
  // find the receptive box that has the highest IOU with the around patch defined aforemetioned
  for (int h = 0; h < feature->height(); ++h) {
    Dtype y1 = h * s_h, y2 = h * s_h + r_h;
    for (int w = 0; w < feature->width(); ++w) {
      Dtype x1 = w * s_w, x2 = w * s_w + r_w;
      x1 = GE_and_LE(x1, 0, img_w);
      x2 = GE_and_LE(x2, 0, img_w);
      y1 = GE_and_LE(y1, 0, img_h);
      y2 = GE_and_LE(y2, 0, img_h);
      //BoundingBox<Dtype> receptive_box(x1 + r_w/2, y1 + r_h/2, x2 - x1, y2 - y1);
      BoundingBox<Dtype> receptive_box(x1, y1, x2 - x1, y2 - y1);
      //Dtype iou = receptive_box.BoxIou(around_patch);
      Dtype iou = receptive_box.BoxIou_ps(around_patch);
      if (iou > max_iou) {
        fiducial.first = w;
        fiducial.second = h;
        mdist = Distance_xy(x1+r_w/2, y1+r_h/2, landmark_x, landmark_y);
        max_iou = iou;
      } 
      else if(iou == max_iou){
        if(Distance_xy(x1+r_w/2, y1+r_h/2, landmark_x, landmark_y) < mdist){
          fiducial.first = w;
          fiducial.second = h;
          mdist = Distance_xy(x1+r_w/2, y1+r_h/2, landmark_x, landmark_y);
        }
      }

      
    }
  }  
  
}

template <typename Dtype>
void compute_diff_x_and_y(int batch, int channel, bool right, bool down, 
                    const pair<int, int>& fiducial, const Blob<Dtype>* feature, 
                    Dtype* diff_x, Dtype* diff_y){
  int h = fiducial.second, w = fiducial.first;
  if (right && down ) {
    *diff_x = feature->data_at(batch, channel, h, w + 1) - 
              feature->data_at(batch, channel, h, w) +
              feature->data_at(batch, channel, h + 1, w + 1) -
              feature->data_at(batch, channel, h + 1, w);

    *diff_y = feature->data_at(batch, channel, h + 1, w) - 
              feature->data_at(batch, channel, h, w) + 
              feature->data_at(batch, channel, h + 1, w + 1) -
              feature->data_at(batch, channel, h, w + 1);
  } else if (!right && down) {
    *diff_x = feature->data_at(batch, channel, h, w) - 
              feature->data_at(batch, channel, h, w - 1) + 
              feature->data_at(batch, channel, h + 1, w) - 
              feature->data_at(batch, channel, h + 1, w - 1);
    
    *diff_y = feature->data_at(batch, channel, h + 1, w) -
              feature->data_at(batch, channel, h, w) + 
              feature->data_at(batch, channel, h + 1, w - 1) - 
              feature->data_at(batch, channel, h, w - 1);
  } else if (right && !down) {
    *diff_x = feature->data_at(batch, channel, h, w + 1) - 
              feature->data_at(batch, channel, h, w) + 
              feature->data_at(batch, channel, h - 1, w + 1) - 
              feature->data_at(batch, channel, h - 1, w);

    *diff_y = feature->data_at(batch, channel, h, w) - 
              feature->data_at(batch, channel, h - 1, w) + 
              feature->data_at(batch, channel, h, w + 1) - 
              feature->data_at(batch, channel, h - 1, w + 1);
  } else {
    *diff_x = feature->data_at(batch, channel, h, w) - 
              feature->data_at(batch, channel, h, w - 1) + 
              feature->data_at(batch, channel, h - 1, w) - 
              feature->data_at(batch, channel, h - 1, w - 1);

    *diff_y = feature->data_at(batch, channel, h, w) - 
              feature->data_at(batch, channel, h - 1, w) + 
              feature->data_at(batch, channel, h, w - 1) - 
              feature->data_at(batch, channel, h - 1, w - 1);
  }
  
}


// compute the corresponding Fx, Fy and Ft for a certain landmark within a certain image
template <typename Dtype>
void compute_fx_fy_ft(int batch, const pair<Dtype, Dtype>& gt_landmark, const pair<Dtype, Dtype>& predicted_landmark,
                          const pair<int, int>& image, const pair<int, int>& patch,
                          const Blob<Dtype>* feature, const SpatialOperation<Dtype>& sp,
                          Dtype* Fx, Dtype* Fy, Dtype* Ft){
  
  pair<int, int> gt_fiducial;
  best_fiducial(gt_landmark, image, patch, sp, feature, gt_fiducial);
  
  pair<int, int> predicted_fiducial;
  best_fiducial(predicted_landmark, image, patch, sp, feature, predicted_fiducial);
  
  bool gt_right, gt_down;
  Right_and_Down(gt_fiducial, feature->width(), feature->height(), 
                 &gt_right, &gt_down);
  
  bool predicted_right, predicted_down;
  Right_and_Down(predicted_fiducial, feature->width(), feature->height(),
                 &predicted_right, &predicted_down);
  
  // compute Fx and Fy using AVE cross channel strategy 
  // TODO
  // another cross channel strategy (MAX)
  *Fx = Dtype(0);
  *Fy = Dtype(0);
  for (int c = 0; c < feature->channels(); ++c) {
    Dtype diff_x_gt, diff_x_predicted;
    Dtype diff_y_gt, diff_y_predicted;
    // compute diff w.r.t x and y using ground truth landmark
    compute_diff_x_and_y(batch, c, gt_right, gt_down, 
        gt_fiducial, feature, &diff_x_gt, &diff_y_gt);
    // compute diff w.r.t x and y using predicted landmark
    compute_diff_x_and_y(batch, c, predicted_right, predicted_down,
        predicted_fiducial, feature, &diff_x_predicted, &diff_y_predicted);
    *Fx = *Fx + 0.25 * (diff_x_gt + diff_x_predicted);
    *Fy = *Fy + 0.25 * (diff_y_gt + diff_y_predicted);
  }
  *Fx /= feature->channels();
  *Fy /= feature->channels();  
  
  // compute Ft using AVE cross channel strategy
  // TODO
  // another cross channel strategy (MAX) 
  int h_gt = gt_fiducial.second, h_pre = predicted_fiducial.second;
  int w_gt = gt_fiducial.first, w_pre = predicted_fiducial.first;
  // sign for computing
  int h_sign_gt = gt_down ? 1 : -1;
  int w_sign_gt = gt_right ? 1 : -1;
  int h_sign_pre = predicted_down ? 1 : -1;
  int w_sign_pre = predicted_right ? 1 : -1;
  if(w_sign_gt != w_sign_pre && w_gt != 0 && w_pre != 0) w_sign_gt = w_sign_pre = -1;
  if(h_sign_gt != h_sign_pre && h_gt != 0 && h_pre != 0) h_sign_gt = h_sign_pre = -1;
  *Ft = Dtype(0);
  for (int c = 0; c < feature->channels(); ++c) {  
    Dtype ft_this_channel = Dtype(0);
    ft_this_channel = feature->data_at(batch, c, h_pre, w_pre) - 
                      feature->data_at(batch, c, h_gt, w_gt) + 
                      
                      feature->data_at(batch, c, h_pre, w_pre + w_sign_pre) - 
                      feature->data_at(batch, c, h_gt, w_gt + w_sign_gt) + 
                      
                      feature->data_at(batch, c, h_pre + h_sign_pre, w_pre) - 
                      feature->data_at(batch, c, h_gt + h_sign_gt, w_gt) + 
                      
                      feature->data_at(batch, c, h_pre + h_sign_pre, w_pre + w_sign_pre) - 
                      feature->data_at(batch, c, h_gt + h_sign_gt, w_gt + w_sign_gt);
    *Ft = *Ft + 0.25 * ft_this_channel;                     
  }
  *Ft /= feature->channels();
}

template <typename Dtype>
void PatchSemanticLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // there are bottom.size() - 2 feature layers to be used 
  for (int b = 0; b < bottom[0]->num(); ++b) {
    for (int l = 0; l < bottom[0]->channels(); ++l) {
      // bottom[0] stores the current landmark shape
      // while bottom[1] stores the landmark shift w.r.t the current landmark shape
      pair<Dtype, Dtype> gt_landmark(bottom[0]->data_at(b, l, 0, 0),
                                     bottom[0]->data_at(b, l, 1, 0));
      pair<Dtype, Dtype> pre_landmark(bottom[1]->data_at(b, l, 0, 0),
                                      bottom[1]->data_at(b, l, 1, 0));
      
      Dtype fx = Dtype(0), fy = Dtype(0), ft = Dtype(0);
      // compute each layer Fx, Fy and Ft
      for (int j = 0; j < index_sp_used_vec_.size(); ++j) {
        Dtype fx_this_layer, fy_this_layer, ft_this_layer;
        int sp_index = index_sp_used_vec_[j];
        
        compute_fx_fy_ft(b, gt_landmark, pre_landmark,
                         image_, patch_, 
                         bottom[j + 2], sp_vec_[sp_index],
                         &fx_this_layer, &fy_this_layer, &ft_this_layer);
        
        fx = fx + fx_this_layer * sp_vec_[sp_index].Weight();
        fy = fy + fy_this_layer * sp_vec_[sp_index].Weight();
        ft = ft + ft_this_layer * sp_vec_[sp_index].Weight();
      }
      //std::cout << pre_landmark.second << std::endl;
      top[0]->mutable_cpu_data()[top[0]->offset(b, l)] = fx;
      top[0]->mutable_cpu_data()[top[0]->offset(b, l, 1)] = fy;
      top[0]->mutable_cpu_data()[top[0]->offset(b, l, 2)] = ft;
    }  
  }

}

template <typename Dtype>
void PatchSemanticLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(PatchSemanticLayer);

}; // namespace caffe
