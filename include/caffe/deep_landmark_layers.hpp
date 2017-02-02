#ifndef CAFFE_DEEP_LANDMARK_LAYERS_HPP_
#define CAFFE_DEEP_LANDMARK_LAYERS_HPP_
#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {
 public:
  explicit YoloLossLayer(const LayerParameter& param)
                        : LossLayer<Dtype>(param),
                          cls_diff_(), conf_diff_(), loc_diff_(), diff_(),
                          cls_lambda_(), conf_lambda_(), loc_lambda_(){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const {return "YoloLoss";}
 
  virtual inline int ExactNumBottomBlobs() const {return 2;}
  virtual inline int MinBottomBlobs() const {return 2;}
  virtual inline int MaxBottomBlobs() const {return 2;}
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  int AdjustLambda(int batch, int grid, const vector<Blob<Dtype>*>& bottom, Blob<Dtype>& conf_lambda, Dtype* max_iou);
  
  int classes_;
  int coords_;
  int side_;
  int boxes_;
  bool softmax_;
  
  Dtype object_scale_;
  Dtype noobject_scale_;
  Dtype class_scale_;
  Dtype coord_scale_;
  
  Blob<Dtype> cls_diff_;
  Blob<Dtype> conf_diff_;
  Blob<Dtype> loc_diff_;
  Blob<Dtype> diff_;
  
  Blob<Dtype> cls_lambda_;
  Blob<Dtype> conf_lambda_;
  Blob<Dtype> loc_lambda_;
};

template <typename Dtype>
class PerceptualLossLayer : public LossLayer<Dtype> {
  public:
   explicit PerceptualLossLayer(const LayerParameter& param)
                               : LossLayer<Dtype>(param),
                                 derivative_x_(),
                                 derivative_y_(),
                                 derivative_t_(),
                                 ones_(), sum_(), errors_(){}
   virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
   
   virtual inline const char* type() const {return "PerceptualLoss";}
   virtual inline int ExactNumBottomBlobs() const {return 2;}    
   virtual inline int MinBottomBlobs() const {return 2;}
   virtual inline int MaxBottomBlobs() const {return 2;}
  
  protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                            const vector<Blob<Dtype>*>& top);

   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
   
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                             const vector<bool>& propagate_down,
                             const vector<Blob<Dtype>*>& bottom);

   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                             const vector<bool>& propagate_down,
                             const vector<Blob<Dtype>*>& bottom);

   int num_landmark_;
   Dtype epsilon_;
   
   Blob<Dtype> derivative_x_;
   Blob<Dtype> derivative_y_;
   Blob<Dtype> derivative_t_;
   Blob<Dtype> ones_;
   Blob<Dtype> sum_;
   Blob<Dtype> errors_;
};

template <typename Dtype>
class SpatialOperation {
 public:
  SpatialOperation()
      :name_(""), used_(false), weight_(Dtype(0)){
    // default stride equals 1
    stride_ = make_pair(1, 1);
    // set accum_stride default to stride
    accum_stride_ = stride_;
    
    // default kenerl equals 3
    kernel_size_ = make_pair(3, 3);
    // set receptive_filed default to kernel
    receptive_field_ = kernel_size_;
  }

  // default, all spatial sizes here are of square shape
  SpatialOperation(int kernel, int stride, 
                   int receptive_field, int accum_stride,
                   string name, bool used, Dtype weight){
    kernel_size_ = make_pair(kernel, kernel);
    stride_ = make_pair(stride, stride);
    receptive_field_ = make_pair(receptive_field, receptive_field);
    accum_stride_ = make_pair(accum_stride, accum_stride);
   
    name_ = name;
    used_ = used;
    weight_ = weight;
  }

  // to specify the spatial size w.r.t width and height
  SpatialOperation(const pair<int, int>& kernel_size, 
                   const pair<int, int>& stride,
                   const pair<int, int>& receptive_field, 
                   const pair<int, int>& accum_stride,
                   string name, bool used, Dtype weight){
    kernel_size_ = kernel_size;
    stride_ = stride;
    receptive_field_ = receptive_field;
    accum_stride_ = accum_stride;

    name_ = name;
    used_ = used;
    weight_ = weight;
  }
  
  SpatialOperation(const SpatialOperationParameter& sp_param){
    CHECK(!sp_param.has_kernel_size() !=
        !(sp_param.has_kernel_w() && sp_param.has_kernel_h()))
        << "Kernel size is kernel_size OR kernel_w and kernel_h; not both.";
    CHECK(sp_param.has_kernel_size() || 
        (sp_param.has_kernel_w() && sp_param.has_kernel_h())) 
        << "For non-square kernels both kernel_h and kernel_w are specified.";
    
    CHECK((!sp_param.has_stride() && sp_param.has_stride_w() 
        && sp_param.has_stride_h()) 
        || (!sp_param.has_stride_w() && !sp_param.has_stride_h() 
        && sp_param.has_stride()))
        << "stride is stride OR stride_w and stride_h are specified.";
    
    // set kernel size
    if (sp_param.has_kernel_size()) {
      SetKernelW(sp_param.kernel_size());
      SetKernelH(sp_param.kernel_size());
    } else {
      SetKernelW(sp_param.kernel_w());
      SetKernelH(sp_param.kernel_h());
    }
    // set stride 
    if (sp_param.has_stride()) {
      SetStrideW(sp_param.stride());
      SetStrideH(sp_param.stride());
    } else {
      SetStrideW(sp_param.stride_w());
      SetStrideH(sp_param.stride_h());
    }
    // set receptive field
    SetReceptiveField(KernelSize());
    // set accum stride
    SetAccumStride(Stride());    

    SetName(sp_param.name());
    SetUsed(sp_param.used());
    SetWeight(sp_param.weight());
  }

  // copy constructor
  SpatialOperation(const SpatialOperation<Dtype>& sp){
    SpatialOperation(sp.KernelSize(),
                     sp.Stride(),
                     sp.ReceptiveField(),
                     sp.AccumStride(),
                     sp.Name(),
                     sp.Used(),
                     sp.Weight());
  }
  
  // getter
  const pair<int, int>& KernelSize() const{
    return kernel_size_;
  }
  const pair<int, int>& Stride() const {
    return stride_;
  }
  const pair<int, int>& ReceptiveField() const {
    return receptive_field_;
  } 
  const pair<int, int>& AccumStride() const {
    return accum_stride_;
  }

  string Name() const {
    return name_;
  }
  bool Used() const {
    return used_;
  }
  Dtype Weight() const {
    return weight_;
  }

  int KernelW() const {
    return kernel_size_.first;
  }
  int KernelH() const {
    return kernel_size_.second;
  }
  int StrideW() const {
    return stride_.first;
  }
  int StrideH() const { 
    return stride_.second; 
  }
  int ReceptiveFieldW() const {
    return receptive_field_.first;
  }
  int ReceptiveFieldH() const {
    return receptive_field_.second;
  }
  int AccumStrideW() const {
    return accum_stride_.first;
  }
  int AccumStrideH() const {
    return accum_stride_.second;
  }

  // setter
  void SetKernelSize(const pair<int, int>& kernel_size){
    kernel_size_ = kernel_size;
  }
  void SetStride(const pair<int, int>& stride){
    stride_ = stride;
  }
  void SetReceptiveField(const pair<int, int>& receptive_field){
    receptive_field_ = receptive_field;
  }
  void SetAccumStride(const pair<int, int>& accum_stride){
    accum_stride_ = accum_stride;
  }

  void SetKernelW(int kernel_w){
    kernel_size_.first = kernel_w;
  }
  void SetKernelH(int kernel_h){
    kernel_size_.second = kernel_h;
  }
  void SetStrideW(int stride_w){
    stride_.first = stride_w;
  }
  void SetStrideH(int stride_h){
    stride_.second = stride_h;
  }
  void SetReceptiveFieldW(int receptive_field_w){
    receptive_field_.first = receptive_field_w;
  }
  void SetReceptiveFieldH(int receptive_field_h){
    receptive_field_.second = receptive_field_h;
  }
  void SetAccumStrideW(int accum_stride_w){
    accum_stride_.first = accum_stride_w;
  }
  void SetAccumStrideH(int accum_stride_h){
    accum_stride_.second = accum_stride_h;
  }
  void SetName(string name){
    name_ = name;
  }
  void SetUsed(bool used){
    used_ = used;
  }
  void SetWeight(Dtype weight){
    weight_ = weight;
  }
  // given a certain SpatialOperation sp
  // followed by the current SpatialOperation
  // calculate the receptive field and accumulated stride of the current spatial operation
  void Following(const SpatialOperation<Dtype>& sp) {
    // update accumulated stride
    SetAccumStrideW(sp.AccumStrideW() * 
                    StrideW());
    SetAccumStrideH(sp.AccumStrideH() * 
                    StrideH());
    // update receptive field
    SetReceptiveFieldW(sp.ReceptiveFieldW() + 
                       (KernelW() - 1) * sp.StrideW());
    SetReceptiveFieldH(sp.ReceptiveFieldH() + 
                       (KernelH() - 1) * sp.StrideH());
  }

 protected:
  // spatial size pair stores size along width dimension first, 
  // followed by size along height dimension
  pair<int, int> receptive_field_;
  pair<int, int> accum_stride_;
  pair<int, int> kernel_size_;
  pair<int, int> stride_;
  string name_;
  bool used_;
  Dtype weight_; 

};
 
template <typename Dtype>
class PatchSemanticLayer : public Layer<Dtype> {
 public:
  explicit PatchSemanticLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type(){return "PatchSemantic";}

  virtual inline int MinBottomBlobs() const {return 3;}
  virtual inline int MinTopBlobs() const {return 1;}
  virtual inline int MaxTopBlobs() const {return 1;}
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  int num_landmark_;
  bool cross_channel_max_;
  bool cross_channel_ave_;
  vector<SpatialOperation<Dtype> > sp_vec_;
  vector<int> index_sp_used_vec_;
  pair<int, int> image_;
  pair<int, int> patch_;
  
};

template <typename Dtype>
class BoundingBox {
 public:
  BoundingBox(Dtype x, Dtype y, Dtype w, Dtype h) {
    x_ = x;
    y_ = y;
    w_ = w;
    h_ = h; 
  }
  // Getter
  Dtype GetX(){
    return x_;
  }
  Dtype GetY() {
    return y_;
  }
  Dtype GetW() {
    return w_;
  }
  Dtype GetH() {
    return h_;
  }
  // Setter
  void SetX(Dtype x){
    x_ = x;
  }
  void SetY(Dtype y){
    y_ = y;
  }
  void SetW(Dtype w){
    w_ = w;
  }
  void SetH(Dtype h){
    h_ = h;
  }

  BoundingBox(const Dtype* in) {
    BoundingBox(in[0], in[1], in[2], in[3]);
  }
  Dtype BoxIou(BoundingBox& b) {
    return Intersection(b) / Union(b);
  }
  Dtype BoxRmse(BoundingBox& b) {
    return sqrt(pow(x_ - b.GetX(), 2) + 
                pow(y_ - b.GetY(), 2) +
                pow(w_ - b.GetW(), 2) + 
                pow(h_ - b.GetH(), 2));
  }
 protected:
  Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
    Dtype l1 = x1 - w1 / 2;
    Dtype l2 = x2 - w2 / 2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1 / 2;
    Dtype r2 = x2 + w2 / 2;
    Dtype right = r1 < r2 ? r1 : r2;
    return right - left;
  }

  Dtype Intersection(BoundingBox& b) {
    Dtype w = Overlap(x_, w_, b.GetX(), b.GetW());
    Dtype h = Overlap(y_, h_, b.GetY(), b.GetH());
    if (w < 0 || h < 0) return 0;
    return w * h;
  }
  
  Dtype Union(BoundingBox& b) {
    Dtype intersection = Intersection(b);
    return w_ * h_ + b.GetW() * b.GetH() - intersection;
  }
 Dtype x_;
 Dtype y_;
 Dtype w_;
 Dtype h_;

};



} // namespace caffe

#endif // CAFFE_DEEP_LANDMARK_LAYERS_HPP_
