#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/deep_landmark_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;
template <typename TypeParam>
class StructualLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
    StructualLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 12, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(2, 12, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~StructualLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifying a weight of 1.
    LayerParameter layer_param;
    StructualLossParameter *struct_param = layer_param.mutable_structual_loss_param();
    struct_param->set_w0(-1);
    struct_param->set_w1(0.5);
    struct_param->set_w2(-1);
    struct_param->set_w3(0.5);
    StructualLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    StructualLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }
  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom_data = new Blob<Dtype>(2, 12, 1, 1);
    Blob<Dtype>* blob_bottom_label = new Blob<Dtype>(2, 12, 1, 1);
    Blob<Dtype>* blob_top_loss = new Blob<Dtype>();
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    // 
     Dtype landmark_gt[24] = {
                       Dtype(3), Dtype(8),
                       Dtype(12), Dtype(7),
                       Dtype(8), Dtype(10),
                       Dtype(5), Dtype(15),
                       Dtype(13), Dtype(16),
                       Dtype(11), Dtype(7),

                       Dtype(5), Dtype(6),
                       Dtype(16), Dtype(5),
                       Dtype(9), Dtype(11),
                       Dtype(6), Dtype(17),
                       Dtype(14), Dtype(15),
                       Dtype(12), Dtype(9)};
    // 
    Dtype landmark_pre[24] = {
                       Dtype(4), Dtype(7),
                       Dtype(17), Dtype(8),
                       Dtype(6), Dtype(10),
                       Dtype(7), Dtype(10),
                       Dtype(6), Dtype(18),
                       Dtype(10), Dtype(8),

                       Dtype(3), Dtype(9),
                       Dtype(13), Dtype(7),
                       Dtype(10), Dtype(14),
                       Dtype(8), Dtype(14),
                       Dtype(17), Dtype(11),
                       Dtype(13), Dtype(10)};
    	      
    for (int i = 0; i < blob_bottom_label->count(); ++i){ 
      blob_bottom_label->mutable_cpu_data()[i] = landmark_gt[i];
    }
    
    for (int i = 0; i < blob_bottom_data->count(); ++i){
      blob_bottom_data->mutable_cpu_data()[i] = landmark_pre[i];
    }
   
    blob_bottom_vec.push_back(blob_bottom_data);
    blob_bottom_vec.push_back(blob_bottom_label);
    blob_top_vec.push_back(blob_top_loss);
    // 
    LayerParameter layer_param;
    StructualLossParameter* param = 
        layer_param.mutable_structual_loss_param();
    param->set_w0(-1);
    param->set_w1(0.5);
    param->set_w2(-1);
    param->set_w3(0.5);
    param->set_num_landmark(5);
    StructualLossLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec); 
    layer.Forward(blob_bottom_vec, blob_top_vec);
    
    // print and test 
    const Dtype error_margin = 1e-2;
    Dtype loss = Dtype(989.625);
    
    EXPECT_NEAR(loss, blob_top_vec[0]->cpu_data()[0], error_margin);
    

    delete blob_bottom_data;
    delete blob_bottom_label;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

  

TYPED_TEST_CASE(StructualLossLayerTest, TestDtypesGPU);


TYPED_TEST(StructualLossLayerTest, TestForward) {
  this->TestForward();
}

  

TYPED_TEST_CASE(StructualLossLayerTest, TestDtypesGPU);

TYPED_TEST(StructualLossLayerTest, TestForwardSpecifiedExample) {
  this->TestForwardSpecifiedExample();
}


TYPED_TEST(StructualLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  StructualLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe
