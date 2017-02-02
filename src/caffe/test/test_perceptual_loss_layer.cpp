#include <cmath>
#include <cstdlib>
#include <cstring>
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
class PerceptualLossLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;
 
 protected:
  // num_landmark = 21
  // 2 * num_landmark = 42
  // 3 * num_landmark = 63
  PerceptualLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 42, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(4, 63, 1, 1 )),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter gaussian_filler_param;
    GaussianFiller<Dtype> gaussian_filler(gaussian_filler_param);
    
    gaussian_filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    FillerParameter uniform_filler_param;
    uniform_filler_param.set_min(0);
    uniform_filler_param.set_max(1);
    UniformFiller<Dtype> uniform_filler(uniform_filler_param);
    uniform_filler.Fill(this->blob_bottom_label_);
    // make some change to label
    /*for (int i = 0; i < 2; ++i){  // batch 
      for (int j = 0; j < 2 * 2; ++j){  // side_ * side_      
        this->blob_bottom_label_->mutable_cpu_data()[i * 28 + j * 7] = (i * j > 0 ? 1 : 0);
      }
    }*/
    blob_bottom_vec_.push_back(blob_bottom_label_);

   
    blob_top_vec_.push_back(blob_top_loss_);
  }
  
  virtual ~PerceptualLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  } 

  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom_data = new Blob<Dtype>(2, 10, 1, 1);
    Blob<Dtype>* blob_bottom_label = new Blob<Dtype>(2, 15, 1, 1);
    Blob<Dtype>* blob_top_loss = new Blob<Dtype>();
    
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    // 
    Dtype data[20] = {Dtype(1), Dtype(2),
                       Dtype(0), Dtype(1),
                       Dtype(1), Dtype(0),
                       Dtype(0.5), Dtype(0.5),
                       Dtype(4), Dtype(0.5),
                       
                       Dtype(1), Dtype(2),
                       Dtype(0), Dtype(1),
                       Dtype(1), Dtype(0),
                       Dtype(0.5), Dtype(0.5),
                       Dtype(4), Dtype(0.5)};
    // 
    Dtype label[30] = {Dtype(0.1), Dtype(2), Dtype(0.2), 
                      Dtype(0), Dtype(0.5), Dtype(0.4),
                      Dtype(0.2), Dtype(1), Dtype(0.1),
                      Dtype(0.5), Dtype(0.6), Dtype(0.5),
                      Dtype(0.2), Dtype(0.4), Dtype(0.3),
                      
                      Dtype(0.1), Dtype(2), Dtype(0.2), 
                      Dtype(0), Dtype(0.5), Dtype(0.4),
                      Dtype(0.2), Dtype(1), Dtype(0.1),
                      Dtype(0.5), Dtype(0.6), Dtype(0.5),
                      Dtype(0.2), Dtype(0.4), Dtype(0.3)};
		      
    for (int i = 0; i < blob_bottom_label->count(); ++i){ 
      blob_bottom_label->mutable_cpu_data()[i] = label[i % 15];
    }
   
    for (int i = 0; i < blob_bottom_data->count(); ++i){
      blob_bottom_data->mutable_cpu_data()[i] = data[i % 10];
    }
   
    blob_bottom_vec.push_back(blob_bottom_data);
    blob_bottom_vec.push_back(blob_bottom_label);
    blob_top_vec.push_back(blob_top_loss);

    // 
    LayerParameter layer_param;
    PerceptualLossParameter* loss_param = 
        layer_param.mutable_perceptual_loss_param();
    loss_param->set_num_landmark(5);
    loss_param->set_epsilon(0.1);

    PerceptualLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(blob_bottom_vec, blob_top_vec);
    const Dtype loss_weight_1 = 
        layer_weight_1.Forward(blob_bottom_vec, blob_top_vec);
    
    const Dtype expected_loss = Dtype(8.1359);
    const Dtype error_margin = 1e-2;
    
    EXPECT_NEAR(expected_loss, loss_weight_1, error_margin);
    std::cout << "epsilon = " << loss_param->epsilon() << std::endl;
    std::cout << "num_landmark = " << loss_param->num_landmark() << std::endl;

    delete blob_bottom_data;
    delete blob_bottom_label;
    delete blob_top_loss;
  }
  
  void TestForward(){
    // Get the loss without a specified objective weight -- should be 
    // equivalent to explicitly specifying a weight of 1
    LayerParameter layer_param;
    PerceptualLossParameter* loss_param = 
        layer_param.mutable_perceptual_loss_param();
    loss_param->set_num_landmark(21);
    loss_param->set_epsilon(0.05);
    
    PerceptualLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 = 
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is scaled appropriately
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    
    PerceptualLossLayer<Dtype> layer_weight_k(layer_param);
    layer_weight_k.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_k = 
        layer_weight_k.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_k, kErrorMargin);
    // make sure that the loss is non-trival
    const Dtype kNonTrivalAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivalAbsThresh);
  }
  
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
};

TYPED_TEST_CASE(PerceptualLossLayerTest, TestDtypesGPU);

TYPED_TEST(PerceptualLossLayerTest, TestForward){
  this->TestForward();
}


TYPED_TEST(PerceptualLossLayerTest, TestGradient){
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PerceptualLossParameter* loss_param = 
        layer_param.mutable_perceptual_loss_param();
  loss_param->set_num_landmark(21);
  loss_param->set_epsilon(0.05);
  
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  PerceptualLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);

}

TYPED_TEST(PerceptualLossLayerTest, TestForwardSpecifiedExample){
  this->TestForwardSpecifiedExample();
}


} // namespace caffe
