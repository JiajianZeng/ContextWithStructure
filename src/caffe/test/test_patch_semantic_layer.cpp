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
class PatchSemanticLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;
 
 protected:
  // num_landmark = 21
  // 2 * num_landmark = 42
  // 3 * num_landmark = 63
  PatchSemanticLayerTest()
      : blob_bottom_gt_(new Blob<Dtype>(2, 5, 2, 1)),
        blob_bottom_pre_(new Blob<Dtype>(2, 5, 2, 1 )),
        blob_bottom_feature1_(new Blob<Dtype>(2, 2, 18, 18 )),
        blob_bottom_feature2_(new Blob<Dtype>(2, 2, 8, 8 )),
        blob_top_(new Blob<Dtype>(2, 21, 3, 1))

        {
    // fill the values

    FillerParameter uniform_filler_param;
    uniform_filler_param.set_min(0);
    uniform_filler_param.set_max(19);
    UniformFiller<Dtype> uniform_filler(uniform_filler_param);
    uniform_filler.Fill(this->blob_bottom_gt_);
    uniform_filler.Fill(this->blob_bottom_pre_);

    uniform_filler_param.set_min(-10);
    uniform_filler_param.set_max(10);
    UniformFiller<Dtype> feature_filler(uniform_filler_param);
    feature_filler.Fill(this->blob_bottom_feature1_);
    feature_filler.Fill(this->blob_bottom_feature2_);
    // make some change to label
    /*for (int i = 0; i < 2; ++i){  // batch 
      for (int j = 0; j < 2 * 2; ++j){  // side_ * side_      
        this->blob_bottom_label_->mutable_cpu_data()[i * 28 + j * 7] = (i * j > 0 ? 1 : 0);
      }
    }*/
    blob_bottom_vec_.push_back(blob_bottom_gt_);
    blob_bottom_vec_.push_back(blob_bottom_pre_);
    blob_bottom_vec_.push_back(blob_bottom_feature1_);
    blob_bottom_vec_.push_back(blob_bottom_feature2_);
    blob_top_vec_.push_back(blob_top_);
  }
  
  virtual ~PatchSemanticLayerTest() {
    delete blob_bottom_gt_;
    delete blob_bottom_pre_;
    delete blob_bottom_feature1_;
    delete blob_bottom_feature2_;
    delete blob_top_;
  } 

  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom_gt = new Blob<Dtype>(2, 5, 2, 1);
    Blob<Dtype>* blob_bottom_pre = new Blob<Dtype>(2, 5, 2, 1);
    Blob<Dtype>* blob_bottom_feature1 = new Blob<Dtype>(2, 2, 18, 18);
    Blob<Dtype>* blob_bottom_feature2 = new Blob<Dtype>(2, 2, 8, 8);
    Blob<Dtype>* blob_top = new Blob<Dtype>();
    
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    // 
    Dtype landmark_gt[20] = {
                       Dtype(3), Dtype(8),
                       Dtype(12), Dtype(7),
                       Dtype(8), Dtype(10),
                       Dtype(5), Dtype(15),
                       Dtype(13), Dtype(16),
                       
                       Dtype(5), Dtype(6),
                       Dtype(16), Dtype(5),
                       Dtype(9), Dtype(11),
                       Dtype(6), Dtype(17),
                       Dtype(14), Dtype(15)};
    // 
    Dtype landmark_pre[20] = {
                       Dtype(4), Dtype(7),
                       Dtype(17), Dtype(8),
                       Dtype(6), Dtype(10),
                       Dtype(7), Dtype(10),
                       Dtype(6), Dtype(18),
                       
                       Dtype(3), Dtype(9),
                       Dtype(13), Dtype(7),
                       Dtype(10), Dtype(14),
                       Dtype(8), Dtype(14),
                       Dtype(17), Dtype(11)};
    Dtype feature1_data[324] = {
      Dtype(-9), Dtype(-5), Dtype(0), Dtype(6), Dtype(8), Dtype(-2), Dtype(7), Dtype(-3), Dtype(7), Dtype(-10), Dtype(-2), Dtype(5), Dtype(-1), Dtype(0), Dtype(-4), Dtype(8), Dtype(7), Dtype(7), 
      Dtype(-1), Dtype(7), Dtype(1), Dtype(6), Dtype(8), Dtype(3), Dtype(5), Dtype(9), Dtype(-9), Dtype(4), Dtype(1), Dtype(10), Dtype(-4), Dtype(-4), Dtype(-6), Dtype(-8), Dtype(-4), Dtype(3), 
      Dtype(1), Dtype(9), Dtype(0), Dtype(1), Dtype(-8), Dtype(-6), Dtype(-2), Dtype(-1), Dtype(-10), Dtype(-7), Dtype(-2), Dtype(-9), Dtype(4), Dtype(-9), Dtype(6), Dtype(2), Dtype(9), Dtype(5), 
      Dtype(3), Dtype(-3), Dtype(1), Dtype(-3), Dtype(-2), Dtype(0), Dtype(-3), Dtype(5), Dtype(6), Dtype(-1), Dtype(4), Dtype(5), Dtype(9), Dtype(1), Dtype(5), Dtype(6), Dtype(3), Dtype(-10), 
      Dtype(0), Dtype(7), Dtype(1), Dtype(-3), Dtype(-9), Dtype(0), Dtype(8), Dtype(-5), Dtype(6), Dtype(5), Dtype(-6), Dtype(6), Dtype(1), Dtype(10), Dtype(5), Dtype(-9), Dtype(-3), Dtype(4), 
      Dtype(9), Dtype(5), Dtype(3), Dtype(10), Dtype(10), Dtype(9), Dtype(-10), Dtype(-5), Dtype(2), Dtype(9), Dtype(-10), Dtype(-4), Dtype(8), Dtype(1), Dtype(-5), Dtype(8), Dtype(9), Dtype(-9), 
      Dtype(-5), Dtype(8), Dtype(-3), Dtype(7), Dtype(3), Dtype(4), Dtype(-4), Dtype(4), Dtype(-2), Dtype(-3), Dtype(-3), Dtype(-4), Dtype(1), Dtype(1), Dtype(-5), Dtype(1), Dtype(5), Dtype(-8), 
      Dtype(8), Dtype(6), Dtype(0), Dtype(6), Dtype(-7), Dtype(-10), Dtype(3), Dtype(-5), Dtype(3), Dtype(4), Dtype(6), Dtype(-2), Dtype(3), Dtype(-8), Dtype(-10), Dtype(-10), Dtype(0), Dtype(-8), 
      Dtype(-8), Dtype(-5), Dtype(1), Dtype(4), Dtype(2), Dtype(-7), Dtype(0), Dtype(8), Dtype(1), Dtype(8), Dtype(6), Dtype(-6), Dtype(9), Dtype(9), Dtype(4), Dtype(4), Dtype(5), Dtype(4), 
      Dtype(-6), Dtype(-3), Dtype(9), Dtype(9), Dtype(-7), Dtype(-9), Dtype(-5), Dtype(-1), Dtype(3), Dtype(-10), Dtype(4), Dtype(4), Dtype(10), Dtype(0), Dtype(9), Dtype(0), Dtype(-8), Dtype(-8), 
      Dtype(-3), Dtype(-1), Dtype(-1), Dtype(-7), Dtype(1), Dtype(3), Dtype(-5), Dtype(0), Dtype(5), Dtype(-4), Dtype(7), Dtype(9), Dtype(-3), Dtype(-1), Dtype(10), Dtype(5), Dtype(1), Dtype(0), 
      Dtype(8), Dtype(-1), Dtype(7), Dtype(-4), Dtype(-1), Dtype(-5), Dtype(-4), Dtype(3), Dtype(10), Dtype(-4), Dtype(-5), Dtype(3), Dtype(8), Dtype(-3), Dtype(-9), Dtype(6), Dtype(-7), Dtype(7), 
      Dtype(-8), Dtype(-9), Dtype(5), Dtype(3), Dtype(-1), Dtype(3), Dtype(-8), Dtype(3), Dtype(2), Dtype(-6), Dtype(8), Dtype(6), Dtype(4), Dtype(-6), Dtype(-9), Dtype(4), Dtype(9), Dtype(-4), 
      Dtype(-6), Dtype(-3), Dtype(-3), Dtype(3), Dtype(-8), Dtype(-6), Dtype(-9), Dtype(-3), Dtype(-7), Dtype(-7), Dtype(6), Dtype(-1), Dtype(-3), Dtype(-6), Dtype(-7), Dtype(2), Dtype(2), Dtype(-2), 
      Dtype(-4), Dtype(4), Dtype(2), Dtype(-2), Dtype(-9), Dtype(-9), Dtype(-9), Dtype(10), Dtype(-7), Dtype(5), Dtype(4), Dtype(-9), Dtype(-5), Dtype(6), Dtype(-3), Dtype(-3), Dtype(-9), Dtype(-2), 
      Dtype(-8), Dtype(7), Dtype(9), Dtype(-4), Dtype(1), Dtype(-3), Dtype(4), Dtype(6), Dtype(8), Dtype(-9), Dtype(-7), Dtype(4), Dtype(5), Dtype(7), Dtype(4), Dtype(7), Dtype(7), Dtype(4), 
      Dtype(4), Dtype(7), Dtype(-7), Dtype(-6), Dtype(-2), Dtype(-2), Dtype(0), Dtype(8), Dtype(-4), Dtype(3), Dtype(-6), Dtype(10), Dtype(-3), Dtype(3), Dtype(2), Dtype(-2), Dtype(-3), Dtype(-2), 
      Dtype(2), Dtype(1), Dtype(-3), Dtype(-6), Dtype(-10), Dtype(7), Dtype(-7), Dtype(-4), Dtype(5), Dtype(-9), Dtype(-6), Dtype(-2), Dtype(2), Dtype(6), Dtype(10), Dtype(7), Dtype(1), Dtype(5)
    };
		      
    Dtype feature2_data[64] = {
      Dtype(10), Dtype(2), Dtype(-1), Dtype(-6), Dtype(-2), Dtype(-7), Dtype(-6), Dtype(9), 
      Dtype(-10), Dtype(7), Dtype(3), Dtype(0), Dtype(5), Dtype(-6), Dtype(-8), Dtype(-7), 
      Dtype(-6), Dtype(7), Dtype(6), Dtype(4), Dtype(4), Dtype(3), Dtype(6), Dtype(9), 
      Dtype(3), Dtype(-2), Dtype(6), Dtype(-3), Dtype(4), Dtype(5), Dtype(-3), Dtype(-10), 
      Dtype(9), Dtype(-6), Dtype(-3), Dtype(-10), Dtype(1), Dtype(-2), Dtype(-8), Dtype(4), 
      Dtype(1), Dtype(-10), Dtype(0), Dtype(0), Dtype(0), Dtype(-10), Dtype(-10), Dtype(5), 
      Dtype(1), Dtype(-2), Dtype(1), Dtype(3), Dtype(8), Dtype(6), Dtype(-3), Dtype(-10), 
      Dtype(-6), Dtype(7), Dtype(-6), Dtype(2), Dtype(5), Dtype(-8), Dtype(-9), Dtype(6) 
    };
    for (int i = 0; i < blob_bottom_gt->count(); ++i){ 
      blob_bottom_gt->mutable_cpu_data()[i] = landmark_gt[i];
    }
   
    for (int i = 0; i < blob_bottom_pre->count(); ++i){
      blob_bottom_pre->mutable_cpu_data()[i] = landmark_pre[i];
    }

    for (int i = 0; i < blob_bottom_feature1->count(); ++i){
      blob_bottom_feature1->mutable_cpu_data()[i] = feature1_data[i%324];
    }
    for (int i = 0; i < blob_bottom_feature2->count(); ++i){
      blob_bottom_feature2->mutable_cpu_data()[i] = feature2_data[i%64];
    }
    blob_bottom_vec.push_back(blob_bottom_gt);
    blob_bottom_vec.push_back(blob_bottom_pre);
    blob_bottom_vec.push_back(blob_bottom_feature1);
    blob_bottom_vec.push_back(blob_bottom_feature2);
    blob_top_vec.push_back(blob_top);

    // 
    LayerParameter layer_param;
    PatchSemanticParameter* patch_param = 
        layer_param.mutable_patch_semantic_param();
    patch_param->set_num_landmark(5);
    patch_param->set_image_w(20);
    patch_param->set_image_h(20);
    patch_param->set_patch_w(3);
    patch_param->set_patch_h(3);

    SpatialOperationParameter* sp_param1 = patch_param->add_spatial_operation_param();
    sp_param1->set_kernel_size(3);
    sp_param1->set_stride(1);
    sp_param1->set_weight(0.4);
    sp_param1->set_name("sp1");
    sp_param1->set_used(true);
    SpatialOperationParameter* sp_param2 = patch_param->add_spatial_operation_param();
    sp_param2->set_kernel_size(4);
    sp_param2->set_stride(2);
    sp_param2->set_weight(0.6);
    sp_param2->set_name("sp2");
    sp_param2->set_used(true);

    PatchSemanticLayer<Dtype> layer_weight(layer_param);
    layer_weight.SetUp(blob_bottom_vec, blob_top_vec);
    layer_weight.Forward(blob_bottom_vec, blob_top_vec);
    
    const Dtype expected_Ex[10] = {Dtype(2.4), Dtype(1.3), Dtype(-0.65), Dtype(-4.6), Dtype(-0.8), Dtype(-4.35), Dtype(0.25), Dtype(2.5), Dtype(0.1), Dtype(2.15)};
    const Dtype expected_Ey[10] = {Dtype(-0.8), Dtype(-4.8), Dtype(-6.85), Dtype(-3.4), Dtype(-0.3), Dtype(2.95), Dtype(1.25), Dtype(2.3), Dtype(0), Dtype(-0.75)};
    const Dtype expected_Et[10] = {Dtype(0.6), Dtype(0.65), Dtype(-1.55), Dtype(-2.0), Dtype(2.35), Dtype(-3.45), Dtype(1.05), Dtype(0.5), Dtype(-3.05), Dtype(-2.4)};
/*
    const Dtype expected_Ex[10] = {Dtype(-1.95), Dtype(1.9), Dtype(0.7), Dtype(-3.25), Dtype(2.2), Dtype(-2.55), Dtype(0.25), Dtype(-0.5), Dtype(4.0), Dtype(4.85)};
    const Dtype expected_Ey[10] = {Dtype(-1.85), Dtype(-0.6), Dtype(0.8), Dtype(0.95), Dtype(-1.8), Dtype(1.15), Dtype(1.25), Dtype(-1.9), Dtype(-2.7), Dtype(3.75)};
    const Dtype expected_Et[10] = {Dtype(2.55), Dtype(-2.5), Dtype(-2.9), Dtype(-2.45), Dtype(1.9), Dtype(-2.55), Dtype(1.05), Dtype(0.5), Dtype(1.6), Dtype(1.65)};*/
    const Dtype error_margin = 1e-2;
    for (int b = 0; b < 2; ++b) {
      for (int l = 0; l < 5; ++l) {
        Dtype Ex = blob_top_vec[0]->cpu_data()[blob_top_vec[0]->offset(b, l)];
        Dtype Ey = blob_top_vec[0]->cpu_data()[blob_top_vec[0]->offset(b, l, 1)];
        Dtype Et = blob_top_vec[0]->cpu_data()[blob_top_vec[0]->offset(b, l, 2)];
        EXPECT_NEAR(expected_Ex[b*5 + l], Ex, error_margin);
        EXPECT_NEAR(expected_Ey[b*5 + l], Ey, error_margin);
        EXPECT_NEAR(expected_Et[b*5 + l], Et, error_margin);
      }
    }
    

    
    // std::cout << "epsilon = " << loss_param->epsilon() << std::endl;
    // std::cout << "num_landmark = " << loss_param->num_landmark() << std::endl;

    delete blob_bottom_gt;
    delete blob_bottom_pre;
    delete blob_bottom_feature1;
    delete blob_bottom_feature2;
    delete blob_top;
  }
  
  void TestForward(){
    // Get the loss without a specified objective weight -- should be 
    // equivalent to explicitly specifying a weight of 1
    // LayerParameter layer_param;
    // PerceptualLossParameter* loss_param = 
    //     layer_param.mutable_perceptual_loss_param();
    // loss_param->set_num_landmark(21);
    // loss_param->set_epsilon(0.05);
    
    // PerceptualLossLayer<Dtype> layer_weight_1(layer_param);
    // layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // const Dtype loss_weight_1 = 
    //     layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // // Get the loss again with a different objective weight; check that it is scaled appropriately
    // const Dtype kLossWeight = 3.7;
    // layer_param.add_loss_weight(kLossWeight);
    
    // PerceptualLossLayer<Dtype> layer_weight_k(layer_param);
    // layer_weight_k.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // const Dtype loss_weight_k = 
    //     layer_weight_k.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // const Dtype kErrorMargin = 1e-5;
    // EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_k, kErrorMargin);
    // // make sure that the loss is non-trival
    // const Dtype kNonTrivalAbsThresh = 1e-1;
    // EXPECT_GE(fabs(loss_weight_1), kNonTrivalAbsThresh);
  }
  
  Blob<Dtype>* const blob_bottom_gt_;
  Blob<Dtype>* const blob_bottom_pre_;
  Blob<Dtype>* const blob_bottom_feature1_;
  Blob<Dtype>* const blob_bottom_feature2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
};

TYPED_TEST_CASE(PatchSemanticLayerTest, TestDtypesGPU);

// TYPED_TEST(PatchSemanticLayerTest, TestForward){
//   this->TestForward();
// }


// TYPED_TEST(PerceptualLossLayerTest, TestGradient){
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   PerceptualLossParameter* loss_param = 
//         layer_param.mutable_perceptual_loss_param();
//   loss_param->set_num_landmark(21);
//   loss_param->set_epsilon(0.05);
  
//   const Dtype kLossWeight = 3.7;
//   layer_param.add_loss_weight(kLossWeight);
//   PerceptualLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);

// }

TYPED_TEST(PatchSemanticLayerTest, TestForwardSpecifiedExample){
  this->TestForwardSpecifiedExample();
}


} // namespace caffe
