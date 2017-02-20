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
class ChannelExpandLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;
 
 protected:
  // num_landmark = 21
  // 2 * num_landmark = 42
  // 3 * num_landmark = 63
  ChannelExpandLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 1, 18, 18)),
        blob_top_(new Blob<Dtype>(2, 3, 18, 18))

        {
    // fill the values

    FillerParameter uniform_filler_param;

    uniform_filler_param.set_min(-10);
    uniform_filler_param.set_max(10);
    UniformFiller<Dtype> feature_filler(uniform_filler_param);
    feature_filler.Fill(this->blob_bottom_);
  
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  
  virtual ~ChannelExpandLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  } 

  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom = new Blob<Dtype>(2, 1, 18, 18);
    Blob<Dtype>* blob_top = new Blob<Dtype>();
    
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;

    Dtype data[324] = {
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
		      

    for (int i = 0; i < blob_bottom->count(); ++i){ 
      blob_bottom->mutable_cpu_data()[i] = data[i%324];
    }
   
    blob_bottom_vec.push_back(blob_bottom);
    blob_top_vec.push_back(blob_top);

    // 
    LayerParameter layer_param;
    ChannelExpandParameter* ce_param = 
        layer_param.mutable_channel_expand_param();
    ce_param->set_num_channel(3);


    ChannelExpandLayer<Dtype> layer_weight(layer_param);
    layer_weight.SetUp(blob_bottom_vec, blob_top_vec);
    layer_weight.Forward(blob_bottom_vec, blob_top_vec);
    
    const Dtype error_margin = 1e-2;
    for (int b = 0; b < 2; ++b){    
     for (int n = 0; n < 3; ++n){
      for(int h = 0; h < 18; ++h){
        for(int w = 0; w < 18; ++w){
          Dtype bt = blob_bottom->cpu_data()[blob_bottom->offset(b,0,h,w)];
          Dtype tp = blob_top_vec[0]->cpu_data()[blob_top_vec[0]->offset(b,n,h,w)];
          EXPECT_NEAR(bt, tp, error_margin);
        }
      }
    }
  }
    

    delete blob_bottom;
    delete blob_top;
  }
  
  void TestForward(){

  }
  
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
};

TYPED_TEST_CASE(ChannelExpandLayerTest, TestDtypesGPU);

TYPED_TEST(ChannelExpandLayerTest, TestForwardSpecifiedExample){
  this->TestForwardSpecifiedExample();
}


} // namespace caffe
