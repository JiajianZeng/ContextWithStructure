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
class AveragePointLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
    AveragePointLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_top_value_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_value_);
  }
  virtual ~AveragePointLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_value_;
  }

  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom_data = new Blob<Dtype>(2, 10, 1, 1);
    Blob<Dtype>* blob_top_value = new Blob<Dtype>();
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;

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
    
    for (int i = 0; i < blob_bottom_data->count(); ++i){
      blob_bottom_data->mutable_cpu_data()[i] = landmark_pre[i];
    }
   
    blob_bottom_vec.push_back(blob_bottom_data);
    blob_top_vec.push_back(blob_top_value);
    // 
    LayerParameter layer_param;
    AveragePointParameter* param = 
        layer_param.mutable_average_point_param();
    param->set_num_landmark(5);
    AveragePointLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec); 
    layer.Forward(blob_bottom_vec, blob_top_vec);
    
    // print and test 
    const Dtype error_margin = 1e-2;
    Dtype value_1x = Dtype(8);
    Dtype value_1y = Dtype(10.6);
    Dtype value_2x = Dtype(10.2);
    Dtype value_2y = Dtype(11);

    EXPECT_NEAR(value_1x, blob_top_vec[0]->cpu_data()[0], error_margin);
    EXPECT_NEAR(value_1y, blob_top_vec[0]->cpu_data()[1], error_margin);
    EXPECT_NEAR(value_2x, blob_top_vec[0]->cpu_data()[2], error_margin);
    EXPECT_NEAR(value_2y, blob_top_vec[0]->cpu_data()[3], error_margin);
    

    delete blob_bottom_data;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_value_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
  

TYPED_TEST_CASE(AveragePointLayerTest, TestDtypesGPU);

TYPED_TEST(AveragePointLayerTest, TestForwardSpecifiedExample) {
  this->TestForwardSpecifiedExample();
}


TYPED_TEST(AveragePointLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  AveragePointLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe
