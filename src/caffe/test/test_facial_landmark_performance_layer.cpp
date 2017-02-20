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
class FacialLandmarkPerformanceLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;
 
 protected:
  // num_landmark = 5
  // 2 * num_landmark = 10
  FacialLandmarkPerformanceLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(4, 10, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(4, 10, 1, 1)),
        blob_bottom_normalizer_(new Blob<Dtype>(4, 1, 1, 1)),
        blob_top_average_error_(new Blob<Dtype>()), 
	blob_top_false_rate_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter gaussian_filler_param;
    GaussianFiller<Dtype> gaussian_filler(gaussian_filler_param);
    
    gaussian_filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    
    gaussian_filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    
    gaussian_filler.Fill(this->blob_bottom_normalizer_);
    blob_bottom_vec_.push_back(blob_bottom_normalizer_);

    
    blob_top_vec_.push_back(blob_top_average_error_);
    blob_top_vec_.push_back(blob_top_false_rate_);
  }
  
  virtual ~FacialLandmarkPerformanceLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_normalizer_;
    delete blob_top_average_error_;
    delete blob_top_false_rate_;
  } 

  void TestForwardSpecifiedExample(){
    Blob<Dtype>* blob_bottom_data = new Blob<Dtype>(2, 10, 1, 1);
    Blob<Dtype>* blob_bottom_label = new Blob<Dtype>(2, 10, 1, 1);
    Blob<Dtype>* blob_bottom_normalizer = new Blob<Dtype>(2, 1, 1, 1);
    Blob<Dtype>* blob_top_average_error = new Blob<Dtype>();
    Blob<Dtype>* blob_top_false_rate = new Blob<Dtype>();
    
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    // 
    Dtype data[20] = { Dtype(10.5), Dtype(10.5),
                       Dtype(15.5), Dtype(10.5),
                       Dtype(13), Dtype(13),
                       Dtype(11.5), Dtype(15),
                       Dtype(17), Dtype(16),
                       
                       Dtype(10.5), Dtype(10.5),
                       Dtype(14.5), Dtype(10.5),
                       Dtype(13), Dtype(13),
                       Dtype(11.5), Dtype(15),
                       Dtype(17), Dtype(16)};
    // 
    Dtype label[20] = {Dtype(9.5), Dtype(9.5), 
                      Dtype(13), Dtype(9.5), 
                      Dtype(11), Dtype(11), 
                      Dtype(10), Dtype(13), 
                      Dtype(15), Dtype(12), 
                      
                      Dtype(7), Dtype(7), 
                      Dtype(17), Dtype(7), 
                      Dtype(13), Dtype(13), 
                      Dtype(7.5), Dtype(17),
                      Dtype(15), Dtype(19)};
    // 
    Dtype normalizer[2] = {Dtype(5), Dtype(4)};
    
    for (int i = 0; i < blob_bottom_normalizer->count(); ++i){
      blob_bottom_normalizer->mutable_cpu_data()[i] = normalizer[i];
    }
		      
    for (int i = 0; i < blob_bottom_label->count(); ++i){ 
      blob_bottom_label->mutable_cpu_data()[i] = label[i];
    }
   
    for (int i = 0; i < blob_bottom_data->count(); ++i){
      blob_bottom_data->mutable_cpu_data()[i] = data[i];
    }
   
    blob_bottom_vec.push_back(blob_bottom_data);
    blob_bottom_vec.push_back(blob_bottom_label);
    blob_bottom_vec.push_back(blob_bottom_normalizer);
    blob_top_vec.push_back(blob_top_average_error);
    blob_top_vec.push_back(blob_top_false_rate);

    // 
    LayerParameter layer_param;
    FacialLandmarkPerformanceParameter* param = 
        layer_param.mutable_facial_landmark_performance_param();
    param->set_num_landmark(5);
    param->set_error_threshold(0.05);

    FacialLandmarkPerformanceLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec); 
    layer.Forward(blob_bottom_vec, blob_top_vec);
    
    // print and test 
    const Dtype error_margin = 1e-2;
    Dtype average_error[5] = {Dtype(0.76014), Dtype(0.8069035), Dtype(0.2828425), Dtype(0.809017), Dtype(0.8979075)};
    std::cout << "#####average error of each landmakr#####" << std::endl;
    for (int i = 0; i < blob_top_average_error->count(); ++i) {
      std::cout << blob_top_average_error->cpu_data()[i] << ",";
      EXPECT_NEAR(average_error[i], blob_top_average_error->cpu_data()[i], error_margin);
    }
    std::cout << std::endl;
    
    const Dtype false_margin = 1e-4;
    Dtype false_rate[5] = {Dtype(1), Dtype(1), Dtype(0.5), Dtype(1), Dtype(1)};
    std::cout << "#####false rate of each landmark#####" << std::endl;
    for (int i = 0; i < blob_top_false_rate->count(); ++i) {
      std::cout << blob_top_false_rate->cpu_data()[i] << ",";
      EXPECT_NEAR(false_rate[i], blob_top_false_rate->cpu_data()[i], false_margin);
    }
    std::cout << std::endl;
    
    std::cout << "error_threshold = " << param->error_threshold() << std::endl;
    std::cout << "num_landmark = " << param->num_landmark() << std::endl;

    delete blob_bottom_data;
    delete blob_bottom_label;
    delete blob_bottom_normalizer;
    delete blob_top_average_error;
    delete blob_top_false_rate;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_normalizer_;
  Blob<Dtype>* const blob_top_average_error_;
  Blob<Dtype>* const blob_top_false_rate_;
  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
};

TYPED_TEST_CASE(FacialLandmarkPerformanceLayerTest, TestDtypesGPU);

TYPED_TEST(FacialLandmarkPerformanceLayerTest, TestForwardSpecifiedExample){
  this->TestForwardSpecifiedExample();
}


} // namespace caffe
