#include <torch/script.h>
#include <ATen/Parallel.h>
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include "a1_fusion_cpp/msg/a2_features.hpp"
#include <deque>
#include <set>

class CameraNode: public rclcpp::Node {
 public:
  CameraNode():Node("a1_a2_camera"),device_(declare_parameter<std::string>("device","cpu")) {
    at::set_num_threads(declare_parameter("cpu_threads",2));
    cv::setNumThreads(1);
    // The unrolled 21-frame head makes profiling-executor specialization very slow.
    // Use the regular graph executor consistently with the export validation.
    torch::jit::getProfilingMode() = false;
    torch::jit::getExecutorMode() = false;
    const auto dir=declare_parameter<std::string>("model_dir","");
    if(dir.empty())throw std::invalid_argument("model_dir is required; run export_a2.py first");
    tokenizer_=torch::jit::load(dir+"/tokenizer.pt",device_);head_=torch::jit::load(dir+"/a2_head.pt",device_);
    tokenizer_.eval();head_.eval();
    if(device_.is_cuda()) {
      if(!torch::cuda::is_available())throw std::runtime_error("CUDA requested but unavailable");
      at::globalContext().setBenchmarkCuDNN(true);
      at::globalContext().setAllowTF32CuDNN(false);
      at::globalContext().setAllowTF32CuBLAS(false);
    }
    // Compile graphs/select convolution algorithms before accepting live frames.
    {
      torch::InferenceMode guard;
      auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device_);
      auto x=torch::zeros({1,3,250,1280},opts);
      const int warmup=declare_parameter("warmup_frames",10);
      for(int i=0;i<warmup;++i) {
        auto t=tokenizer_.forward({x}).toTensor();
        head_.forward({t.unsqueeze(0).repeat({21,1,1,1}),torch::zeros({1,2},opts)}).toTensor().to(torch::kCPU);
      }
    }
    RCLCPP_INFO(get_logger(),"Inference ready on %s",device_.str().c_str());
    roi_=declare_parameter("roi_y0",235);
    pub_=create_publisher<a1_fusion_cpp::msg::A2Features>("/lead_fusion/a2",rclcpp::SensorDataQoS().keep_last(2));
    sub_=create_subscription<sensor_msgs::msg::Image>(declare_parameter<std::string>("image_topic","/camera_driver_gv/vis/image_raw"),rclcpp::SensorDataQoS().keep_last(4),[this](sensor_msgs::msg::Image::ConstSharedPtr m){process(m);});
  }
 private:
  void process(sensor_msgs::msg::Image::ConstSharedPtr m) {
    try {
      torch::InferenceMode guard;
      RCLCPP_INFO_ONCE(get_logger(),"First image received; initializing inference graphs");
      const double stamp=rclcpp::Time(m->header.stamp).seconds();
      if(!history_.empty() && (stamp<=history_.back().first||stamp-history_.back().first>.25))history_.clear();
      if(m->width!=1280||m->height!=720||roi_<0||roi_+250>720)throw std::runtime_error("expected 1280x720, ROI height 250");
      auto bgr=cv_bridge::toCvCopy(m,"bgr8")->image;
      cv::Mat rgb;cv::cvtColor(bgr(cv::Rect(0,roi_,1280,250)),rgb,cv::COLOR_BGR2RGB);
      auto input=torch::from_blob(rgb.data,{250,1280,3},torch::kUInt8).to(torch::kFloat32).permute({2,0,1}).unsqueeze(0)/255.0;
      auto mean=torch::tensor({.485f,.456f,.406f}).view({1,3,1,1});
      auto std=torch::tensor({.229f,.224f,.225f}).view({1,3,1,1});
      input=((input-mean)/std).contiguous().to(device_);
      auto token=tokenizer_.forward({input}).toTensor();
      history_.emplace_back(stamp,token);while(history_.size()>128)history_.pop_front();
      std::vector<torch::Tensor> tokens{token};std::set<size_t> unique;
      bool ready=stamp-history_.front().first>=.98;
      for(int i=1;i<=20;++i) {
        size_t best=0;double error=1e9;
        for(size_t j=0;j<history_.size();++j){double d=std::abs(stamp-history_[j].first-i*.05);if(d<error){error=d;best=j;}}
        tokens.push_back(history_[best].second);unique.insert(best);ready=ready&&error<=.035;
      }
      ready=ready&&unique.size()==20;
      // Preserve training's zero ego_motion feature; navigation speed is used downstream.
      auto out=head_.forward({torch::stack(tokens),torch::zeros({1,2},input.options())}).toTensor().to(torch::kCPU).contiguous();
      if(out.numel()!=280||!torch::isfinite(out).all().item<bool>())throw std::runtime_error("invalid A2 shape or nonfinite output");
      a1_fusion_cpp::msg::A2Features msg;msg.header=m->header;msg.history_ready=ready;
      std::copy_n(out.data_ptr<float>(),280,msg.raw.begin());pub_->publish(msg);
    } catch(const std::exception& e) {
      history_.clear();RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,"camera: %s",e.what());
    }
  }
  torch::Device device_;torch::jit::Module tokenizer_,head_;int roi_{};
  std::deque<std::pair<double,torch::Tensor>> history_;
  rclcpp::Publisher<a1_fusion_cpp::msg::A2Features>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<CameraNode>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}rclcpp::shutdown();}
