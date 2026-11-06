#pragma once
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <future>
#include <string>

namespace a1::inspection {
// The camera's dynamic ExposureTime feature uses microseconds (15..1,000,000).
// Poll futures from the UI thread: never block the perception executor.
class ExposureControl {
  using Get = rcl_interfaces::srv::GetParameters;
  using Set = rcl_interfaces::srv::SetParameters;
  using Clock = std::chrono::steady_clock;
 public:
  explicit ExposureControl(rclcpp::Node& node) {
    auto remote=node.declare_parameter<std::string>("camera_control_node","/camera_driver_gv");
    get_=node.create_client<Get>(remote+"/get_parameters");
    set_=node.create_client<Set>(remote+"/set_parameters");
  }
  ~ExposureControl(){if(reading_)get_->remove_pending_request(read_id_);if(writing_)set_->remove_pending_request(write_id_);}
  static bool parse(const std::string& text,double& value) {
    try {size_t end;value=std::stod(text,&end);return end==text.size()&&std::isfinite(value)&&value>=15.&&value<=1000000.;}
    catch(...) {return false;}
  }
  void tick() {
    auto now=Clock::now();
    if(writing_){
      if(write_.wait_for(std::chrono::seconds(0))==std::future_status::ready){
        auto result=write_.get();writing_=false;
        status_=result->results.size()==1&&result->results[0].successful?"Accepted; reading back":"Rejected by camera";
        if(result->results.size()==1&&!result->results[0].successful)status_+=": "+result->results[0].reason;
        next_read_=now;
      }else if(now-write_start_>std::chrono::seconds(3)){
        set_->remove_pending_request(write_id_);writing_=false;status_="Set timed out; checking camera";next_read_=now;
      }
    }
    if(reading_){
      if(read_.wait_for(std::chrono::seconds(0))==std::future_status::ready){
        auto result=read_.get();reading_=false;
        if(result->values.size()==1&&result->values[0].type==rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE){
          actual_=result->values[0].double_value;last_read_=now;
          if(!editing_&&!writing_)input_=cv::format("%.0f",actual_);
        }
      }else if(now-read_start_>std::chrono::seconds(3)){get_->remove_pending_request(read_id_);reading_=false;}
    }
    if(!reading_&&!writing_&&now>=next_read_){
      next_read_=now+std::chrono::seconds(1);
      if(get_->service_is_ready()){
        auto req=std::make_shared<Get::Request>();req->names={"ExposureTime"};
        auto pending=get_->async_send_request(req);read_=pending.future.share();read_id_=pending.request_id;reading_=true;read_start_=now;
      }
    }
  }
  void apply(){
    double value;
    if(writing_){status_="Waiting for camera";return;}
    if(!parse(input_,value)){status_="Enter 15..1000000 us";return;}
    if(!set_->service_is_ready()){status_="Camera service unavailable";return;}
    // Discard older readback so it cannot replace the new input.
    if(reading_){get_->remove_pending_request(read_id_);reading_=false;}
    auto req=std::make_shared<Set::Request>();rcl_interfaces::msg::Parameter p;
    p.name="ExposureTime";p.value.type=rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;p.value.double_value=value;req->parameters={p};
    auto pending=set_->async_send_request(req);write_=pending.future.share();write_id_=pending.request_id;writing_=true;write_start_=Clock::now();
    editing_=false;status_="Applying "+input_+" us";
  }
  void mouse(int event,int x,int y){
    if(event!=cv::EVENT_LBUTTONDOWN)return;
    if(field_.contains({x,y})){editing_=true;replace_=true;return;}
    if(apply_.contains({x,y})){apply();return;}
    if(minus_.contains({x,y})||plus_.contains({x,y})){
      double value;if(!parse(input_,value)){if(!fresh())return;value=actual_;}
      value=std::clamp(value+(plus_.contains({x,y})?1000.:-1000.),15.,1000000.);
      input_=cv::format("%.0f",value);apply();return;
    }
    editing_=false;
  }
  void key(int key){
    if(key<0)return;
    if(key=='e'||key=='E'){editing_=true;replace_=true;return;}
    if(!editing_)return;
    if(key==13||key==10){apply();return;}
    if(key==27){editing_=false;input_=fresh()?cv::format("%.0f",actual_):"";return;}
    if(key==8||key==127){if(replace_)input_.clear();else if(!input_.empty())input_.pop_back();replace_=false;return;}
    if((key>='0'&&key<='9')||key=='.'){
      if(replace_)input_.clear();replace_=false;
      if(input_.size()<12&&(key!='.'||input_.find('.')==std::string::npos))input_+=char(key);
    }
  }
  void draw(cv::Mat& panel){
    tick();cv::rectangle(panel,{12,95,1000,90},{25,30,38},-1);
    auto text=[&](const std::string& s,int x,int y,cv::Scalar color=cv::Scalar(230,230,230)){cv::putText(panel,s,{x,y},cv::FONT_HERSHEY_SIMPLEX,.47,color,1,cv::LINE_AA);};
    text("Exposure (us)",22,122);
    for(auto r:{field_,apply_,minus_,plus_})cv::rectangle(panel,r,{70,80,95},1);
    if(editing_)cv::rectangle(panel,field_,{0,220,255},2);
    text(input_+(editing_?"|":""),178,122);text("Apply",340,122);text("-1ms",425,122);text("+1ms",505,122);
    text(fresh()?cv::format("Actual: %.0f us (%.1f ms)",actual_,actual_/1000.):"Actual: unavailable / stale",595,122,{100,230,180});
    text(status_.substr(0,115),22,149);text("Click value or E; Enter = apply. Long exposure may reduce FPS.",22,173,{160,170,185});
  }
  double actual()const{return actual_;}
 private:
  bool fresh()const{return std::isfinite(actual_)&&Clock::now()-last_read_<std::chrono::seconds(4);}
  rclcpp::Client<Get>::SharedPtr get_;rclcpp::Client<Set>::SharedPtr set_;
  rclcpp::Client<Get>::SharedFuture read_;rclcpp::Client<Set>::SharedFuture write_;
  int64_t read_id_{},write_id_{};bool reading_{},writing_{},editing_{},replace_{};
  Clock::time_point next_read_{},read_start_{},write_start_{},last_read_{};
  double actual_=std::numeric_limits<double>::quiet_NaN();
  std::string input_,status_="Manual exposure / camera readback";
  cv::Rect field_{170,101,150,30},apply_{330,101,75,30},minus_{415,101,70,30},plus_{495,101,70,30};
};
}
