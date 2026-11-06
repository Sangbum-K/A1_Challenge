#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <iostream>
#include <sstream>

// Independent bounded-buffer recorder: samples the latest complete UI at 10 Hz.
// Each 60-second segment is finalized independently; no unbounded frame queue.
class UiRecorder {
 public:
  explicit UiRecorder(const std::string& dir):dir_(dir){
    std::filesystem::create_directories(dir_);
    log_.open(dir_+"/frames.csv");if(!log_)throw std::runtime_error("cannot create recording index");
    log_<<"segment,frame,wall_unix_s,ui_stamp_s\n";
    open_segment();
    worker_=std::thread([this]{run();});
  }
  ~UiRecorder(){stop_=true;cv_.notify_all();if(worker_.joinable())worker_.join();writer_.release();}
  void push(const cv::Mat& frame,double stamp){std::lock_guard<std::mutex> lock(mutex_);latest_=frame.clone();stamp_=stamp;cv_.notify_all();}
  bool failed()const{return failed_;}
  int segment()const{return segment_;}
 private:
  void open_segment(){
    writer_.release();++segment_;frames_=0;
    std::ostringstream name;name<<dir_<<"/ui_"<<std::setw(5)<<std::setfill('0')<<segment_.load()<<".mp4";
    if(!writer_.open(name.str(),cv::CAP_FFMPEG,cv::VideoWriter::fourcc('m','p','4','v'),10.,{1600,860}))
      throw std::runtime_error("cannot open MP4 recording: "+name.str());
  }
  void run(){
    try{
      auto next=std::chrono::steady_clock::now();
      while(!stop_){
        cv::Mat frame;double stamp;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait_until(lock,next,[this]{return stop_.load();});
          if(stop_)break;
          if(latest_.empty()){next=std::chrono::steady_clock::now()+std::chrono::milliseconds(100);continue;}
          frame=latest_;stamp=stamp_;
        }
        if(frames_==600)open_segment();
        if(frames_%10==0&&std::filesystem::space(dir_).available<128ULL*1024*1024)
          throw std::runtime_error("recording disk space exhausted");
        writer_.write(frame);
        double wall=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        log_<<segment_<<','<<frames_++<<','<<std::setprecision(16)<<wall<<','<<stamp<<'\n';
        if(frames_%10==0)log_.flush();
        next+=std::chrono::milliseconds(100);
        if(std::chrono::steady_clock::now()-next>std::chrono::seconds(1))next=std::chrono::steady_clock::now();
      }
    }catch(const std::exception& e){failed_=true;std::cerr<<"UI recording failed: "<<e.what()<<std::endl;}
    writer_.release();log_.flush();
  }
  std::string dir_;cv::VideoWriter writer_;std::ofstream log_;
  std::thread worker_;std::mutex mutex_;std::condition_variable cv_;cv::Mat latest_;
  double stamp_{};int frames_{};std::atomic<int> segment_{0};std::atomic<bool> stop_{false},failed_{false};
};
