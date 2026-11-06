#include "a1_fusion/ui_recorder.hpp"
int main(int argc,char** argv){
 if(argc!=2)return 2;
 {
  UiRecorder recorder(argv[1]);
  cv::Mat frame(860,1600,CV_8UC3,cv::Scalar(20,30,40));
  cv::putText(frame,"SYNTHETIC RECORDING TEST",{100,400},cv::FONT_HERSHEY_SIMPLEX,2,{0,0,255},3);
  recorder.push(frame,123.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(650));
  if(recorder.failed())return 3;
 }
 const auto path=std::string(argv[1])+"/ui_00001.mp4";
 cv::VideoCapture capture(path);cv::Mat frame;int count=0;
 while(capture.read(frame)){if(frame.cols!=1600||frame.rows!=860)return 4;++count;}
 if(count<5||count>9)return 5;
 std::cout<<"MP4 finalized on destructor and decoded "<<count<<" 1600x860 frames: PASS\n";
}
