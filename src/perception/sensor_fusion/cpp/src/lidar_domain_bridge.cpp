// Read-only cross-domain PointCloud2 forwarding. Never opens a sensor socket.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <thread>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <cerrno>
#include <rclcpp/serialization.hpp>
using Cloud=sensor_msgs::msg::PointCloud2;
static std::atomic_bool running{true};
static void stop(int){running=false;}
// Separate processes are essential: Humble/Fast DDS caches transport/locality state.
// Local socketpair only: no TCP/UDP listening port, no source-side publisher.
static bool transfer(int fd,void* data,size_t size,bool writing){
 auto* bytes=static_cast<char*>(data);
 while(size&&running){ssize_t n=writing?send(fd,bytes,size,MSG_NOSIGNAL):recv(fd,bytes,size,0);
  if(n>0){bytes+=n;size-=n;continue;}
  if(n<0&&errno==EINTR)continue;
  if(n<0&&!writing&&(errno==EAGAIN||errno==EWOULDBLOCK))continue;
  return false;
 }return size==0;
}
static int sink(int fd,int domain){
 setenv("ROS_LOCALHOST_ONLY","1",1);
 setenv("ROS_DOMAIN_ID",std::to_string(domain).c_str(),1);
 if(const char* profile=getenv("A1_BRIDGE_TARGET_PROFILE"))setenv("FASTRTPS_DEFAULT_PROFILES_FILE",profile,1);
 auto context=std::make_shared<rclcpp::Context>();rclcpp::InitOptions opt;opt.set_domain_id(domain);context->init(0,nullptr,opt);
 auto node=std::make_shared<rclcpp::Node>("a1_lidar_bridge_sink",rclcpp::NodeOptions().context(context));
 std::cout<<"Sink context domain="<<context->get_domain_id()<<" localhost="<<getenv("ROS_LOCALHOST_ONLY")<<std::endl;
 auto pub=node->create_publisher<Cloud>("/lead_fusion/input/lidar_points",rclcpp::SensorDataQoS().keep_last(2));
 rclcpp::Serialization<Cloud> serializer;int forwarded=0;
 while(running){uint32_t size=0;if(!transfer(fd,&size,sizeof(size),false))break;
  if(size==0||size>32*1024*1024)throw std::runtime_error("invalid IPC cloud size");
  rclcpp::SerializedMessage wire(size);auto& raw=wire.get_rcl_serialized_message();
  if(!transfer(fd,raw.buffer,size,false))break;
  raw.buffer_length=size;
  Cloud cloud;serializer.deserialize_message(&wire,&cloud);pub->publish(cloud);if(++forwarded%30==0)std::cout<<"sink forwarded="<<forwarded<<" subscribers="<<pub->get_subscription_count()<<std::endl;
 }
 context->shutdown("sink stopped");close(fd);return 0;
}
struct Child {pid_t pid{-1};int fd{-1};~Child(){if(fd>=0)close(fd);if(pid>0){kill(pid,SIGTERM);while(waitpid(pid,nullptr,0)<0&&errno==EINTR){}}}};
int main(int argc,char** argv){
 try{
  std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
  if(argc==4&&std::string(argv[1])=="--sink")return sink(std::stoi(argv[2]),std::stoi(argv[3]));
  if(argc<4||argc>5)throw std::runtime_error("usage: lidar_domain_bridge SOURCE_DOMAIN SOURCE_TOPIC DEST_DOMAIN [--check]");
  auto domain=[](const char* s){std::string v(s);size_t end=0;int d=std::stoi(v,&end);if(end!=v.size()||d<0||d>232)throw std::runtime_error("domain must be 0..232");return d;};
  int src=domain(argv[1]),dst=domain(argv[3]);std::string topic=argv[2];bool check=argc==5;
  if((check&&std::string(argv[4])!="--check")||src==dst||topic.empty()||topic[0]!='/')throw std::runtime_error("distinct domains and an absolute source topic required");
  std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
  Child child;
  if(!check){int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)!=0)throw std::runtime_error("socketpair failed");
   timeval timeout{0,250000};for(int fd:pair){setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));}
   const pid_t parent=getpid();child.pid=fork();
   if(child.pid<0){close(pair[0]);close(pair[1]);throw std::runtime_error("fork failed");}
   if(child.pid==0){close(pair[0]);prctl(PR_SET_PDEATHSIG,SIGTERM);if(getppid()!=parent)_exit(1);
    setenv("ROS_LOCALHOST_ONLY","1",1);
    if(const char* profile=getenv("A1_BRIDGE_TARGET_PROFILE"))setenv("FASTRTPS_DEFAULT_PROFILES_FILE",profile,1);
    std::string fd=std::to_string(pair[1]),domain_arg=std::to_string(dst);
    execl("/proc/self/exe","lidar_domain_bridge","--sink",fd.c_str(),domain_arg.c_str(),static_cast<char*>(nullptr));_exit(1);
   }
   close(pair[1]);child.fd=pair[0];
  }
  setenv("ROS_LOCALHOST_ONLY","0",1);
  setenv("ROS_DOMAIN_ID",std::to_string(src).c_str(),1);
  auto source=std::make_shared<rclcpp::Context>();rclcpp::InitOptions si;si.set_domain_id(src);source->init(0,nullptr,si);
  std::cout<<"Source context domain="<<source->get_domain_id()<<std::endl;
  auto sn=std::make_shared<rclcpp::Node>("a1_lidar_readonly_bridge_source",rclcpp::NodeOptions().context(source));
  rclcpp::Serialization<Cloud> serializer;bool ipc_failed=false;
  using Clock=std::chrono::steady_clock;auto start=Clock::now(),last=start,log=start;
  int received=0;int64_t stamp=-1;
  auto sub=sn->create_subscription<Cloud>(topic,rclcpp::SensorDataQoS().keep_last(2),[&](Cloud::ConstSharedPtr m){
   if(sn->count_publishers(topic)!=1)return; // Refuse mixed sources even after startup.
   int64_t ns=int64_t(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec;
   if(m->header.frame_id!="hesai_lidar"||m->width==0||m->height==0||m->point_step==0||m->row_step<uint64_t(m->width)*m->point_step||m->data.size()<uint64_t(m->height)*m->row_step)return;
   if(ns<=0||ns<=stamp)return; // Duplicate / backward sensor time is not fresh input.
   stamp=ns;last=Clock::now();++received;
   if(child.fd>=0){rclcpp::SerializedMessage wire;serializer.serialize_message(m.get(),&wire);auto& raw=wire.get_rcl_serialized_message();
    uint32_t size=raw.buffer_length;
    if(raw.buffer_length>32*1024*1024||!transfer(child.fd,&size,sizeof(size),true)||!transfer(child.fd,raw.buffer,size,true)){ipc_failed=true;running=false;}
   } // All fields and bytes unchanged.
  });
  rclcpp::ExecutorOptions eo;eo.context=source;rclcpp::executors::SingleThreadedExecutor ex(eo);ex.add_node(sn);
  std::cout<<"Read-only LiDAR bridge: domain "<<src<<" "<<topic<<" -> domain "<<dst<<" /lead_fusion/input/lidar_points; no driver ownership"<<std::endl;
  int result=0;
  while(running){ex.spin_some();auto now=Clock::now();auto n=sn->count_publishers(topic);
   if(check&&n>1){std::cerr<<"Multiple source publishers; refusing mixed LiDAR input\n";result=3;break;}
   if(check&&received>=3&&n==1){std::cout<<"PASS: one publisher, three increasing-stamp hesai_lidar clouds received\n";break;}
   if(check&&now-start>std::chrono::seconds(6)){std::cerr<<"No verified LiDAR stream: start the localization/shared driver first. Publishers="<<n<<", valid clouds="<<received<<"\n";result=2;break;}
   if(!check&&now-log>std::chrono::seconds(3)){std::cout<<"publishers="<<n<<" received="<<received<<" state="<<(n==1&&received>0&&now-last<std::chrono::milliseconds(500)?"LIVE":"WAITING_OR_STALE")<<std::endl;log=now;}
   std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  source->shutdown("bridge stopped");return ipc_failed?4:result;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
