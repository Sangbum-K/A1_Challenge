#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include "a1_fusion/map_surfaces.hpp"
class Server:public rclcpp::Node{
 public:Server():Node("a1_map_surface_server"){
  auto file=declare_parameter<std::string>("surface_file","");auto frame=declare_parameter<std::string>("map_frame","map");
  a1::inspection::Surfaces points;if(pcl::io::loadPCDFile(file,points)!=0||points.empty())throw std::runtime_error("invalid map surface cache: "+file);
  sensor_msgs::msg::PointCloud2 msg;pcl::toROSMsg(points,msg);msg.header.frame_id=frame;msg.header.stamp=now();epoch_=msg.header;
  map_=create_publisher<sensor_msgs::msg::PointCloud2>("/localization/map_surfaces",rclcpp::QoS(1).reliable().transient_local());
  beat_=create_publisher<std_msgs::msg::Header>("/localization/map_surfaces_epoch",rclcpp::QoS(1).reliable().transient_local());map_->publish(msg);
  timer_=create_wall_timer(std::chrono::milliseconds(100),[this]{beat_->publish(epoch_);});
  RCLCPP_INFO(get_logger(),"Map surfaces: %zu from %s",points.size(),file.c_str());
 }
 private:std_msgs::msg::Header epoch_;rclcpp::TimerBase::SharedPtr timer_;
 rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_;rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr beat_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<Server>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}rclcpp::shutdown();}
