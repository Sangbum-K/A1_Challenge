#include "a1_fusion/exposure_control.hpp"
#include <stdexcept>
#define CHECK(x) do {if(!(x))throw std::runtime_error("check failed: " #x);} while(false)
#include <thread>
int main(int argc,char** argv){
 rclcpp::init(argc,argv);
 auto camera=std::make_shared<rclcpp::Node>("camera_driver_gv");camera->declare_parameter("ExposureTime",35000.);
 auto callback=camera->add_on_set_parameters_callback([](const std::vector<rclcpp::Parameter>& p){rcl_interfaces::msg::SetParametersResult r;r.successful=p[0].as_double()!=36000.;r.reason="test rejection";return r;});
 auto ui=std::make_shared<rclcpp::Node>("test_exposure_ui");a1::inspection::ExposureControl control(*ui);
 rclcpp::executors::SingleThreadedExecutor exec;exec.add_node(camera);exec.add_node(ui);
 auto spin=[&](int ms){auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(ms);while(std::chrono::steady_clock::now()<until){exec.spin_some();control.tick();std::this_thread::sleep_for(std::chrono::milliseconds(5));}};
 spin(2200);CHECK(control.actual()==35000.);
 auto enter=[&](std::string s){control.mouse(cv::EVENT_LBUTTONDOWN,200,110);for(char c:s)control.key(c);control.key(13);spin(1300);};
 enter("40000");CHECK(camera->get_parameter("ExposureTime").as_double()==40000.);CHECK(control.actual()==40000.);
 enter("36000");CHECK(control.actual()==40000.);
 enter("0");CHECK(control.actual()==40000.);
 control.mouse(cv::EVENT_LBUTTONDOWN,520,110);spin(1300);CHECK(control.actual()==41000.);
 double v;CHECK(!a1::inspection::ExposureControl::parse("nan",v));CHECK(!a1::inspection::ExposureControl::parse("35000junk",v));CHECK(!a1::inspection::ExposureControl::parse("1000001",v));
 rclcpp::shutdown();
}
