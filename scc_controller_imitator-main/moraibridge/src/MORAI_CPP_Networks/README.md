# MORAI C++ Network Examples

이 프로젝트는 MORAI 시뮬레이터와의 통신을 위한 C++ 네트워크 ROS 패키지입니다.

## 빌드 요구사항

### 공통 요구사항:
- ROS
- CMake (>= 3.10)
- OpenCV (>= 4.0)
- Eigen3
- C++17 호환 컴파일러

## ROS 패키지로 사용하기

1. ROS 워크스페이스에 패키지 추가
   ~~~~ bash
   # ROS 워크스페이스로 이동
   cd ~/catkin_ws/src
   
   # 프로젝트 클론 또는 복사
   git clone <repository_url>
   # 또는
   cp -r /path/to/MORAI_CPP_Networks .
   ~~~~

2. 의존성 설치
   ~~~~ bash
   # OpenCV 설치
   sudo apt-get install libopencv-dev
   
   # Eigen3 설치
   sudo apt-get install libeigen3-dev
   ~~~~

3. 빌드
   ~~~~ bash
   # 워크스페이스 루트로 이동
   cd ~/catkin_ws
   
   # 빌드 수행
   catkin_make
   
   # 환경 설정 소싱
   source devel/setup.bash
   ~~~~

## 다른 ROS 패키지에서 사용하기

1. package.xml에 의존성 추가:
~~~~ xml
<depend>morai_cpp_networks</depend>
~~~~

2. CMakeLists.txt에 의존성 추가:
~~~~ cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  morai_cpp_networks
  # 기타 필요한 패키지들...
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

target_link_libraries(your_node
  ${catkin_LIBRARIES}
)
~~~~

### 사용 예시

~~~~ cpp
#include <morai_cpp_networks/network/udp_receiver.hpp>
#include <morai_cpp_networks/sensors/camera.hpp>

// 카메라 센서 콜백 함수 정의
void CameraCallback(const CameraData& data) {
   cv::Mat image(data.height, data.width, CV_8UC3, data.image);
   cv::imshow("Camera", image);
   cv::waitKey(1);
}

int main(int argc, char** argv) {
   ros::init(argc, argv, "camera_example");
   ros::NodeHandle nh;

   // UDP 수신기 생성
   std::function<void(const CameraData&)> camera_callback = CameraCallback;
   Camera receiver("127.0.0.1", 7777, camera_callback);

   ros::spin();
   return 0;
}
~~~~

## 제공되는 라이브러리

1. Network Module (`morai_cpp_networks_network`):
   - UDP 통신 기본 기능
   - 데이터 송수신 클래스

2. Sensor Module (`morai_cpp_networks_sensor`):
   - Camera
   - IMU
   - GPS
   - Vehicle State
   - Traffic Light
   - Object Info

3. Actuator Module (`morai_cpp_networks_actuator`):
   - Vehicle Control
   - Turn Signal Control

4. Simulator Module (`morai_cpp_networks_simulator`):
   - Multi-Ego Setting

## 주의사항

1. ROS 환경:
   - ROS Noetic 기반으로 개발됨
   - catkin 빌드 시스템 사용

2. 의존성:
   - OpenCV (>= 4.0)
   - Eigen3
   - ROS 기본 패키지들

3. 네트워크 설정:
   - MORAI 시뮬레이터가 실행 중이어야 함
   - 포트와 IP 설정이 올바르게 되어 있어야 함
