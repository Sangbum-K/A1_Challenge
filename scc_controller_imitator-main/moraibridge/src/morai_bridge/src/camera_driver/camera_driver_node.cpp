/**
 * @file        camera_driver_node.cpp
 * @brief       morai camera driver node cpp file
 * 
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2025-01-17 created by Jiwon Seok
 * 
 */

#include "camera_driver/camera_driver_node.hpp"

CameraDriver::CameraDriver(std::string node_name, double period)
    : TaskManager(node_name, period) {
    // Initialize
    ROS_WARN_STREAM("[" << node_name << "] Initialize node (Period: " << 1/period << " Hz)");

    // Ini init
    std::string dir(getenv("PWD"));
    std::string ini_dir("/config/simulation.ini");
    util_ini_parser_.Init((dir + ini_dir).c_str());

    // Parameter init
    ProcessINI();
    ProcessRosparam();
    cfg_.task_period = period;

    // Publisher init
    for(int i = 0; i < cfg_.num_of_camera; i++) {
        // if(cfg_.publish_raw_image) {
            p_cameras_.push_back(nh_.advertise<sensor_msgs::Image>(
                "raw_img_" + cfg_.camera_topic_name_list[i], 10));
        // }
        // if(cfg_.publish_compressed_image) {
            p_compressed_cameras_.push_back(nh_.advertise<sensor_msgs::CompressedImage>(
                cfg_.camera_topic_name_list[i], 10));
        // }
    }
    b_is_publisher_initialized_ = true;

    // Initialize the camera driver
    for(int i = 0; i < cfg_.num_of_camera; i++) {
        if (cfg_.morai_simulator_version == 1) {
            morai_udp_camera_receivers_.push_back(std::make_unique<MoraiCppUdp::Camera>(
                cfg_.udp_destination_ip,
                cfg_.udp_camera_driver_destination_ports[i],
                std::bind(&CameraDriver::CallbackMoraiUdpCamera, this, std::placeholders::_1, i)));
        }
        else if (cfg_.morai_simulator_version == 2) {
            morai_udp_camera2_receivers_.push_back(std::make_unique<MoraiCppUdp::Camera2>(
                cfg_.udp_destination_ip,
                cfg_.udp_camera_driver_destination_ports[i],
                std::bind(&CameraDriver::CallbackMoraiUdpCamera2, this, std::placeholders::_1, i)));
        }
    }
}

CameraDriver::~CameraDriver() {
    // Terminate
}

void CameraDriver::Run() {
    ProcessINI();
}

void CameraDriver::Publish() {
    // Non in publish.
    // Publish in callbackMoraiUdpCamera
}

void CameraDriver::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()) {
        ROS_WARN("[Camera Driver] Ini file is updated!");
        // Simulator
        if(util_ini_parser_.ParseConfig("MoraiBridge", "morai_simulator_version", cfg_.morai_simulator_version) == false) {
            std::cout << "morai_simulator_version is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiBridge", "udp_source_ip", cfg_.udp_source_ip) == false) {
            std::cout << "udp_source_ip is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiBridge", "udp_destination_ip", cfg_.udp_destination_ip) == false) {
            std::cout << "udp_destination_ip is not found in ini file" << std::endl;
        }

        // Morai Camera Driver
        std::vector<int> udp_camera_driver_source_ports;
        std::vector<int> udp_camera_driver_destination_ports;
        std::vector<std::string> camera_topic_name_list;
        std::vector<std::string> camera_topic_frame_id_list;
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "num_of_camera", cfg_.num_of_camera) == false) {
            std::cout << "num_of_camera is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "publish_raw_image", cfg_.publish_raw_image) == false) {
            std::cout << "publish_raw_image is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "publish_compressed_image", cfg_.publish_compressed_image) == false) {
            std::cout << "publish_compressed_image is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "udp_camera_driver_source_ports", udp_camera_driver_source_ports) == false) {
            std::cout << "udp_camera_driver_source_ports is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "udp_camera_driver_destination_ports", udp_camera_driver_destination_ports) == false) {
            std::cout << "udp_camera_driver_destination_ports is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "camera_topic_name_list", camera_topic_name_list) == false) {
            std::cout << "camera_topic_name_list is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiCameraDriver", "camera_topic_frame_id_list", camera_topic_frame_id_list) == false) {
            std::cout << "camera_topic_frame_id_list is not found in ini file" << std::endl;
        }

        for(int i = 0; i < cfg_.num_of_camera; i++) {
            cfg_.udp_camera_driver_source_ports[i] = udp_camera_driver_source_ports[i];
            cfg_.udp_camera_driver_destination_ports[i] = udp_camera_driver_destination_ports[i];
            cfg_.camera_topic_name_list[i] = camera_topic_name_list[i];
            cfg_.camera_topic_frame_id_list[i] = camera_topic_frame_id_list[i];
        }
    }
}

void CameraDriver::ProcessRosparam() {}

void CameraDriver::CallbackMoraiUdpCamera(const MoraiCppUdp::Camera::CameraData& camera_data, int camera_index) {
    if(cfg_.publish_raw_image) {
        sensor_msgs::Image ros_msg;
        ros_msg.header.stamp = ros::Time(camera_data.timestamp);
        ros_msg.header.frame_id = cfg_.camera_topic_frame_id_list[camera_index];
        ros_msg.encoding = "bgr8";
        ros_msg.height = camera_data.height;
        ros_msg.width = camera_data.width;
        ros_msg.is_bigendian = false;
        ros_msg.step = camera_data.width * 3;  // 3 channels (BGR)
        
        // 이미지 데이터 복사
        ros_msg.data.resize(camera_data.width * camera_data.height * 3);
        std::memcpy(ros_msg.data.data(), camera_data.image_data.data, camera_data.width * camera_data.height * 3);

        // 토픽 발행
        if (b_is_publisher_initialized_) {
            if(!p_cameras_[camera_index]) {
                p_cameras_[camera_index] = nh_.advertise<sensor_msgs::Image>(
                    cfg_.camera_topic_name_list[camera_index], 10);
            }
            p_cameras_[camera_index].publish(ros_msg);
        }
    }
    if (cfg_.publish_compressed_image) {
        sensor_msgs::CompressedImage ros_msg;
        ros_msg.header.stamp = ros::Time(camera_data.timestamp);
        ros_msg.header.frame_id = cfg_.camera_topic_frame_id_list[camera_index];
        ros_msg.format = "jpeg";

        // Convert BGR image to JPEG using OpenCV
        std::vector<uchar> jpeg_buffer;
        cv::Mat bgr_image(camera_data.height, camera_data.width, CV_8UC3, (void*)camera_data.image_data.data);
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(100); // JPEG quality (0-100)
        cv::imencode(".jpg", bgr_image, jpeg_buffer, compression_params);
        
        // Copy compressed JPEG data to ROS message
        ros_msg.data.assign(jpeg_buffer.begin(), jpeg_buffer.end());

        // Publish topic
        if (b_is_publisher_initialized_) {
            if(!p_compressed_cameras_[camera_index]) {
                p_compressed_cameras_[camera_index] = nh_.advertise<sensor_msgs::CompressedImage>(
                    cfg_.camera_topic_name_list[camera_index] + "_compressed", 10);
            }
            p_compressed_cameras_[camera_index].publish(ros_msg);
        }
    }
}

void CameraDriver::CallbackMoraiUdpCamera2(const MoraiCppUdp::Camera2::CameraData& camera_data, int camera_index) {
    if(cfg_.publish_raw_image) {
        sensor_msgs::Image ros_msg;
        ros_msg.header.stamp = ros::Time::now();
        ros_msg.header.frame_id = cfg_.camera_topic_frame_id_list[camera_index];
        ros_msg.encoding = "bgr8";
        ros_msg.height = camera_data.height;
        ros_msg.width = camera_data.width;
        ros_msg.is_bigendian = false;
        ros_msg.step = camera_data.width * 3;  // 3 channels (BGR)
        
        // 이미지 데이터 복사
        ros_msg.data.resize(camera_data.width * camera_data.height * 3);
        std::memcpy(ros_msg.data.data(), camera_data.image_data.data, camera_data.width * camera_data.height * 3);

        // 토픽 발행
        if (b_is_publisher_initialized_) {
            if(!p_cameras_[camera_index]) {
                p_cameras_[camera_index] = nh_.advertise<sensor_msgs::Image>(
                    cfg_.camera_topic_name_list[camera_index], 10);
            }
            p_cameras_[camera_index].publish(ros_msg);
        }
    }
    if (cfg_.publish_compressed_image) {
        sensor_msgs::CompressedImage ros_msg;
        ros_msg.header.stamp = ros::Time::now();
        ros_msg.header.frame_id = cfg_.camera_topic_frame_id_list[camera_index];
        ros_msg.format = "jpeg";

        // Convert BGR image to JPEG using OpenCV
        std::vector<uchar> jpeg_buffer;
        cv::Mat bgr_image(camera_data.height, camera_data.width, CV_8UC3, (void*)camera_data.image_data.data);
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(100); // JPEG quality (0-100)
        cv::imencode(".jpg", bgr_image, jpeg_buffer, compression_params);
        
        // Copy compressed JPEG data to ROS message
        ros_msg.data.assign(jpeg_buffer.begin(), jpeg_buffer.end());

        // Publish topic
        if (b_is_publisher_initialized_) {
            if(!p_compressed_cameras_[camera_index]) {
                p_compressed_cameras_[camera_index] = nh_.advertise<sensor_msgs::CompressedImage>(
                    cfg_.camera_topic_name_list[camera_index] + "_compressed", 10);
            }
            p_compressed_cameras_[camera_index].publish(ros_msg);
        }
    }
}

int main(int argc, char** argv) {
    std::string node_name = "camera_driver";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;

    double period;
    if (!nh.getParam("task_period/period_camera_driver", period))
        period = 1.0;

    CameraDriver main_task(node_name, period);
    main_task.Exec(2);

    return 0;
}