/**
 * @file        function_task_manager.hpp
 * @brief       function for task managing
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2024-04-01 created by Yuseung Na: Refactoring
 *              2024-04-12 created by Junhee Lee: add ShowAILab()
 */

#ifndef __TASK_MANAGER_HPP__
#define __TASK_MANAGER_HPP__
#pragma once

// STD Header
#include <mutex>

// Boost Header
#include <boost/thread/thread.hpp>

// ROS Header
#include <ros/ros.h>

// ROS Message Header
#include <std_msgs/Float64.h>

class TaskManager {
    protected:
        TaskManager(std::string task_name, double period);
        ~TaskManager();

    public:
        void Exec(int num_thread=1);
        void MainLoop();

    protected:
        virtual void Run()=0;
        virtual void Publish()=0;

    private:
        ros::Publisher pub_life_;
        void ShowAILab();
        void PublishLifeSignal();

    protected:
        std::string	  task_name_;
        double	      task_period_;   
        double	      task_rate_;
        ros::Time	  update_time_;
        ros::Duration duration_time_;
        ros::Duration execution_time_;
};

TaskManager::TaskManager(std::string task_name, double period)
    :task_name_(""), task_period_(0.0), task_rate_(0.0) {
    ShowAILab();
    task_name_ = task_name;
    task_period_ = period;
    task_rate_ = 1.0/period;

    update_time_ = ros::Time::now();
    execution_time_ = ros::Duration(0.0);

    ros::NodeHandle nh;
    pub_life_ = nh.advertise<std_msgs::Float64>("life_sig_" + task_name_, 100);
}

TaskManager::~TaskManager(){
}

void TaskManager::Exec(int num_thread){
    boost::thread main_thread( boost::bind(  &TaskManager::MainLoop, this ) );

    ros::AsyncSpinner spinner(num_thread);
    spinner.start();
    ros::waitForShutdown();

    main_thread.join();
}

void TaskManager::MainLoop(){
    ros::Rate loop_rate(task_rate_);
    ros::Time last_log_time = ros::Time::now();
    while( ros::ok() ){
        update_time_ = ros::Time::now();
        
        // Run algorithm
        Run();

        // Calculate execution time
        execution_time_ = ros::Time::now() - update_time_;

        if ((ros::Time::now() - last_log_time).toSec() >= 1.0) {
            if (execution_time_.toSec() > task_period_) {
                ROS_ERROR_STREAM("[" << task_name_ << "] Rate: " << task_period_*1000.0 << 
                                 "ms, Exec Time:" << (execution_time_).toSec()*1000.0 << "ms");
            }
            else {
                ROS_INFO_STREAM("[" << task_name_ << "] Rate: " << task_period_*1000.0 << 
                                "ms, Exec Time:" << (execution_time_).toSec()*1000.0 << "ms");
            }
            last_log_time = ros::Time::now();
        }

        // Publish topics
        Publish();
        PublishLifeSignal();

        loop_rate.sleep();
    }
}
void TaskManager::ShowAILab(){
    // std::cout << "                                                   \n";
    // std::cout << "                 █████████████████                 \n";
    // std::cout << "             █████████████████████████             \n";
    // std::cout << "           █████████████████████████████           \n";
    // std::cout << "         █████████████████████████████████         \n";
    // std::cout << "        ████████████           ████████████        \n";
    // std::cout << "       ██████████                 ██████████       \n";
    // std::cout << "      █████████   ██           ██   █████████      \n";
    // std::cout << "      █████████                     █████████      \n";
    // std::cout << "      ████████                       ████████      \n";
    // std::cout << "      ████████                       ████████      \n";
    // std::cout << "      ████████                       ████████      \n";
    // std::cout << "      ████████        ███████        ████████      \n";
    // std::cout << "      ████████    ███████████████    ████████      \n";
    // std::cout << "      ████████    ███████████████    ████████      \n";
    // std::cout << "      █████        █████████████        █████      \n";
    // std::cout << "      █████                             █████      \n";
    // std::cout << "      ████████                       ████████      \n";
    // std::cout << "      ████████                       ████████      \n";
    // std::cout << "      ████████                       ████████      \n";
    std::cout << "                                                   \n";
    std::cout << "     ___      __     __          ___      ______   \n";
    std::cout << "    /   \\    |  | | |  |        /   \\    |   _  \\  \n";
    std::cout << "   /  ^  \\   |  |   |  |       /  ^  \\   |  |_)  |   \n";
    std::cout << "  /  /_\\  \\  |  | | |  |      /  / \\  \\  |   _  <  \n";
    std::cout << " /  _____  \\ |  |   |  `----./  /   \\  \\ |  |_)  |  \n";
    std::cout << "/__/     \\__\\|__| | |_______/__/     \\__\\|______/  \n";
    std::cout << "                                                   \n";

}

void TaskManager::PublishLifeSignal(){
    std_msgs::Float64 time;
    time.data = update_time_.toSec();

    pub_life_.publish(time);
}

#endif // __TASK_MANAGER_HPP__