#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from morai_msgs.msg import CtrlCmd


class AccelToPedal:
    def __init__(self):
        # ROS Node 초기화
        rospy.init_node('acceleration_control_node', anonymous=True)
        self.rate = rospy.Rate(100) 

        # Publisher 
        self.p_command_pedal_accel = rospy.Publisher('/app/con/command_pedal_accel', Float32, queue_size=10)
        self.p_command_pedal_brake = rospy.Publisher('/app/con/command_pedal_brake', Float32, queue_size=10)
            
        self.o_command_pedal_accel = Float32()
        self.o_command_pedal_brake = Float32()

        # Subscriber
        rospy.Subscriber('/app/con/command_accel', Float32, self.target_accel_callback)
        rospy.Subscriber('/app/con/command_torque', Float32, self.target_torque_callback)

        # Vehicle Parameters
        self.params = {
            'mass': 2300,                # [kg]
            'wheel_radius': 0.36,        # [m]
            'neutral_torque': -360,      # [Nm]
            'accel_to_torque_ratio': 0.0002655337,   # torque*accel_to_torque_ratio = accel_pedal (0~1)   
            'brake_to_torque_ratio': 0.0000769231    # torque*brake_to_torque_ratio = brake_pedal (0~1)
        }

        # variables
        self.last_log_time = rospy.Time.now()

        rospy.loginfo("AccelToPedal Node Initialized")

        self.i_command_acceleration = None
        self.i_command_torque = None
        


    def target_accel_callback(self, msg):
        self.i_command_acceleration = msg.data

    def target_torque_callback(self, msg):
        self.i_command_torque = msg.data

    def run(self):
        while not rospy.is_shutdown():
            # Calculate Target Torque
            if self.i_command_acceleration is None or self.i_command_torque is None:
                rospy.loginfo("Waiting for Command ...")
                continue
            elif self.i_command_acceleration is not None:
                target_torque = self.i_command_acceleration * self.params['mass'] * self.params['wheel_radius']
            else:   # getting target torque
                target_torque = self.i_command_torque
                
            # Convert Torque to Pedal Input
            if target_torque > self.params['neutral_torque']:
                accel_pedal = (target_torque - self.params['neutral_torque']) * self.params['accel_to_torque_ratio']
                brake_pedal = 0.0
            elif target_torque < self.params['neutral_torque']:
                accel_pedal = 0.0
                brake_pedal = (self.params['neutral_torque'] - target_torque) * self.params['brake_to_torque_ratio']
            else:
                accel_pedal = 0.0
                brake_pedal = 0.0

            # Publish 
            self.o_command_pedal_accel.data = accel_pedal
            self.o_command_pedal_brake.data = brake_pedal
            self.p_command_pedal_accel.publish(self.o_command_pedal_accel)
            self.p_command_pedal_brake.publish(self.o_command_pedal_brake)

            # Log
            current_time = rospy.Time.now()
            if (current_time - self.last_log_time).to_sec() >= 1.0:
                # rospy.loginfo(f"Target Acceleration: {self.i_command_acceleration:.2f}, Accel Pedal: {accel_pedal:.2f}, Brake Pedal: {brake_pedal:.2f}")
                self.last_log_time = current_time

            self.rate.sleep()
        

if __name__ == '__main__':
    try:
        node = AccelToPedal()
        node.run()
    except rospy.ROSInterruptException:
        pass
