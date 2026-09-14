# std_msgs/Header
std_msgs/Header header

# gnss
autohyu_msgs/WGS84 gnss

# position and motion
float64 x
float64 y
float64 z

float64 vx
float64 vy
float64 vz

float64 ax
float64 ay
float64 az

float64 roll
float64 pitch
float64 yaw

float64 roll_vel
float64 pitch_vel
float64 yaw_vel

string link_id

# vehicle can
autohyu_msgs/VehicleCAN vehicle_can