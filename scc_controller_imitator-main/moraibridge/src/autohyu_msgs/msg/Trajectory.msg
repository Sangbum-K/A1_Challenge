# std_msgs/Header
std_msgs/Header header

uint16 id
uint8 behavior_zone
uint8 next_behavior_zone
uint8 decision_behavior
float64 behavior_distance
uint8 is_target

autohyu_msgs/TrajectoryBoundary left_boundary
autohyu_msgs/TrajectoryBoundary right_boundary
autohyu_msgs/TrajectoryBoundary center_boundary
autohyu_msgs/TrajectoryPoint[] point
