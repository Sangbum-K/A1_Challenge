int64 id
bool confirmed
uint32 points
# Filtered position and dimensions; raw observed bounds retained separately.
float64[3] raw_center_m
float64[3] raw_size_m
float64[3] center_m
float64[3] size_m
# Apparent motion in the sensor-aligned frame; no ego-motion compensation.
float64[2] apparent_velocity_mps
bool projectable
float64[4] image_box_xyxy
float64 horizontal_iou
bool lidar_candidate
bool camera_matched
int32 matched_bin_left
int32 matched_bin_right
float64 range_m
# Current state timestamp is Association.header; raw bounds are last measured bounds.
bool measured
bool prediction_valid
float64 observation_age_s
float64[4] position_covariance_xy
