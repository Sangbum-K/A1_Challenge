int64 id
bool confirmed
uint32 points
# Filtered position and dimensions; raw observed bounds retained separately.
float64[3] raw_center_m
float64[3] raw_size_m
float64[3] center_m
float64[3] size_m
# Body axes, lead minus ego. Apparent sensor-frame motion when Association.ego_status is unavailable.
float64[2] relative_velocity_mps
# Nearest rear corner, body X from the LiDAR origin at the state timestamp.
float64 near_surface_x_m
bool projectable
float64[4] image_box_xyxy
bool lidar_candidate
bool camera_matched
int32 matched_bin_left
int32 matched_bin_right
float64 range_m
# Current state timestamp is Association.header; raw bounds are last measured bounds.
bool measured
bool prediction_valid
# Large visible-shape change: identity can survive while motion settles.
bool motion_reacquiring
float64 observation_age_s
float64[4] position_covariance_xy

bool weak_measurement
# Accumulated camera evidence and how the camera last saw this track.
float64 camera_evidence
string camera_state
# EMA of points on mapped static structure. NaN when the map cannot judge.
float64 map_static_ratio
bool map_static
# Far-vehicle cue sum. NaN below 35 m or when its validity conditions fail.
float64 far_score
bool far_vehicle
# Racing line coordinates: distance ahead of the ego, signed lateral offset. NaN when unavailable.
float64 path_s_m
float64 path_n_m

# Auditable geometry and selection outcome, including rejected tracks.
string geometry_status
string selection_reason
