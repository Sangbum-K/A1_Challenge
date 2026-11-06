std_msgs/Header header
builtin_interfaces/Time image_stamp
builtin_interfaces/Time a2_stamp
float64 image_delta_s
float64 a2_delta_s
bool calibration_verified
bool clock_verified
bool motion_compensated
bool candidate_unique
bool verified_match
int64 candidate_track_id
int32 bin_left
int32 bin_right
string status
uint32 ground_points
uint32 nonground_points
float64 processing_ms
ClusterTrack[] tracks
# At most max_targets; sorted by measured planar center range, NOT a brake priority.
int64[] candidate_track_ids
int64[] lidar_candidate_track_ids
bool association_ambiguous
# LiDAR-led primary only; far target remains standby and does not generate P5.
int64 primary_track_id
int64 standby_track_id
string primary_status
