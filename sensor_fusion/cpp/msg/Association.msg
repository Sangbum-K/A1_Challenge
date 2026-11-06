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
# Current camera matches among the selected primary/standby (at most max_targets).
int64[] candidate_track_ids
# Selected camera-supported tracks, including a bounded held/predicted primary.
int64[] lidar_candidate_track_ids
bool association_ambiguous
# Camera-supported primary. A short held/predicted target is display-only.
int64 primary_track_id
int64 standby_track_id
string primary_status

# Time since last unambiguous camera support; NaN without a primary.
float64 primary_camera_age_s

# compensated | unavailable
string ego_status
# ok, or why the map cannot judge: no_structure, localization_untrusted, no_map, ...
string map_status
# active | no_localization | unverified | misaligned | disabled
string raceline_status
uint32 dropped_scans
