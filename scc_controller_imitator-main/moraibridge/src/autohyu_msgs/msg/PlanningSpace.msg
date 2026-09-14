std_msgs/Header header

# Ego
autohyu_msgs/PlanningSpaceEgo ego

# Objects
autohyu_msgs/PlanningSpaceObject[] objects

# Vector Map
autohyu_msgs/PlanningSpaceVectorMap[] lanes
autohyu_msgs/PlanningSpaceVectorMap[] routes
autohyu_msgs/PlanningSpaceVectorMap[] crosswalks
autohyu_msgs/PlanningSpaceVectorMap[] goal_routes

# Drivable Map
autohyu_msgs/Trajectory drivable_area

# Vocab
autohyu_msgs/PlanningSpaceTrafficLight traffic_light
autohyu_msgs/PredictObjects predict_objects
autohyu_msgs/Trajectory imitation_trajectory