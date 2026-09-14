int32 longlCmdType      # 1: Throttle control, 2: Velocity control, 3: Acceleration control

float64 accel           # [0~1], Accel pedal position
float64 brake           # [0~1], Brake pedal position
float64 steering        # [rad], front tire steering angle

float64 velocity        # [km/h], only active if longlCmdType == 2
float64 acceleration    # [m/s^2], only active if longlCmdType == 3