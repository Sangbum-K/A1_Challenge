import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PedalValues:
    """Pedal control result structure"""
    accel_pedal: float = 0.0  # Accelerator pedal value (0.0 ~ 1.0)
    brake_pedal: float = 0.0  # Brake pedal value (0.0 ~ 1.0)

class PedalMap:
    """
    2D map system for mapping pedal values according to vehicle velocity and acceleration
    """
    def __init__(self):
        self.velocity_breakpoints: List[float] = []
        self.acc_pedal_acceleration_breakpoints: List[float] = []
        self.brk_pedal_acceleration_breakpoints: List[float] = []
        self.accel_values: np.ndarray = np.array([])
        self.brake_values: np.ndarray = np.array([])
        self.is_initialized: bool = False
        self.prev_brake_pedal: float = -1.0

    def load_from_file(self, file_path: str) -> bool:
        """
        Load pedal map data from file
        
        Args:
            file_path: Path to pedal map data file
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            velocity_points = []
            acc_pedal_acceleration_points = []
            brk_pedal_acceleration_points = []
            accel_values = []
            brake_values = []
            current_section = None

            # print(f"Opening file: {file_path}")
            with open(file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    # Print raw line for debugging
                    # print(f"Line {line_num}: '{line.strip()}'")
                    
                    # Remove comments and trim whitespace
                    line = line.split('#')[0].strip()
                    if not line:
                        continue

                    # Check section headers
                    if "Velocity Points" in line:
                        current_section = "VELOCITY"
                        # print(f"Found Velocity Points section at line {line_num}")
                        continue
                    elif "Acceleration Pedal Map" in line:
                        current_section = "ACCEL_MAP"
                        # print(f"Found Acceleration Pedal Map section at line {line_num}")
                        continue
                    elif "Brake Pedal Map" in line:
                        current_section = "BRAKE_MAP"
                        # print(f"Found Brake Pedal Map section at line {line_num}")
                        continue

                    # Parse data
                    values = []
                    for value in line.split(','):
                        value = value.strip()
                        if value:  # 빈 문자열이 아닌 경우만 처리
                            try:
                                values.append(float(value))
                            except ValueError:
                                print(f"Skipping non-numeric value: '{value}'")
                                continue

                    if not values:  # Skip empty lines
                        continue

                    # print(f"Current section: {current_section}, Values: {values}")

                    if current_section == "VELOCITY":
                        velocity_points = values
                        # print(f"Loaded velocity points: {velocity_points}")
                    elif current_section == "ACCEL_MAP":
                        if values[0] in acc_pedal_acceleration_points:
                            raise ValueError(f"Redundant acceleration value in accel map: {values[0]}")
                        acc_pedal_acceleration_points.append(values[0])
                        if len(values[1:]) != len(velocity_points):
                            raise ValueError(f"Invalid row size in accel map: {len(values[1:])}")
                        accel_values.append(values[1:])
                    elif current_section == "BRAKE_MAP":
                        if values[0] in brk_pedal_acceleration_points:
                            raise ValueError(f"Redundant acceleration value in brake map: {values[0]}")
                        brk_pedal_acceleration_points.append(values[0])
                        if len(values[1:]) != len(velocity_points):
                            raise ValueError(f"Invalid row size in brake map: {len(values[1:])}")
                        brake_values.append(values[1:])

            # Validate data
            if not velocity_points:
                raise ValueError("No velocity points loaded")
            if not acc_pedal_acceleration_points or not accel_values:
                raise ValueError("No acceleration pedal map data loaded")
            if not brk_pedal_acceleration_points or not brake_values:
                raise ValueError("No brake pedal map data loaded")

            # Initialize map with parsed data
            self._initialize(velocity_points, acc_pedal_acceleration_points, brk_pedal_acceleration_points)
            
            # Fill pedal values
            self.accel_values = np.array(accel_values).T
            self.brake_values = np.array(brake_values).T

            # print(f"Successfully loaded pedal map:")
            # print(f"Velocity points: {len(velocity_points)}")
            # print(f"Acceleration points: {len(acc_pedal_acceleration_points)}")
            # print(f"Brake points: {len(brk_pedal_acceleration_points)}")
            
            return True

        except Exception as e:
            print(f"Error loading pedal map: {str(e)}")
            return False

    def get_pedal_values(self, velocity: float, acceleration: float) -> PedalValues:
        """
        Get interpolated pedal values for given velocity and acceleration
        
        Args:
            velocity: Vehicle velocity (m/s)
            acceleration: Target acceleration (m/s^2)
            
        Returns:
            PedalValues: Interpolated accelerator and brake pedal values
        """
        if not self.is_initialized:
            print("PedalMap not initialized")
            return PedalValues()

        return PedalValues(
            accel_pedal=self._interpolate_2d(velocity, acceleration, is_accel=True),
            brake_pedal=self._interpolate_2d(velocity, acceleration, is_accel=False)
        )
    
    def get_pedal_values_progressive_brake(self, velocity: float, acceleration: float, prev_brake_pedal: float = -1.0) -> PedalValues:
        """
        Get interpolated pedal values with progressive brake logic
        
        Args:
            velocity: Vehicle velocity (m/s)
            acceleration: Target acceleration (m/s^2)
            prev_brake_pedal: Previous brake pedal value
            
        Returns:
            PedalValues: Interpolated accelerator and brake pedal values with progressive brake
        """
        if not self.is_initialized:
            print("PedalMap not initialized")
            return PedalValues()

        if prev_brake_pedal >= 0.0:
            self.prev_brake_pedal = prev_brake_pedal

        # Find the closest velocity and acceleration
        pedal_values = self.get_pedal_values(velocity, acceleration)

        # Progressive brake condition check
        # Velocity < 1.08 km/h, Acceleration < 0.1 m/s^2 is stop condition. Progressive brake is applied.
        if velocity < 0.3 and acceleration < 0.3:
            prev_brake_pedal = max(self.prev_brake_pedal, pedal_values.brake_pedal)
            current_brake_pedal = prev_brake_pedal * 1.02
            saturated_brake_pedal = min(current_brake_pedal, 1.0)
            pedal_values.brake_pedal = saturated_brake_pedal
        
        self.prev_brake_pedal = pedal_values.brake_pedal
        return pedal_values

    def get_acceleration(self, velocity: float, pedal_values: PedalValues) -> float:
        """
        Calculate acceleration for given velocity and pedal values
        
        Args:
            velocity: Vehicle velocity (m/s)
            pedal_values: Current pedal values
            
        Returns:
            float: Calculated acceleration (m/s^2)
        """
        if not self.is_initialized:
            print("PedalMap not initialized")
            return 0.0

        # Clamp velocity to valid range
        velocity = min(max(velocity, self.velocity_breakpoints[0]), self.velocity_breakpoints[-1])
        
        # Find velocity index for interpolation
        v_idx = np.searchsorted(self.velocity_breakpoints, velocity)
        v_idx = min(max(1, v_idx), len(self.velocity_breakpoints))
        
        # Get surrounding velocity points for interpolation
        v1 = self.velocity_breakpoints[v_idx-1]
        v2 = self.velocity_breakpoints[v_idx] if v_idx < len(self.velocity_breakpoints) else v1
        
        # Calculate interpolation coefficient for velocity
        t = (velocity - v1) / (v2 - v1) if abs(v2 - v1) > 1e-6 else 0.0
        
        # Get acceleration values at both velocity points
        accel_acc1 = self._inverse_interpolate_1d(
            pedal_values.accel_pedal,
            self.acc_pedal_acceleration_breakpoints,
            self.accel_values[v_idx-1]
        )
        
        brake_acc1 = self._inverse_interpolate_1d(
            pedal_values.brake_pedal,
            self.brk_pedal_acceleration_breakpoints,
            self.brake_values[v_idx-1]
        )
        
        if v_idx < len(self.velocity_breakpoints):
            accel_acc2 = self._inverse_interpolate_1d(
                pedal_values.accel_pedal,
                self.acc_pedal_acceleration_breakpoints,
                self.accel_values[v_idx]
            )
            
            brake_acc2 = self._inverse_interpolate_1d(
                pedal_values.brake_pedal,
                self.brk_pedal_acceleration_breakpoints,
                self.brake_values[v_idx]
            )
        else:
            accel_acc2 = accel_acc1
            brake_acc2 = brake_acc1
        
        # Interpolate accelerations between velocity points
        accel_acc = (1 - t) * accel_acc1 + t * accel_acc2
        brake_acc = (1 - t) * brake_acc1 + t * brake_acc2
        
        # Return the sum of accelerations
        # Since brake acceleration values are already negative in the map,
        # we just add them together
        if pedal_values.brake_pedal > 0.0:
            return brake_acc
        else:
            return accel_acc

    def _inverse_interpolate_1d(self, target_value: float, x_points: List[float], y_values: np.ndarray) -> float:
        """
        Perform 1D inverse interpolation to find x value for given y
        
        Args:
            target_value: Target y value to find x for
            x_points: List of x points
            y_values: Array of y values (can be ascending or descending)
            
        Returns:
            float: Interpolated x value
        """
        # Check if y_values is in ascending or descending order
        is_ascending = y_values[0] <= y_values[-1] if len(y_values) > 1 else True
        
        # Find the index where target_value would be inserted
        if is_ascending:
            idx = np.searchsorted(y_values, target_value)
        else:
            # For descending order, flip the comparison
            idx = np.searchsorted(-y_values, -target_value)
        
        idx = min(max(1, idx), len(y_values))
        
        # Get surrounding points
        y1 = y_values[idx-1]
        y2 = y_values[idx-1] if idx >= len(y_values) else y_values[idx]
        x1 = x_points[idx-1]
        x2 = x_points[idx-1] if idx >= len(x_points) else x_points[idx]
        
        # Handle edge cases
        if abs(y2 - y1) < 1e-6:
            return x1
        
        # Linear interpolation
        t = (target_value - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def _initialize(self, velocity_points: List[float], 
                   accel_points: List[float], 
                   brake_points: List[float]) -> None:
        """Initialize internal data structures of the pedal map"""
        self.velocity_breakpoints = velocity_points
        self.acc_pedal_acceleration_breakpoints = accel_points
        self.brk_pedal_acceleration_breakpoints = brake_points
        
        self.accel_values = np.zeros((len(velocity_points), len(accel_points)))
        self.brake_values = np.zeros((len(velocity_points), len(brake_points)))
        
        self.is_initialized = True

    def _interpolate_2d(self, velocity: float, acceleration: float, is_accel: bool) -> float:
        """
        Perform 2D linear interpolation
        
        Args:
            velocity: Vehicle velocity (m/s)
            acceleration: Target acceleration (m/s^2)
            is_accel: True for accelerator pedal, False for brake pedal
            
        Returns:
            float: Interpolated pedal value (0.0 ~ 1.0)
        """
        # Get breakpoints and values based on pedal type
        acc_points = self.acc_pedal_acceleration_breakpoints if is_accel else self.brk_pedal_acceleration_breakpoints
        values = self.accel_values if is_accel else self.brake_values

        # Clamp velocity and acceleration to valid ranges
        velocity = min(max(velocity, self.velocity_breakpoints[0]), self.velocity_breakpoints[-1])
        acceleration = min(max(acceleration, acc_points[0]), acc_points[-1])

        # Find indices for interpolation
        v_idx = np.searchsorted(self.velocity_breakpoints, velocity)
        a_idx = np.searchsorted(acc_points, acceleration)

        # Ensure indices are within valid range
        v_idx = min(max(1, v_idx), len(self.velocity_breakpoints))
        a_idx = min(max(1, a_idx), len(acc_points))

        # Get surrounding points
        v1 = self.velocity_breakpoints[v_idx-1]
        v2 = self.velocity_breakpoints[v_idx-1] if v_idx >= len(self.velocity_breakpoints) else self.velocity_breakpoints[v_idx]
        a1 = acc_points[a_idx-1]
        a2 = acc_points[a_idx-1] if a_idx >= len(acc_points) else acc_points[a_idx]

        # Get surrounding values
        f11 = values[v_idx-1, a_idx-1]
        f12 = values[v_idx-1, a_idx-1] if a_idx >= len(acc_points) else values[v_idx-1, a_idx]
        f21 = values[v_idx-1, a_idx-1] if v_idx >= len(self.velocity_breakpoints) else values[v_idx, a_idx-1]
        f22 = values[v_idx-1, a_idx-1] if (v_idx >= len(self.velocity_breakpoints) or a_idx >= len(acc_points)) else values[v_idx, a_idx]

        # Calculate interpolation coefficients
        t = 0.0 if abs(v2 - v1) < 1e-6 else (velocity - v1) / (v2 - v1)
        u = 0.0 if abs(a2 - a1) < 1e-6 else (acceleration - a1) / (a2 - a1)

        # Perform bilinear interpolation
        interpolated = (1 - t) * (1 - u) * f11 + t * (1 - u) * f21 + \
                      (1 - t) * u * f12 + t * u * f22

        # Saturate output
        return np.clip(interpolated, 0.0, 1.0)