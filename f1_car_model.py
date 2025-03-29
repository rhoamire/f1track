import numpy as np

class F1CarModel:
    def __init__(self):
        """Initialize F1 car physical model."""
        # Car physical parameters
        self.mass = 798  # kg (F1 car without driver/fuel)
        self.wheelbase = 3.6  # meters
        self.cg_height = 0.28  # meters (center of gravity height)
        self.max_power = 735  # kW (~1000 HP)
        self.drag_coefficient = 0.7
        self.downforce_coefficient = 3.0

        # Tire parameters (simplified)
        self.max_grip_coefficient = 1.5
        self.optimal_slip_angle = 7.0  # degrees

        # Current state
        self.reset()

    def reset(self):
        """Reset car state."""
        self.position_s = 0.0  # Distance along track
        self.position_d = 0.0  # Lateral distance from centerline
        self.speed = 10.0      # m/s
        self.heading = 0.0     # radians, relative to track
        self.slip_angle = 0.0  # radians
        self.acceleration = 0.0  # m/s^2
        self.tire_wear = 0.0     # 0-1 (percentage)

    def step(self, action, track, dt=0.1):
        """Update car state based on controls and track."""
        # Unpack actions
        steering, throttle, brake = action

        # Get current track info
        curvature = track.get_curvature_at_s(self.position_s)
        track_width = track.get_width_at_s(self.position_s)

        # Calculate aerodynamic effects
        drag_force = 0.5 * self.drag_coefficient * (self.speed ** 2)
        downforce = 0.5 * self.downforce_coefficient * (self.speed ** 2)

        # Longitudinal dynamics (acceleration)
        # Simplified power model with drag
        if self.speed < 1.0:  # Prevent division by zero at very low speeds
            power_limit = self.max_power
        else:
            power_limit = self.max_power / self.speed  # N

        # Apply throttle and brake
        throttle_force = throttle * power_limit
        brake_force = brake * 30000  # N, maximum brake force

        # Net longitudinal force (positive = acceleration, negative = deceleration)
        long_force = throttle_force - brake_force - drag_force

        # Convert to acceleration (F = ma)
        self.acceleration = long_force / self.mass

        # Update speed
        self.speed += self.acceleration * dt
        self.speed = max(0.0, self.speed)  # Ensure speed is non-negative

        # Lateral dynamics (cornering)
        # Simplified lateral acceleration model based on grip
        effective_grip = self.max_grip_coefficient + (downforce / (self.mass * 9.81)) * 0.5

        # Calculate max lateral acceleration
        max_lat_accel = effective_grip * 9.81  # m/s^2

        # Calculate required lateral force for current curvature
        desired_lat_accel = self.speed**2 * (curvature - (steering / self.wheelbase))
        actual_lat_accel = np.clip(desired_lat_accel, -max_lat_accel, max_lat_accel)

        # Calculate slip angle (simplified)
        self.slip_angle = np.arctan2(actual_lat_accel, self.speed**2 / self.wheelbase)

        # Update lateral position (d) based on slip angle and heading
        self.position_d += self.speed * np.sin(self.slip_angle) * dt

        # Ensure we stay within track boundaries (optional, can be handled by rewards instead)
        self.position_d = np.clip(self.position_d, -track_width/2, track_width/2)

        # Update longitudinal position (s) based on speed
        # (accounting for effective distance along racing line)
        effective_speed = self.speed * np.cos(self.slip_angle)
        self.position_s += effective_speed * dt

        # Tire wear increases with high slip angles and high speeds
        slip_angle_deg = np.abs(np.degrees(self.slip_angle))
        self.tire_wear += (slip_angle_deg / 100) * (self.speed / 100) * dt
        self.tire_wear = min(1.0, self.tire_wear)

        # Return updated state
        return {
            "position_s": self.position_s,
            "position_d": self.position_d,
            "speed": self.speed,
            "slip_angle": self.slip_angle,
            "tire_wear": self.tire_wear
        }