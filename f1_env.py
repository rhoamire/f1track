import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting
from gymnasium import spaces
import json

class F1RacingEnv(gym.Env):
    def __init__(self, track_name: str = "Singapore", year: int = 2023, tire_compound: str = "soft"):
        super().__init__()

        self.track_name = track_name  # Store track name for later use
        fastf1.Cache.enable_cache('cache')
        session = fastf1.get_session(year, track_name, 'R')
        session.load()
        lap = session.laps.pick_fastest()
        self.telemetry = lap.get_telemetry()

        self.X = self.telemetry['X'].values
        self.Y = self.telemetry['Y'].values
        self.Speed = self.telemetry['Speed'].values
        
        # Store previous values for acceleration calculation
        self.prev_speed = 0
        self.prev_direction = np.array([0, 1])

        self.current_step = 0
        self.max_steps = len(self.X)
        self.current_lap = 1

        # Load track-specific parameters
        with open('track_params.json', 'r') as f:
            track_params = json.load(f)
            if track_name not in track_params:
                raise ValueError(f"No parameters found for track: {track_name}")
            params = track_params[track_name]

        self.mu_base = params['mu_base']
        self.mu_max_values = params['mu_max_values']
        self.D_t_values = params['D_t_values']
        self.k_evolution = params['k_evolution']

        self.tire_compound = tire_compound
        self.mu_max = self.mu_max_values[self.tire_compound]
        self.D_t = self.D_t_values[self.tire_compound]

        self.mu = self.mu_max
        self.g = 9.81

        # Time and distance tracking
        self.time_step = 0.1
        self.total_time = 0.0
        self.distance_traveled = 0.0
        self.distance_scale = 1.0
        self.last_lap_time = 0.0
        self.best_lap_time = float('inf')

        self.track_length = self._calculate_track_length()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

    def _calculate_track_length(self):
        """Calculate the total track length in meters using the X,Y coordinates"""
        length = 0
        for i in range(1, len(self.X)):
            dx = self.X[i] - self.X[i-1]
            dy = self.Y[i] - self.Y[i-1]
            length += np.sqrt(dx**2 + dy**2) * self.distance_scale
        return length

    def step(self, action):
        steer, throttle, brake = action
        x, y = self.X[self.current_step], self.Y[self.current_step]

        # Modified grip calculation
        track_grip = self.mu_base + (self.mu_max - self.mu_base) * (1 - np.exp(-self.k_evolution * self.current_lap))
        tire_wear_effect = self.D_t * (self.current_lap ** 0.3)  # Reduced wear impact
        self.mu = max(track_grip - tire_wear_effect, 1.3)

        velocity = self.Speed[self.current_step]
        curvature_radius = max(np.linalg.norm([self.X[min(self.current_step+2, self.max_steps-1)] - x,
                                               self.Y[min(self.current_step+2, self.max_steps-1)] - y]), 1)
        
        # Adjusted cornering speed calculation
        max_cornering_speed = np.sqrt(self.mu * self.g * curvature_radius)
        
        # Load track-specific tire bonuses
        with open('track_params.json', 'r') as f:
            track_params = json.load(f)
            params = track_params[self.track_name]
            tire_speed_bonus = params['tire_speed_bonus']
            acceleration_bonus = params['acceleration_bonus']
        
        # Apply compound-specific effects
        base_speed = velocity + (throttle * 10 - brake * self.g) * 0.1 * acceleration_bonus[self.tire_compound]
        speed = base_speed * tire_speed_bonus[self.tire_compound]
        speed = min(speed, max_cornering_speed * tire_speed_bonus[self.tire_compound])

        if self.current_step > 0:
            dx = self.X[self.current_step] - self.X[self.current_step - 1]
            dy = self.Y[self.current_step] - self.Y[self.current_step - 1]
            step_distance = np.sqrt(dx**2 + dy**2)
            
            # Adjusted speed conversion and time calculation
            speed_ms = max(speed * (1000/3600), 5.0)
            actual_time_step = (step_distance / speed_ms) * 0.02
            
            self.total_time += actual_time_step
            self.distance_traveled += step_distance
        else:
            self.total_time += 0.01

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Modified reward function to account for tire performance
        reward = speed * tire_speed_bonus[self.tire_compound] - abs(steer) * 5 - brake * 10
        if done:
            self.current_lap += 1
            reward += 100

        info = {
            "distance": self.distance_traveled,
            "time": self.total_time,
            "track_length": self.track_length,
            "current_speed": speed,
            "grip_level": self.mu
        }

        return np.array([x, y, speed, steer, throttle, self.mu, tire_wear_effect], dtype=np.float32), reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_time = 0.0
        self.distance_traveled = 0.0
        return np.array([self.X[0], self.Y[0], self.Speed[0], 0, 0, self.mu_max, 0], dtype=np.float32), {}

    def render(self, mode="human"):
        fastf1.plotting.setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.X, self.Y, linestyle="dotted", color="black", label="Track Layout")

        max_index = min(self.current_step, self.max_steps - 1)

        scatter = ax.scatter(self.X[:max_index], self.Y[:max_index],
                            c=self.Speed[:max_index], cmap="coolwarm", s=5, label="Racing Line")

        ax.scatter(self.X[max_index], self.Y[max_index], color="yellow", marker="o", label="Current Position")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"AI Racing Line (Lap {self.current_lap}, Tire: {self.tire_compound}, Grip: {self.mu:.2f})")
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Speed (km/h)")

        # Save the plot instead of showing it
        plt.savefig(f"racing_line_{self.tire_compound}_lap_{self.current_lap}.png")
        plt.close(fig)

    def get_speed(self):
        return self.Speed[min(self.current_step, len(self.Speed) - 1)]  # Ensure valid index

    def get_distance_traveled(self):
        """Return the total distance traveled in meters"""
        return self.distance_traveled 