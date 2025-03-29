import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from f1_car_model import F1CarModel
from track import Track
import pandas as pd

class F1RacingEnv(gym.Env):
    """F1 Racing Environment for Reinforcement Learning."""

    def __init__(self, track_name="monza", year=2023, max_steps=1000):
        super(F1RacingEnv, self).__init__()

        # Load track data
        self.track = Track(track_name, year)

        # Create car model
        self.car = F1CarModel()

        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.dt = 0.1  # seconds
        self.lap_completed = False
        self.crashed = False
        self.last_action = None
        self.lap_time = 0.0
        self.best_lap_time = float('inf')
        self.racing_line = []  # List to store (s, d) points of the racing line

        # Define action and observation space
        # Actions: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # State: [position_s (normalized), position_d (normalized),
        #         speed, slip_angle, track_curvature, track_width]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -np.pi/2, -0.1, 5.0]),
            high=np.array([1.0, 1.0, 400.0, np.pi/2, 0.1, 20.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.car.reset()
        self.current_step = 0
        self.lap_time = 0.0
        self.lap_completed = False
        self.crashed = False
        self.last_action = None
        self.racing_line = []

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """Step the environment by one timestep."""
        self.current_step += 1

        # Store action for use in reward function
        self.last_action = action

        # Update car state
        car_state = self.car.step(action, self.track, self.dt)

        # Update lap time
        self.lap_time += self.dt

        # Store current position for racing line visualization
        self.racing_line.append((car_state["position_s"], car_state["position_d"]))

        # Check if lap completed (crossed start/finish line)
        lap_completed = False
        if self.car.position_s >= self.track.track_length:
            self.car.position_s %= self.track.track_length
            lap_completed = True
            self.lap_time = 0.0  # Reset lap time for next lap

            # Update best lap time
            if self.lap_time < self.best_lap_time:
                self.best_lap_time = self.lap_time

        # Check if car crashed (outside track limits)
        track_width = self.track.get_width_at_s(self.car.position_s)
        self.crashed = abs(self.car.position_d) > track_width/2

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action, lap_completed)

        # Check if episode is done
        done = self.crashed or self.current_step >= self.max_steps

        # Additional info
        info = {
            "lap_time": self.lap_time,
            "best_lap_time": self.best_lap_time,
            "lap_completed": lap_completed,
            "crashed": self.crashed,
            "speed": self.car.speed,
            "tire_wear": self.car.tire_wear,
            "position": (self.car.position_s, self.car.position_d),
            "telemetry": car_state,  # Ensure telemetry data is included
            "racing_line": self.racing_line  # Ensure racing line data is included
        }

        # Return observation, reward, done, truncated, info
        return observation, reward, done, False, info

    def _calculate_reward(self, action, lap_completed):
        """Calculate reward based on current state and action."""
        # Base reward is forward progress
        reward = self.car.speed * self.dt

        # Additional reward for completing a lap
        if lap_completed:
            reward += 1000.0 / self.lap_time  # Faster lap time = higher reward

        # Penalize being far from centerline (quadratic penalty)
        # But allow some deviation for racing line optimization
        track_width = self.track.get_width_at_s(self.car.position_s)
        centered_d = self.car.position_d / (track_width/2)  # Normalize by track width
        center_penalty = 0.1 * centered_d**2
        reward -= center_penalty

        # Penalize excessive steering, throttle changes, and brake use for smooth driving
        if self.last_action is not None:
            steering, throttle, brake = action
            prev_steering, prev_throttle, prev_brake = self.last_action

            # Penalize steering changes (for smooth steering)
            steering_change = abs(steering - prev_steering)
            reward -= 0.5 * steering_change

            # Penalize throttle changes (for smooth acceleration)
            throttle_change = abs(throttle - prev_throttle)
            reward -= 0.3 * throttle_change

            # Small penalty for using brakes (encourage momentum conservation)
            reward -= 0.2 * brake

        # Large penalty for crashing
        if self.crashed:
            reward -= 100.0

        # Scale reward to appropriate range
        return reward

    def _get_observation(self):
        """Get current observation state."""
        # Normalize s to [0, 1]
        normalized_s = (self.car.position_s % self.track.track_length) / self.track.track_length

        # Normalize d based on track width at current position
        track_width = self.track.get_width_at_s(self.car.position_s)
        normalized_d = (2 * self.car.position_d) / track_width

        # Get current track curvature
        curvature = self.track.get_curvature_at_s(self.car.position_s)

        # Create observation vector
        observation = np.array([
            normalized_s,           # Position along track (0-1)
            normalized_d,           # Lateral position (-1 to 1)
            self.car.speed,         # Speed (m/s)
            self.car.slip_angle,    # Slip angle (radians)
            curvature,              # Track curvature at current position
            track_width             # Track width at current position
        ], dtype=np.float32)

        return observation

    def render(self):
        """Visualize current state (simplified)."""
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.plot(self.track.x, self.track.y, 'k-', linewidth=1)
            self.ax.set_aspect('equal')
            self.ax.set_title(f'F1 Racing Simulation - {self.track.name}')
            self.ax.grid(True)

            # Plot car position
            x, y = self.track.frenet_to_cartesian(self.car.position_s, self.car.position_d)
            self.car_point = self.ax.plot(x, y, 'ro', markersize=8)[0]

            # Racing line (will be updated)
            self.line_plot = self.ax.plot([], [], 'g-', linewidth=2)[0]

            plt.tight_layout()
            plt.show(block=False)

        # Update car position
        x, y = self.track.frenet_to_cartesian(self.car.position_s, self.car.position_d)
        self.car_point.set_data(x, y)

        # Update racing line
        if len(self.racing_line) > 0:
            x_line, y_line = [], []
            for s, d in self.racing_line:
                x, y = self.track.frenet_to_cartesian(s, d)
                x_line.append(x)
                y_line.append(y)

            # Ensure we have valid sequences before updating the plot
            if len(x_line) > 0 and len(y_line) > 0:
                self.line_plot.set_data(x_line, y_line)

        # Add text for current speed and lap time
        self.ax.set_title(
            f'F1 Racing - {self.track.name} - '
            f'Speed: {self.car.speed:.1f} m/s, '
            f'Lap time: {self.lap_time:.2f}s'
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.ioff()

    def simulate_optimal_racing_line(self, model, render=True):
        """Simulate a lap with a trained model to get the optimal racing line."""
        # Reset environment
        obs, _ = self.reset()
        done = False

        # Store telemetry for analysis
        telemetry = {
            'distance': [],
            'speed': [],
            'lat_position': [],
            'curvature': [],
            'steering': [],
            'throttle': [],
            'brake': []
        }

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Store telemetry
            telemetry['distance'].append(self.car.position_s)
            telemetry['speed'].append(self.car.speed)
            telemetry['lat_position'].append(self.car.position_d)
            telemetry['curvature'].append(self.track.get_curvature_at_s(self.car.position_s))
            telemetry['steering'].append(action[0])
            telemetry['throttle'].append(action[1])
            telemetry['brake'].append(action[2])

            # Step environment
            obs, reward, done, _, info = self.step(action)

            if render:
                self.render()

            # Stop if lap completed
            if info.get('lap_completed', False):
                break

        # Create a pandas DataFrame for analysis
        df = pd.DataFrame(telemetry)

        # Plot the racing line on the track
        if render:
            self.track.plot_track(racing_line=self.racing_line)

            # Plot speed profile
            plt.figure(figsize=(12, 6))
            plt.subplot(211)
            plt.plot(df['distance'], df['speed'])
            plt.title('Speed Profile')
            plt.ylabel('Speed (m/s)')
            plt.grid(True)

            plt.subplot(212)
            plt.plot(df['distance'], df['steering'], label='Steering')
            plt.plot(df['distance'], df['throttle'], label='Throttle')
            plt.plot(df['distance'], df['brake'], label='Brake')
            plt.title('Control Inputs')
            plt.xlabel('Distance (m)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return df, self.racing_line