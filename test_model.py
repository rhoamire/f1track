import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from f1_env import F1RacingEnv

def test_racing_line(
    model_path: str,
    env_path: str,
    track_name: str = "Monza",
    year: int = 2023,
    n_episodes: int = 5
):
    # Create environment
    env = DummyVecEnv([lambda: F1RacingEnv(track_name, year)])
    
    # Load normalization parameters
    vec_normalize = VecNormalize.load(env_path, venv=env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=vec_normalize)
    
    # Run episodes
    all_positions = []
    all_telemetry = []
    
    for episode in range(n_episodes):
        obs = vec_normalize.reset()[0]
        done = False
        positions = []
        telemetry = []
        
        while not done:
            # Get action from model and ensure correct shape
            action, _ = model.predict(obs, deterministic=True)
            action = np.array(action, dtype=np.float32)
            
            # Ensure action is 2D
            if len(action.shape) == 0:  # Scalar
                action = np.array([[action, 0.0]], dtype=np.float32)
            elif len(action.shape) == 1:  # 1D array
                if action.shape[0] == 1:  # Single element
                    action = np.array([[action[0], 0.0]], dtype=np.float32)
                else:  # Multiple elements
                    action = action.reshape(1, -1)
            
            # Take step in environment
            step_result = vec_normalize.step(action)
            obs = step_result[0]
            done = step_result[2]
            
            # Record position and telemetry
            positions.append(vec_normalize.envs[0].current_pos.copy())
            telemetry.append({
                'speed': np.linalg.norm(vec_normalize.envs[0].current_vel),
                'steering': action[0, 0],  # First element of first row
                'throttle': action[0, 1]   # Second element of first row
            })
        
        all_positions.append(positions)
        all_telemetry.append(telemetry)
    
    vec_normalize.close()
    
    # Calculate average positions and telemetry
    avg_positions = np.mean(all_positions, axis=0)
    avg_telemetry = {
        'speed': np.mean([t['speed'] for t in all_telemetry], axis=0),
        'steering': np.mean([t['steering'] for t in all_telemetry], axis=0),
        'throttle': np.mean([t['throttle'] for t in all_telemetry], axis=0)
    }
    
    return avg_telemetry, avg_positions

def plot_results(telemetry, positions, track_name):
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot racing line
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', label='AI Racing Line')
    ax1.set_title(f'Racing Line - {track_name}')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot speed
    ax2.plot(telemetry['speed'], 'g-', label='Speed')
    ax2.set_title('Speed Profile')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Speed (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot steering and throttle
    ax3.plot(telemetry['steering'], 'r-', label='Steering')
    ax3.plot(telemetry['throttle'], 'b-', label='Throttle')
    ax3.set_title('Control Inputs')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test parameters
    model_path = "models/f1_racing/final_model"
    env_path = "models/f1_racing/vec_normalize.pkl"
    track_name = "Monza"
    
    # Run test
    telemetry, positions = test_racing_line(
        model_path=model_path,
        env_path=env_path,
        track_name=track_name
    )
    
    # Plot results
    plot_results(telemetry, positions, track_name)