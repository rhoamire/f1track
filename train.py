import os
from stable_baselines3 import PPO
from f1_env import F1RacingEnv

def train_model(tire_compound="soft"):
    print(f"\nTraining with {tire_compound} tires...")
    env = F1RacingEnv(tire_compound=tire_compound)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000)
    model.save(f"ppo_racing_{tire_compound}_tires")
    env.close()

if __name__ == "__main__":
    # Create cache directory for FastF1
    os.makedirs("f1_cache", exist_ok=True)
    
    # Train on different tire compounds
    for tire in ["soft", "medium", "hard"]:
        train_model(tire_compound=tire) 