import argparse
from stable_baselines3 import PPO
from f1_env import F1RacingEnv
import os

def train_model(tire_compound):
    # Skip training if model exists
    if os.path.exists(f"models/ppo_racing_{tire_compound}_tires_marina_bay.zip"):
        print(f"\nModel for {tire_compound} tires already exists, skipping training...")
        return
    
    print(f"\nTraining with {tire_compound} tires on Marina Bay Street Circuit...")
    env = F1RacingEnv(track_name="Singapore", year=2023, tire_compound=tire_compound)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(f"models/ppo_racing_{tire_compound}_tires_marina_bay")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train F1 Racing Model')
    parser.add_argument('--tire', type=str, default="soft", choices=['soft', 'medium', 'hard'],
                      help='Tire compound to train with')
    
    args = parser.parse_args()
    train_model(args.tire) 