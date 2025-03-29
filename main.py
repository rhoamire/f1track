import os
import logging
from train import train_model
from test_model import test_racing_line, plot_results
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting
from stable_baselines3 import PPO
from f1_env import F1RacingEnv

# Configure logging
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("gymnasium").setLevel(logging.ERROR)

def main():
    # Create cache directory for FastF1
    os.makedirs("f1_cache", exist_ok=True)
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Only evaluate existing models
    evaluate_tire_performance()

def train_model(tire_compound):
    # Skip training if model exists
    if os.path.exists(f"models/ppo_racing_{tire_compound}_tires.zip"):
        print(f"\nModel for {tire_compound} tires already exists, skipping training...")
        return
    
    print(f"\nTraining with {tire_compound} tires...")
    env = F1RacingEnv(tire_compound=tire_compound)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000)
    model.save(f"models/ppo_racing_{tire_compound}_tires")

def evaluate_tire_performance():
    tire_types = ["soft", "medium", "hard"]
    lap_times = {}

    for tire in tire_types:
        if not os.path.exists(f"models/ppo_racing_{tire}_tires.zip"):
            print(f"\nNo model found for {tire} tires, skipping evaluation...")
            continue
            
        model = PPO.load(f"models/ppo_racing_{tire}_tires")
        env = F1RacingEnv(tire_compound=tire)
        obs, _ = env.reset()
        done = False

        track_length = None

        print(f"\nEvaluating {tire} tires\n" + "=" * 50)
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)

            lap_time = info.get("time", 0)
            distance_covered = info.get("distance", 0)

            if track_length is None and "track_length" in info:
                track_length = info["track_length"]
                print(f"Track length: {track_length:.2f} meters")

            if track_length and distance_covered >= track_length * 0.33 and distance_covered < track_length * 0.34:
                print(f"1/3 Lap completed: {lap_time:.2f} sec")
            if track_length and distance_covered >= track_length * 0.66 and distance_covered < track_length * 0.67:
                print(f"2/3 Lap completed: {lap_time:.2f} sec")

        lap_times[tire] = lap_time
        env.render()
        print("\n-------------------------")
        print(f"Lap time for {tire} tires: {lap_time:.2f} seconds")
        print("-------------------------\n")

    if lap_times:  # Only plot if we have times to compare
        plot_lap_times(lap_times)

def plot_lap_times(lap_times):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(lap_times.keys(), lap_times.values(), color=["red", "orange", "gray"])
    ax.set_xlabel("Tire Compound")
    ax.set_ylabel("Lap Time (seconds, lower is better)")
    ax.set_title("Lap Time Comparison Across Tire Compounds")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

    # Save the plot instead of showing it
    plt.savefig("lap_times_comparison.png")
    plt.close(fig)

if __name__ == "__main__":
    main()