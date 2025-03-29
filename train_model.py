from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from f1_racing_env import F1RacingEnv

def train_f1_racing_agent(track_name="monza", total_steps=1_000_000, save_path="models/f1_racing"):
    # Create environment
    env = F1RacingEnv(track_name=track_name)

    # Wrap environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./f1_racing_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]}
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // env.num_envs,
        save_path=save_path,
        name_prefix="ppo_f1_racing"
    )

    # Train the model
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback
    )

    # Save final model
    model.save(f"{save_path}/final_model")
    env.save(f"{save_path}/vec_normalize.pkl")

    return model, env