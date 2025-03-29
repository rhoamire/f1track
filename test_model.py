from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from f1_racing_env import F1RacingEnv

def test_racing_line(model_path, env_path, track_name="monza"):
    # Create environment
    env = F1RacingEnv(track_name=track_name)

    # Wrap environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(env_path, env)  # Normalize same as training

    # Load model
    model = PPO.load(model_path, env=env)

    # Reset environment
    obs = env.reset()  # Reset without unpacking

    done = False
    telemetry = []
    racing_line = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        telemetry.append(info[0]["telemetry"])
        racing_line.append(info[0]["position"])

    return telemetry, racing_line