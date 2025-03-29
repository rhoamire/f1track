import os
import logging
from test_model import test_racing_line

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("gymnasium").setLevel(logging.ERROR)

# Choose a track
track_name = "Monza"  # You can also try "monaco", "silverstone", etc.

# Paths for model and normalization
model_path = "models/f1_racing/final_model"
env_path = "models/f1_racing/vec_normalize.pkl"

# Check if the model exists, if not, train a new model
if not os.path.exists(f"{model_path}.zip"):
    print(f"Model does not exist at {model_path}.zip. Please train the model first.")
else:
    print(f"Model already exists at {model_path}.zip. Skipping training.")

# Test trained model
print(f"Testing racing line on {track_name}...")
telemetry, racing_line = test_racing_line(
    model_path=model_path,
    env_path=env_path,
    track_name=track_name
)

print("Done!")