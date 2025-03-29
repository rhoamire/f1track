# F1 Racing AI Simulation

This project implements an AI-powered Formula 1 racing simulation using reinforcement learning. The AI agent learns to race around the Monza circuit with different tire compounds, optimizing lap times while managing tire wear and grip levels.

## Features

- **Realistic F1 Physics**: Simulates realistic racing physics including:
  - Tire grip and wear
  - G-force calculations
  - Speed and acceleration dynamics
  - Different tire compounds (Soft, Medium, Hard)

- **AI Training**: Uses PPO (Proximal Policy Optimization) reinforcement learning to train the AI agent
- **Multiple Tire Compounds**: Supports all three F1 tire compounds with realistic performance differences
- **Real-time Telemetry**: Displays live racing data including:
  - Speed
  - Grip levels
  - G-forces
  - Lap times
  - Track position

## Requirements

- Python 3.8+
- FastF1 (for F1 telemetry data)
- Stable-Baselines3 (for reinforcement learning)
- Gymnasium (for environment simulation)
- Matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rhoamire/f1track.git
cd f1track
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with a specific tire compound:
```bash
python train.py --tire soft  # or medium/hard
```

### Evaluating Performance

To evaluate all trained models and compare lap times:
```bash
python main.py
```

This will:
1. Test each tire compound
2. Display real-time telemetry
3. Show lap times
4. Generate a comparison plot

## Project Structure

- `f1_env.py`: The racing environment implementation
- `train.py`: Training script for the AI model
- `test_model.py`: Evaluation and testing script
- `main.py`: Main script for running evaluations
- `models/`: Directory containing trained models
- `cache/`: Directory for FastF1 telemetry data

## Results

The AI agent achieves realistic lap times around Monza circuit and can be used to evaluate track times on other circuits too:
- Soft tires: ~85-86 seconds
- Medium tires: ~86-87 seconds
- Hard tires: ~87-88 seconds

## License

This project is licensed under the MIT License - see the LICENSE file for details.
