# Intelligent Microrobotic Swarm in Unknown Environments

This repository is related to an unpublished manuscript.

This work presents a reinforcement learning-based approach for intelligent navigation of microrobotic swarms in unknown environments with dynamic obstacles. The system utilizes Proximal Policy Optimization (PPO) with Transformer architecture to train agents capable of autonomous exploration and navigation in complex, dynamic environments. The training framework supports both static and dynamic obstacle scenarios, making it suitable for real-world applications in microrobotics.

## System Requirements

For training and running the reinforcement learning models:

- **Operating System**: Ubuntu 20.04 (recommended) or Windows 10+
- **Python Environment**: Miniconda or Anaconda
- **CUDA**: Version 12.2 (for GPU acceleration)
- **Python**: 3.8+

### Required Python Packages

- PyTorch (with CUDA support)
- Gymnasium 
- NumPy
- PyYAML
- TensorBoard
- Pygame (for visualization)

## Project Structure

```
src/train/
├── agent/                    # RL agent implementation
│   ├── agent.py             # Main agent class
│   ├── ppo.py               # PPO algorithm implementation
│   └── transformer.py      # Transformer network architecture
├── config/                  # Configuration files
│   └── dynamic_obstacle.yaml # Training hyperparameters
├── environment/             # Environment implementation
│   ├── dynamic_obstacle_env.py # Main environment with dynamic obstacles
│   └── env_wrapper.py       # Environment wrapper utilities
├── utils/                   # Utility functions
│   ├── buffer.py           # Experience buffer
│   ├── worker.py           # Parallel worker implementation
│   └── plugin.py           # Additional plugins
├── models/                  # Pre-trained models
├── logs/                    # Training logs and TensorBoard files
├── main.py                  # Main training script
└── register_env.py         # Environment registration
```

## Training Configuration

The training uses the following key hyperparameters:

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network Architecture**: Transformer with 3 blocks, 4 attention heads
- **Parallel Workers**: 120 workers
- **Steps per Worker**: 1024
- **Training Updates**: 4050
- **Learning Rate**: 3e-4 → 5e-5 (scheduled)
- **Environment**: Dynamic obstacle environment with static and moving obstacles

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n microrobot python=3.8
conda activate microrobot

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install gymnasium numpy pyyaml tensorboard pygame
```

### 2. Training

Navigate to the training directory and run:

```bash
cd src/train
python main.py
```

The training process will:
- Load configuration from `config/dynamic_obstacle.yaml`
- Initialize the dynamic obstacle environment
- Train the PPO agent with Transformer architecture
- Save training logs to the `logs/` directory
- Save trained models to the `models/` directory

### 3. Monitoring Training Progress

View training progress using TensorBoard:

```bash
cd src/train
tensorboard --logdir=./logs
```

Then open your browser and navigate to `http://localhost:6006` to view:
- Training loss curves
- Reward progression
- Policy performance metrics
- Network statistics

## Environment Details

### Dynamic Obstacle Environment

The environment simulates a 2D space where:

- **Agent**: Microrobotic swarm represented as an observation entity
- **Obstacles**: Both static and dynamic obstacles with collision detection
- **Goal**: Navigate to target positions while avoiding obstacles
- **Observation Space**: 35-dimensional vector including:
  - Agent position, velocity, and dimensions
  - Target position and distance
  - Nearest obstacle information (position, size, velocity)
- **Action Space**: 2D continuous control (speed and direction)

### Key Features

- **Dynamic Obstacles**: Moving obstacles with realistic physics
- **Collision Detection**: Sophisticated overlap detection and avoidance
- **Batch Obstacle Generation**: Efficient obstacle placement algorithms
- **Configurable Scenarios**: Adjustable static/dynamic obstacle ratios
- **Real-time Visualization**: Pygame-based rendering for training monitoring

## Model Architecture

### Transformer-based Policy Network

- **Input Processing**: 35D observation → 256D embedding
- **Transformer Blocks**: 3 layers with 4 attention heads each
- **Memory Length**: 128 steps for temporal context
- **Output**: Continuous action distribution with learned variance

### Training Algorithm

- **PPO**: Stable policy gradient method
- **Parallel Training**: 120 concurrent workers
- **Experience Buffer**: 1024 steps per worker
- **Mini-batch Training**: 16 mini-batches per update
- **Gradient Clipping**: Max norm of 1.0

## Usage Examples

### Basic Training

```python
from agent import Agent
from config import YamlParser

# Load configuration
config = YamlParser(path='config/dynamic_obstacle.yaml').get_config()

# Initialize agent
agent = Agent(config=config, model_id='Turbo', device='cuda')

# Start training
agent.train()
```

### Environment Testing

```python
from environment.env_wrapper import DynamicObstacleEnvWrapper

# Create environment
env = DynamicObstacleEnvWrapper(render_mode='human')

# Test random actions
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        observation, info = env.reset()
```

## Results and Performance

Training results and performance metrics can be monitored through:

1. **TensorBoard Logs**: Real-time training progress in `logs/` directory
2. **Model Checkpoints**: Saved models in `models/` directory  
3. **Console Output**: Training statistics and episode information

## Contributing

This codebase supports research in:
- Microrobotics navigation and control
- Multi-agent reinforcement learning
- Dynamic obstacle avoidance
- Transformer-based policy networks

## License

[Include your license information here]

## Citation

If you use this code in your research, please cite our related manuscript:

```
[Citation will be added upon publication]
```

## Contact

For questions and collaborations, please contact [your contact information].