# Intelligent Microrobotic Swarm in Unknown Environments

This repository contains the code for an unpublished manuscript.

This work presents a reinforcement learning-based approach for intelligent navigation of microrobotic swarms in unknown environments with dynamic obstacles. The system utilizes Proximal Policy Optimization (PPO) with a Transformer architecture to train agents capable of autonomous exploration and navigation in complex, dynamic environments. The training framework supports both static and dynamic obstacle scenarios, making it suitable for real-world applications in microrobotics.

## System Requirements

### Training Environment
For training the reinforcement learning models:

- **Operating System**: Ubuntu 20.04 (recommended) or Windows 11
- **Python Environment**: Miniconda or Anaconda
- **CUDA**: Version 12.2
- **Python**: 3.10
- **Development Environment**: Visual Studio Code

### Deployment Environment
For real-world deployment:

- **Operating System**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4060
- **Python Environment**: Miniconda
- **LabVIEW**: Version 2019
- **Development Environment**: Visual Studio Code

### Required Python Packages

**Training Dependencies**:
- PyTorch (with CUDA support)
- Gymnasium 
- NumPy
- PyYAML
- TensorBoard
- Pygame (for visualization)

**Deployment Dependencies**:
- OpenCV (cv2)
- PyTorch (with CUDA support)
- NumPy
- Socket programming libraries
- YOLOv5 dependencies

## Project Structure

```
src/
├── train/                   # Training pipeline
│   ├── agent/              # RL agent implementation
│   │   ├── agent.py       # Main agent class
│   │   ├── ppo.py         # PPO algorithm implementation
│   │   └── transformer.py # Transformer network architecture
│   ├── config/            # Configuration files
│   │   └── dynamic_obstacle.yaml # Training hyperparameters
│   ├── environment/       # Environment implementation
│   │   ├── dynamic_obstacle_env.py # Main environment with dynamic obstacles
│   │   └── env_wrapper.py # Environment wrapper utilities
│   ├── utils/             # Utility functions
│   │   ├── buffer.py     # Experience buffer
│   │   ├── worker.py     # Parallel worker implementation
│   │   └── plugin.py     # Additional plugins
│   ├── models/            # Pre-trained models
│   ├── logs/              # Training logs and TensorBoard files
│   ├── main.py            # Main training script
│   └── register_env.py    # Environment registration
└── deploy/                 # Deployment pipeline
    ├── inference.py        # Main inference and control script
    ├── register_env.py     # Environment registration
    ├── agent/              # Trained agent components
    │   ├── agent.py        # Agent implementation
    │   ├── ppo.py          # PPO policy network
    │   └── transformer.py # Transformer architecture
    ├── config/             # Deployment configuration
    │   ├── dynamic_obstacle.yaml # Deployment parameters
    │   └── yaml_parser.py # Configuration parser
    ├── environment/        # Environment interfaces
    │   └── env_wrapper.py # Environment wrapper
    ├── models/             # Pre-trained models
    │   ├── yolov5.pt      # YOLOv5 detection model
    │   └── Turbo.pt       # Trained RL policy model
    └── tool/               # Utility functions
        ├── buffer.py      # Data buffer management
        ├── plugin.py      # Environment creation utilities
        └── worker.py      # Parallel processing
```

## Training Configuration

The training uses the following key hyperparameters:

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network Architecture**: Transformer with 3 blocks, 4 attention heads
- **Parallel Workers**: 120 workers
- **Steps per Worker**: 1024
- **Training Updates**: 4050
- **Learning Rate**: 3e-4 → 5e-5 (scheduled)
- **Environment**: Dynamic obstacle environment with both static and moving obstacles

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n microrobotic_swarm python=3.10 -y
conda activate microrobotic_swarm

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

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

## Results and Performance

Training results and performance metrics can be monitored through:

1. **TensorBoard Logs**: Real-time training progress in `logs/` directory
2. **Model Checkpoints**: Saved models in `models/` directory  
3. **Console Output**: Training statistics and episode information

## Deployment and Real-World Implementation

### Hardware System Overview

The deployment system utilizes a sophisticated hardware setup for real-world microrobotic swarm control:

**Three-Axis Helmholtz Coil System**: A three-axis Helmholtz coil system is used to globally actuate the swarm, which can generate a uniform magnetic field in any plane. This system comprises several key components:

- **Observation Device**: Digital microscope with light source for video stream acquisition via USB interface
- **Current Detection Device**: ACS712 current conversion chip, 5V DC power supply, and USB3202N I/O card
- **Current Generation Device**: High-power DC power supply (CSP-3000-120), power amplifier (JSP-180-30), and NI MyRIO-1900 driver
- **Host Control System**: Receives digital signals from I/O card and implements closed-loop control algorithms developed in LabVIEW 2019

### YOLOv5 Setup

The deployment system uses YOLOv5 for real-time object detection. To set up YOLOv5:

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Install requirements
pip install -r requirements.txt

# Update the model path in inference.py to your local YOLOv5 directory
```

**Note**: Update the YOLOv5 path in `src/deploy/inference.py` line 82 to match your local installation.

### Running the Deployment System

1. **Hardware Setup**: Ensure all hardware components are properly connected and LabVIEW control system is running.

2. **Model Preparation**: Place your trained models in the `src/deploy/models/` directory:
   - `yolov5.pt`: YOLOv5 detection model
   - `Turbo.pt`: Trained RL policy model

3. **Start Inference**: Navigate to the deployment directory and run:

```bash
cd src/deploy
python inference.py
```

4. **System Operation**: The system will:
   - Capture real-time video from the digital microscope
   - Detect microrobotic swarms and obstacles using YOLOv5
   - Generate control actions using the trained RL policy
   - Send control commands via UDP to the LabVIEW control system
   - Log attention data and system performance metrics


## Acknowledgments

We thank the YOLOv5 team for their excellent object detection framework, which forms a crucial component of our real-time detection system.

## Contributing

This codebase supports research in:
- Microrobotics navigation and control
- Reinforcement learning for microrobotic swarm
- Dynamic obstacle avoidance
- Transformer-based policy networks

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

The citation will be available after publication.