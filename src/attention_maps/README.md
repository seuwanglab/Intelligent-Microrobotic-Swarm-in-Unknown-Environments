# Attention Map Visualization

## Overview

This tool visualizes Turbo's multi-head attention mechanism by projecting temporal attention weights onto spatial locations of the swarm, target, and obstacles. Higher intensity indicates higher weight on corresponding historical states.

## Method

Multi-head attention in Turbo operates over temporal dimension: at time *t*, the current hidden state queries a memory window (128 steps) of historical hidden states. The raw attention weights quantify the relevance of historical timesteps to the current decision. We project these temporal weights onto the spatial locations recorded at corresponding timesteps to visualize decision logic with respect to physical obstacles.

## Usage

### 1. Install Dependencies

```bash
pip install pandas matplotlib tqdm opencv-python scipy
```

### 2. Prepare Input Data

Required CSV files:
- `observation.csv` - Swarm, target, and obstacle positions at each timestep
- `block0_head0.csv` - `block0_head3.csv` - Attention weights from 4 heads in the last Transformer block

### 3. Run Visualization

```bash
cd src/attention_maps
python Attention.py
```

### 4. Output

- **Images**: `B0/step####.png` - Individual attention heatmaps for each timestep
- **Video**: `B0.mp4` - Compiled video at 10 fps

## Configuration

Edit `Attention.py` to customize:
- `output_dir` (line 10) - Output directory name
- `canvas_size` (line 12) - Visualization canvas size
- `colors` (line 14) - Custom colormap
- Smoothing parameters (line 130-131) - Gaussian filter sigma values

## Reference

See paper Fig. 6c, 6e and Supporting Information page 28 for detailed explanation of temporal-to-spatial projection methodology.
