# Agricultural-Multi-UAV

This project introduces a cutting-edge deep reinforcement learning framework for multi-agent coverage path planning in agricultural UAVs. 

Leveraging a scalable Double Deep Q-Network architecture, our approach integrates a customizable training environment, shared state representations, and adaptive neural networks with distance-based spatial transformations and agriculture-specific rewards. Experimental results demonstrate superior efficiency, achieving optimal coverage with minimal steps while significantly outperforming traditional methods in generating structured, straight-line trajectories. Although collision avoidance remains a challenge at higher agent densities, our framework showcases strong scalability across varying map sizes and agent numbers, highlighting its potential for broader applications in multi-agent robotic systems beyond agriculture.

Please read the full report [here](https://drive.google.com/file/d/1NqEvxvRnIfHt01aYO9z7x0KmSDCxX7TW/view?usp=sharing)

<img src="images/pipeline.png" alt="Image text">

# Usage

## Training

To train the framework, run 

```
python experiment.py
```
