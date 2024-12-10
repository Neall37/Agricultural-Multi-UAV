import numpy as np
from env import Grid
from dqn_conv_dis import MultiDroneDoubleQL
import torch
import matplotlib.pyplot as plt
import random

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    # Environment parameters
    MAP_SIZE = [5]
    NUM_DRONES = [2]
    ACTION_SIZE = 4  # TURN_LEFT, TURN_RIGHT, GO_STRAIGHT, TURN_BACK

    # Training parameters
    HIDDEN_SIZE = 128
    NUM_EPISODES = 2500
    MAX_STUCK = 20 # Not actually used
    MAX_STEP = 200

    # Create environment
    env = Grid(
        x_list=MAP_SIZE,
        agents_list=NUM_DRONES,
        fov_x=3,
        fov_y=3
    )

    # Create agent
    agent = MultiDroneDoubleQL(
        env=env,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.001,
        buffer_size=10000,
        batch_size=128,
        weight_decay=1e-3
    )

    # Train agent
    rewards, metrics = agent.train(
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEP,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=0.998,
        max_stuck=MAX_STUCK
    )

    # Save model
    torch.save({
        'Q_primary_state_dict': agent.Q_primary.state_dict(),
        'Q_target_state_dict': agent.Q_target.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'map_size': MAP_SIZE,
        'n_drones': NUM_DRONES,
        'hidden_size': HIDDEN_SIZE
    }, 'drone_dqn_model_5.pth')

    # Plot
    plt.figure(figsize=(15, 5))


    plt.subplot(1, 2, 1)
    window_size = min(100, len(rewards) // 10)
    if len(rewards) > window_size:
        ma_rewards = moving_average(rewards, window_size)
        plt.plot(range(window_size-1, len(rewards)), ma_rewards, label='Moving Average')
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    if len(metrics) > window_size:
        ma_metrics = moving_average(metrics, window_size)
        plt.plot(range(window_size-1, len(metrics)), ma_metrics, label='Moving Average')
    plt.plot(metrics, alpha=0.3, label='Raw')
    plt.title('Coverage Efficiency Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency Metric')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    main()
