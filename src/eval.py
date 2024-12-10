import torch
import numpy as np
from env import Grid
from dqn_conv_dis import MultiDroneDoubleQL
import matplotlib.pyplot as plt

def evaluate_pretrained_model(model_path, num_episodes=100):

    checkpoint = torch.load(model_path)

    env = Grid(
        x_list=[5],
        agents_list=[3],
        fov_x=3,
        fov_y=3
    )

    agent = MultiDroneDoubleQL(
        env=env,
        action_size=4,
        hidden_size=checkpoint['hidden_size']
    )

    agent.Q_primary.load_state_dict(checkpoint['Q_primary_state_dict'])
    agent.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
    agent.Q_primary.eval()
    agent.Q_target.eval()

    coverage_history = []
    steps_history = []
    rewards_history = []
    collisions_history = []
    efficiency_history = []
    per_drone_steps_history = []

    clean_coverage_history = []
    clean_steps_history = []
    clean_rewards_history = []
    clean_efficiency_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        drone_histories = [[] for _ in range(agent.n_drones)]
        stuck_counts = env.stuck_counts
        steps = 0
        collisions = 0
        done = False

        while not done and steps < env.x_size * env.y_size * 10:
            for drone_idx in range(agent.n_drones):
                pre_grid = env.grid_status.copy()
                pre_position = env.agent_pos[drone_idx].copy()
                pre_orientation = env.agent_orientations[drone_idx]

                action, _ = agent.select_action(
                    state,
                    drone_idx,
                    epsilon=0,
                    stuck_counts=stuck_counts,
                    max_stuck=20,
                    training=False
                )

                next_state, reward, done, collision = env.step(action, drone_idx)

                post_grid = env.grid_status.copy()
                post_position = env.agent_pos[drone_idx].copy()
                post_orientation = env.agent_orientations[drone_idx]

                pre_unmapped = (pre_grid == 1).sum()
                post_unmapped = (post_grid == 1).sum()
                cells_mapped = pre_unmapped - post_unmapped

                drone_histories[drone_idx].append({
                    'step': steps,
                    'pre_grid': pre_grid,
                    'post_grid': post_grid,
                    'cells_mapped': cells_mapped,
                    'action': action,
                    'pre_position': pre_position,
                    'post_position': post_position,
                    'pre_orientation': pre_orientation,
                    'post_orientation': post_orientation
                })

                episode_reward += reward
                steps += 1
                stuck_counts = env.stuck_counts
                state = next_state
                collisions += collision

                if done or collision:
                    break

            if done or collision:
                break

        mapped = (env.grid_status == 0).sum()
        total_cells = env.x_size * env.y_size
        coverage = mapped / total_cells
        drone_last_useful_steps = [-1] * agent.n_drones
        for drone_idx, history in enumerate(drone_histories):
            for i, event in enumerate(reversed(history)):
                if event['cells_mapped'] > 0:
                    drone_last_useful_steps[drone_idx] = len(history) - i - 1
                    break

        # Truncate the histories at the last useful step for each drone
        truncated_histories = []
        for drone_idx, history in enumerate(drone_histories):
            last_step = drone_last_useful_steps[drone_idx]
            if last_step != -1: 
                truncated_histories.append(history[:last_step + 1])
            else:
                truncated_histories.append([])

        total_useful_steps = sum(len(history) for history in truncated_histories)
        efficiency = coverage * total_cells / max(total_useful_steps+agent.n_drones, 1)

        coverage_history.append(coverage)
        steps_history.append(total_useful_steps)
        rewards_history.append(episode_reward)
        collisions_history.append(collisions)
        efficiency_history.append(efficiency)
        per_drone_steps_history.append([len(drone_histories[i]) for i in range(agent.n_drones)])

        if collisions == 0:
            clean_coverage_history.append(coverage)
            clean_steps_history.append(total_useful_steps)
            clean_rewards_history.append(episode_reward)
            clean_efficiency_history.append(efficiency)

        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Coverage: {coverage:.2f}")
            print(f"Total steps: {steps}")
            print(f"Total useful steps: {total_useful_steps}")
            print(f"Collisions: {collisions}")
            print(f"Efficiency: {efficiency:.3f}")

    print("\nEvaluation Results:")
    print(f"Average Coverage: {np.mean(coverage_history):.3f} ± {np.std(coverage_history):.3f}")
    print(f"Average Total Useful Steps: {np.mean(steps_history):.1f} ± {np.std(steps_history):.1f}")
    print(f"Average Reward: {np.mean(rewards_history):.2f} ± {np.std(rewards_history):.2f}")
    print(f"Average Collisions: {np.mean(collisions_history):.2f} ± {np.std(collisions_history):.2f}")
    print(f"Average Efficiency: {np.mean(efficiency_history):.3f} ± {np.std(efficiency_history):.3f}")


    return {
        'all_episodes': {
            'coverage': coverage_history,
            'steps': steps_history,
            'rewards': rewards_history,
            'collisions': collisions_history,
            'efficiency': efficiency_history
        },
        'clean_episodes': {
            'coverage': clean_coverage_history,
            'steps': clean_steps_history,
            'rewards': clean_rewards_history,
            'efficiency': clean_efficiency_history
        }
    }



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    results = evaluate_pretrained_model(
        model_path='drone_dqn_model_5.pth',
        num_episodes=100
    )
