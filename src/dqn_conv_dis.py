import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from conv import QNet
from scipy.ndimage import distance_transform_edt, zoom


def scale_map_and_positions(coverage_map, positions=None):
    """ 
    Scale a map to 20*20 and adjust positions accordingly.
    """
  
    target_size = [20, 20]

    scale_y = target_size[0] / coverage_map.shape[0]
    scale_x = target_size[1] / coverage_map.shape[1]

    scaled_map = zoom(coverage_map, (scale_y, scale_x), order=0)

    if positions is None:
        return scaled_map, None
    else:
        scaled_positions = [(int(pos[0] * scale_x), int(pos[1] * scale_y)) for pos in positions]
        return scaled_map, scaled_positions



def generate_gaussian_grid(positions, x_size, current_drone_idx, n_agents, active_amplitude=1.0, other_amplitude=0.8,
                           sigma=None):
    if sigma is None:
        sigma = x_size * 0.1

    x = np.linspace(0, x_size - 1, x_size)
    y = np.linspace(0, x_size - 1, x_size)
    X, Y = np.meshgrid(x, y)

    grid = np.zeros((x_size, x_size))

    for i in range(n_agents):
        pos_x, pos_y = positions[i]
        amplitude = active_amplitude if i == current_drone_idx else other_amplitude
        gaussian = amplitude * np.exp(
            -((X - pos_x) ** 2 + (Y - pos_y) ** 2) / (2 * sigma ** 2)
        )
        grid += gaussian

    return grid


def generate_distance_grid(positions, coverage_map, current_drone_idx, n_agents,
                                    drone_amplitude=1.0, normalize=True, max_distance=None):

    scaled_map, scaled_positions = scale_map_and_positions(coverage_map, positions)
    x_size, y_size = scaled_map.shape

    if max_distance is None:
        max_distance = np.sqrt(2) * max(x_size, y_size)

    binary_grid = np.zeros((x_size, x_size), dtype=bool)

    for i in range(n_agents):
        if i != current_drone_idx:
            pos_x, pos_y = scaled_positions[i]
            pos_x, pos_y = int(pos_x), int(pos_y)
            if 0 <= pos_x < x_size and 0 <= pos_y < y_size:
                binary_grid[pos_y, pos_x] = True

    distances = distance_transform_edt(~binary_grid)

    if normalize:
        if distances.max() > 0:
            distances = distances / max_distance
            distances = 1 - distances

    distance_field = distances * drone_amplitude
    distance_field = np.clip(distance_field, 0, drone_amplitude)

    return distance_field


def coverage_to_distance_field(coverage_map, normalize=True):
    """
    Calculate distance field for uncovered areas.
    """
    scaled_map, scaled_positions = scale_map_and_positions(coverage_map)
    uncovered_mask = (scaled_map == 1)
    uncovered_distances = distance_transform_edt(~uncovered_mask)

    if normalize and uncovered_distances.max() > 0:
        uncovered_distances = uncovered_distances / uncovered_distances.max()
        uncovered_distances = 1 - uncovered_distances

    return uncovered_distances


class ReplayBuffer:
    def __init__(self, capacity, max_x_size, device):
        self.device = device
        self.capacity = capacity
        self.max_x_size = max_x_size

        self.distance_fields = torch.zeros((capacity,max_x_size, max_x_size), device=device)
        self.next_distance_fields = torch.zeros((capacity, max_x_size, max_x_size), device=device)
      
        self.gaussian_grids = torch.zeros((capacity, max_x_size, max_x_size), device=device)
        self.next_gaussian_grids = torch.zeros((capacity, max_x_size, max_x_size), device=device)

        self.current_local = torch.zeros((capacity, 10), device=device)
        self.next_local = torch.zeros((capacity, 10), device=device)


        self.drone_positions = torch.zeros((capacity, 2), device=device)
        self.next_drone_positions = torch.zeros((capacity, 2), device=device)

        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.pos = 0
        self.size = 0

    def push(self, n_drones, state, action, reward, next_state, done, drone_idx):
        current_distance_field = coverage_to_distance_field(state['global_state']['grid_status'])
        next_distance_field = coverage_to_distance_field(next_state['global_state']['grid_status'])
      
        self.current_local[self.pos] = torch.FloatTensor(state['local_obs'][drone_idx]).to(self.device)
        self.next_local[self.pos] = torch.FloatTensor(next_state['local_obs'][drone_idx]).to(self.device)
      
        self.distance_fields[self.pos] = torch.FloatTensor(current_distance_field).to(self.device)
        self.next_distance_fields[self.pos] = torch.FloatTensor(next_distance_field).to(self.device)

        self.gaussian_grids[self.pos] = torch.FloatTensor(
            generate_distance_grid(state['global_state']['drone_positions'], state['global_state']['grid_status'], drone_idx, n_drones)
        ).to(self.device)
        self.next_gaussian_grids[self.pos] = torch.FloatTensor(
            generate_distance_grid(next_state['global_state']['drone_positions'], next_state['global_state']['grid_status'], drone_idx, n_drones)
        ).to(self.device)

        self.drone_positions[self.pos] = torch.FloatTensor(state['global_state']['drone_positions'][drone_idx]).to(
            self.device)
        self.next_drone_positions[self.pos] = torch.FloatTensor(next_state['global_state']['drone_positions'][drone_idx]).to(
            self.device)

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.distance_fields[indices],
            self.gaussian_grids[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_distance_fields[indices],
            self.next_gaussian_grids[indices],
            self.dones[indices],
            self.drone_positions[indices],
            self.next_drone_positions[indices],
            self.current_local[indices],
            self.next_local[indices]
        )


class MultiDroneDoubleQL:
    def __init__(self, env, action_size, hidden_size=128,
                 learning_rate=5e-4, gamma=0.99, tau=0.001, buffer_size=10000,
                 batch_size=128, weight_decay=1e-3):
                   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.x_size = env.x_size
        self.n_drones = env.n_agents
        self.max_x_size = env.max_x_size

        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize Q-Net
        self.Q_primary = QNet(action_size, hidden_size).to(self.device)
        self.Q_target = QNet(action_size, hidden_size).to(self.device)
        self.Q_target.load_state_dict(self.Q_primary.state_dict())

        self.optimizer = optim.AdamW(
            self.Q_primary.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.memory = ReplayBuffer(buffer_size, 20, self.device)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, drone_idx, epsilon, stuck_counts=None, max_stuck=20, training=True):
        # if stuck_counts is not None and stuck_counts[drone_idx] >= max_stuck:
        #     action = random.randint(0, self.action_size - 1)
        #     return action, None

        drone_position = state['global_state']['drone_positions'][drone_idx]

        distance_field = coverage_to_distance_field(state['global_state']['grid_status'])
        gaussian_grid = generate_distance_grid(
            state['global_state']['drone_positions'],
            state['global_state']['grid_status'],
            drone_idx,
            self.n_drones
        )

        local = state['local_obs'][drone_idx]

        with torch.no_grad():
            distance_field = torch.FloatTensor(distance_field).unsqueeze(0).to(self.device)
            gaussian_grid = torch.FloatTensor(gaussian_grid).unsqueeze(0).to(self.device)
            drone_position = torch.FloatTensor(drone_position).unsqueeze(0).to(self.device)
            local = torch.FloatTensor(local).unsqueeze(0).to(self.device)


            self.Q_primary.eval()
            q_values = self.Q_primary(
                distance_field,
                gaussian_grid,
                drone_position,
                local
            )
            self.Q_primary.train()
            q_values = q_values.cpu().numpy()[0]

        if training and np.random.random() < epsilon:
            action = random.randint(0, self.action_size - 1)
            action_value = q_values[action]
        else:
            action = np.argmax(q_values)
            action_value = q_values[action]

        return action, action_value

    def update(self, drone_idx):
        if self.memory.size < self.batch_size:
            return None

        (distance_fields, gaussian_grids, actions, rewards,
         next_distance_fields, next_gaussian_grids, dones,
         drone_positions, next_drone_positions, current_local,
            next_local) = self.memory.sample(self.batch_size)

        # Get current Q values
        current_q = self.Q_primary(distance_fields, gaussian_grids, drone_positions, current_local)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Get target Q values
        with torch.no_grad():
            next_actions = self.Q_primary(
                next_distance_fields,
                next_gaussian_grids,
                next_drone_positions,
                next_local
            ).argmax(1, keepdim=True)

            next_q = self.Q_target(
                next_distance_fields,
                next_gaussian_grids,
                next_drone_positions,
                next_local
            )
            next_q = next_q.gather(1, next_actions)

            target_q = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1).float()) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_primary.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.Q_target.parameters(), self.Q_primary.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def train(self, num_episodes, max_steps, epsilon_start=1.0, epsilon_final=0.05,
              epsilon_decay=0.997, max_stuck=20):
        env = self.env
        episode_rewards = []
        episode_metrics = []
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = env.reset()
            self.x_size = env.x_size
            self.n_drones = env.n_agents
            total_reward = 0
            done = False
            steps = 0
            stuck_counts = env.stuck_counts  # Get stuck counts from environment

            for step in range(max_steps):
                for drone_idx in range(self.n_drones):

                    action, action_value = self.select_action(
                        state,
                        drone_idx,
                        epsilon,
                        stuck_counts=stuck_counts,
                        max_stuck=max_stuck,
                        training=True
                    )

                    next_state, reward, done, crush = env.step(action, drone_idx)

                    # Store transition
                    self.memory.push(self.n_drones, state, action, reward, next_state, done, drone_idx)

                    # Update networks
                    if steps % 4 == 0:
                        loss = self.update(drone_idx)

                    total_reward += reward
                    state = next_state
                    steps += 1
                    stuck_counts = env.stuck_counts

                    if done:
                        break
                if done:
                    break

            epsilon = max(epsilon_final, epsilon * epsilon_decay)

            # metrics
            mapped_poi = (state['global_state']['grid_status'] == 0).sum()
            efficiency_metric = mapped_poi / max(steps, 1)
            coverage = mapped_poi / (self.x_size**2)

            episode_rewards.append(total_reward)
            episode_metrics.append(efficiency_metric)

            if episode % 2 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
                print(f"Episode {episode + 1}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Efficiency Metric: {efficiency_metric:.2f}")
                print(f"Steps: {steps}")
                print(f"Loss: {loss}")
                print(f"Coverage: {coverage}")
                print(f"Epsilon: {epsilon:.3f}")
                print(f"Number of agents: {self.n_drones}, Map size:{self.x_size}")
                print("------------------------")

        return episode_rewards, episode_metrics
