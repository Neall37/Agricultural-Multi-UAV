import numpy as np
import gym
from gym import spaces
import random

import copy
import matplotlib.pyplot as plt

class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True


class Grid(gym.Env):
    metadata = {'render.modes': ['console']}
    # action id
    XM = 0  # x minus
    XP = 1  # x plus
    YM = 2  # y minus
    YP = 3  # y plus

    def __init__(self, x_size=5, y_size=5, n_agents=2, fov_x=3, fov_y=3):
        super(Grid, self).__init__()

        # size of 2D grid
        self.x_size = x_size
        self.y_size = y_size

        # number of agents
        self.n_agents = n_agents
        self.idx_agents = list(range(n_agents))

        # initialize the mapping status
        self.init_grid()

        # initialize the position of the agent
        self.init_agent()


        n_actions = 4 
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(self.n_agents)])

        # define observation space (fielf of view)
        self.fov_x = fov_x 
        self.fov_y = fov_y

        self.obs_low = -np.ones(4) * 2  
        self.obs_high = np.ones(4) 
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self.obs_low, self.obs_high) for _ in range(self.n_agents)])

    def init_agent(self, initial_pos=None):
        self.agent_pos = []
        if initial_pos is not None:
            self.agent_pos = initial_pos
            for i in range(self.n_agents):
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
        else:
            for i in range(self.n_agents):
                agent_pos_x = random.randrange(0, self.x_size)
                agent_pos_y = random.randrange(0, self.x_size)
                self.agent_pos.append([agent_pos_x, agent_pos_y])
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1

        self.stuck_counts = [0] * self.n_agents

    def init_grid(self):
        # initialize the mapping status
        ## -2: out of the grid
        ## -1: obstacle
        ## 0: POI that is not mapped
        ## 1: POI that is mapped
        self.grid_status = np.zeros([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        ## randomly set obstacles
        # n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        n_obstacle = 0
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = - 1
            self.grid_counts[x_obstacle, y_obstacle] = - 1

        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)

    def get_coverage(self):
        mapped_poi = (self.grid_status == 1).sum()
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        self.agent_obs = []

        # observation for each agent
        for agent in range(self.n_agents):
            # default: out of the grid
            single_obs = -np.ones([self.fov_x, self.fov_y]) * 2
            for i in range(self.fov_x):  # 0, 1, 2
                for j in range(self.fov_y):  # 0, 1, 2
                    obs_x = self.agent_pos[agent][0] + (i - 1)  # -1, 0, 1
                    obs_y = self.agent_pos[agent][1] + (j - 1)  # -1, 0, 1
                    if obs_x >= 0 and obs_y >= 0 and obs_x <= self.x_size - 1 and obs_y <= self.y_size - 1:
                        single_obs[i][j] = copy.deepcopy(self.grid_status[obs_x][obs_y])
            single_obs_flat = single_obs.flatten()  # convert matrix to list
            # extract the necessary cells
            xm = single_obs_flat[1]
            xp = single_obs_flat[7]
            ym = single_obs_flat[3]
            yp = single_obs_flat[5]
            single_obs_flat = np.array([xm, xp, ym, yp])
            self.agent_obs.append(single_obs_flat)
        return self.agent_obs

    def reset(self, initial_pos=None):
        # initialize the mapping status
        self.init_grid()
        # initialize the position of the agent
        self.init_agent(initial_pos)

        # check if the drones at initial positions are surrounded by obstacles
        while True:
            obs = self.get_agent_obs()
            obs_tf = []
            for i in range(self.n_agents):
                agent_obs_tf = obs[i][0] != 0 and obs[i][1] != 0 and obs[i][2] != 0 and obs[i][3] != 0
                obs_tf.append(agent_obs_tf)
            if any(obs_tf):
                self.init_grid()
                self.init_agent()
            else:
                break

        return self.get_agent_obs()

    def check_collision(self, idx):
        x = self.agent_pos[idx][0]
        y = self.agent_pos[idx][1]
        for i, pos in enumerate(self.agent_pos):
            if i != idx:
                if pos[0] == x and pos[1] == y:
                    return True
        return False


    def step(self, action, i):  # i: index of the drone
      
        org_x = copy.deepcopy(self.agent_pos[i][0])
        org_y = copy.deepcopy(self.agent_pos[i][1])


        if action == self.XM:
            self.agent_pos[i][0] -= 1
        elif action == self.XP:
            self.agent_pos[i][0] += 1
        elif action == self.YM:
            self.agent_pos[i][1] -= 1
        elif action == self.YP:
            self.agent_pos[i][1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        if self.check_collision(i):
            collision = True
        else:
            collision = False

        # account for the boundaries
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][
            1] > self.y_size - 1 or self.agent_pos[i][1] < 0:
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y
            self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
            reward = 0
        else:
            # previous status
            prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
            if prev_status == -1:  # obstacle
                # go back
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0
            elif prev_status == 0:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
                reward = 10
            elif prev_status == 1:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0

        # update the stuck count
        if org_x == self.agent_pos[i][0] and org_y == self.agent_pos[i][1]:
            self.stuck_counts[i] += 1
        else:
            self.stuck_counts[i] = 0

        mapped_poi = (self.grid_status == 1).sum()
        done = bool(mapped_poi == self.n_poi)

        return self.get_agent_obs(), reward, done, collision

    def close(self):
        pass


# multi-agent setting
# each agent has an individual q table

class QTables():
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1):
        self.num_agents = len(observation_space)

        self.observation_space = observation_space
        self.observation_values = [-2, -1, 0, 1]
        self.observation_num = len(self.observation_values)  # 3
        self.observation_length = observation_space[0].shape[0]  # field of view

        self.action_space = action_space
        self.action_values = [0, 1, 2, 3]  # corresponding to the column numbers in q table
        self.action_num = len(self.action_values)  # 4

        self.eps = eps_start  # current epsilon
        self.eps_end = eps_end  # epsilon lower bound
        self.r = r  # decrement rate of epsilon
        self.gamma = gamma  # discount rate
        self.lr = lr  # learning rate

        self.q_tables = []
        for agent_i in range(self.num_agents):
            self.q_tables.append(np.random.rand(self.observation_num ** self.observation_length, self.action_num))

        self.q_table_counts = []
        for agent_i in range(self.num_agents):
            self.q_table_counts.append(np.zeros([self.observation_num ** self.observation_length, self.action_num]))

    # support function: convert the fov to the unique row number in the q table
    def obs_to_row(self, obs_array):
        obs_shift = map(lambda x: x + 2, obs_array)  # add 1 to each element
        obs_power = [v * (self.observation_num ** i) for i, v in
                     enumerate(obs_shift)]  # apply exponentiation to each element
        return sum(obs_power)  # return the sum (results are between 0 and 256)

    def softmax(self, a):
        # deal with overflow
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def get_action(self, observations, agent_i, stuck_counts, max_stuck, e_greedy=True, softmax=False):
        # convert the observation to a row number
        obs_row = self.obs_to_row(observations[agent_i])
        if stuck_counts[agent_i] >= max_stuck:  # random action to avoid stuck
            action = random.choice(self.action_values)
            greedy = False
            action_value = self.q_tables[agent_i][obs_row][action]
        elif e_greedy:  # epsilon greedy for training (e_greedy=True)
            if np.random.rand() < self.eps:
                action = random.choice(self.action_values)
                greedy = False
                action_value = self.q_tables[agent_i][obs_row][action]
            else:
                action = np.argmax(self.q_tables[agent_i][obs_row])
                greedy = True
                action_value = self.q_tables[agent_i][obs_row][action]
        elif softmax:  # (e_greedy=False and softmax=True)
            p = self.softmax(self.q_tables[agent_i][obs_row])
            action = np.random.choice(np.arange(self.action_num), p=p)
            greedy = False
            action_value = self.q_tables[agent_i][obs_row][action]
        else:  # all greedy choices for testing performance
            action = np.argmax(self.q_tables[agent_i][obs_row])
            greedy = True
            action_value = self.q_tables[agent_i][obs_row][action]

        return action, greedy, action_value

    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end:  # lower bound
            self.eps *= self.r

    def train(self, obs, obs_next, action, reward, done, agent_i):
        obs_row = self.obs_to_row(obs[agent_i])
        obs_next_row = self.obs_to_row(obs_next[agent_i])
        act_col = action

        q_current = self.q_tables[agent_i][obs_row][act_col]  # current q value
        q_next_max = np.max(self.q_tables[agent_i][obs_next_row])  # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * reward
        else:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * (
                        reward + self.gamma * q_next_max - q_current)

        # inclement the corresponding count
        self.q_table_counts[agent_i][obs_row][act_col] += 1


from tqdm import tqdm

def train_agent(env, q, train_episodes, max_steps, coverage_threshold=0.90, max_stuck=100000):
    metrics = {
        'time_steps': [],
        'epsilons': [],
        'greedy_rates': [],
        'coverage': [],
        'speed': [],
        'sum_q_values': [],
        'total_reward': [],
        'efficiency': [],
        'mapping_results': [],
        'count_results': []
    }

    pbar = tqdm(range(train_episodes), desc="Training Progress")

    for episode in pbar:
        state = env.reset()
        state = [arr.astype('int') for arr in state]
        eps_tmp = q.eps

        greedy_count = [0] * env.n_agents
        coverage_track = True
        epi_reward = [0] * env.n_agents

        for step in range(max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents)
            for agent_i in action_order:
                action, greedy_tf, action_value = q.get_action(
                    observations=state,
                    agent_i=agent_i,
                    stuck_counts=env.stuck_counts,
                    max_stuck=max_stuck,
                    e_greedy=True,
                    softmax=False
                )

                next_state, reward, done, collision = env.step(action, agent_i)
                next_state = [arr.astype('int') for arr in next_state]

                q.train(state, next_state, action, reward, done, agent_i)

                epi_reward[agent_i] += reward
                greedy_count[agent_i] += greedy_tf * 1

                if done:
                    break

                state = next_state

            current_coverage = env.get_coverage()
            mapped_poi = (env.grid_status == 1).sum()
            efficiency = mapped_poi / max(step + 2, 1) 

            if current_coverage >= coverage_threshold and coverage_track:
                metrics['speed'].append(step)
                coverage_track = False

            if done:
                break

        metrics['time_steps'].append(step + 1)
        metrics['epsilons'].append(eps_tmp)
        metrics['coverage'].append(env.get_coverage())
        metrics['greedy_rates'].append([x / (step + 1) for x in greedy_count])
        metrics['sum_q_values'].append([q.q_tables[0].sum()])
        metrics['total_reward'].append(epi_reward)
        metrics['efficiency'].append(efficiency)
        metrics['mapping_results'].append(env.grid_status.copy())
        metrics['count_results'].append(env.grid_counts.copy())


        pbar.set_postfix({
            'Epsilon': f'{eps_tmp:.3f}',
            'Steps': step + 1,
            'Coverage': f'{metrics["coverage"][-1]:.3f}',
            'Efficiency': f'{efficiency:.3f}',
            'Reward': f'{np.mean(epi_reward):.1f}'
        })

        q.update_eps()

    return q, metrics


def evaluate_agent(env, q, eval_episodes, max_steps, coverage_threshold=0.90, max_stuck=100000):
    metrics = {
        'time_steps': [],
        'coverage': [],
        'efficiency': [],
        'total_reward': [],
        'collision_rate': [],
        'mapping_results': [],
        'count_results': []
    }

    pbar = tqdm(range(eval_episodes), desc="Evaluation Progress")

    for episode in pbar:
        state = env.reset()
        state = [arr.astype('int') for arr in state]
        epi_reward = [0] * env.n_agents
        collisions = 0
        steps = 0
        for step in range(max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents)
            for agent_i in action_order:
                steps += 1
                action, _, _ = q.get_action(
                    observations=state,
                    agent_i=agent_i,
                    stuck_counts=env.stuck_counts,
                    max_stuck=max_stuck,
                    e_greedy=False,
                    softmax=False
                )

                next_state, reward, done, collision = env.step(action, agent_i)
                next_state = [arr.astype('int') for arr in next_state]

                epi_reward[agent_i] += reward
                collisions += int(collision)

                if done or collision:
                    break

                state = next_state



            if done or collision:
                break

        mapped_poi = (env.grid_status == 1).sum()
        efficiency = mapped_poi / max(steps + 2, 1)

        metrics['time_steps'].append(steps + 2)
        metrics['coverage'].append(env.get_coverage())
        metrics['efficiency'].append(efficiency)
        metrics['total_reward'].append(epi_reward)
        metrics['collision_rate'].append(collisions)
        metrics['mapping_results'].append(env.grid_status.copy())
        metrics['count_results'].append(env.grid_counts.copy())

        pbar.set_postfix({
            'Steps': steps + 1,
            'Coverage': f'{metrics["coverage"][-1]:.3f}',
            'Efficiency': f'{efficiency:.3f}',
            'Reward': f'{np.mean(epi_reward):.1f}',
            'Collisions': f'{metrics["collision_rate"][-1]:.3f}'
        })

    return metrics


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    train_episodes = 2000
    eval_episodes = 1000
    max_steps = 1000  
    coverage_threshold = 0.90
    max_stuck = 100000

    # Initialize environment and Q-tables
    env = Grid(x_size=5, y_size=5, n_agents=2, fov_x=3, fov_y=3)
    q = QTables(
        observation_space=env.observation_space,
        action_space=env.action_space,
        eps_start=1,
        eps_end=0.05,
        gamma=0.5,
        r=0.9985,
        lr=0.01
    )

    # Train the agent
    print("\nStarting Training Phase...")
    trained_q, train_metrics = train_agent(
        env=env,
        q=q,
        train_episodes=train_episodes,
        max_steps=max_steps,
        coverage_threshold=coverage_threshold,
        max_stuck=max_stuck
    )

    # Evaluate
    print("\nStarting Evaluation Phase...")
    eval_metrics = evaluate_agent(
        env=env,
        q=trained_q,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        coverage_threshold=coverage_threshold,
        max_stuck=max_stuck
    )


    print("\nTraining Summary:")
    print(f"Average Coverage: {np.mean(train_metrics['coverage']):.3f}")
    print(f"Average Efficiency: {np.mean(train_metrics['efficiency']):.3f}")
    print(f"Average Steps: {np.mean(train_metrics['time_steps']):.1f}")
    print(f"Average Reward: {np.mean([np.mean(r) for r in train_metrics['total_reward']]):.1f}")

    print("\nEvaluation Summary:")
    print(f"Average Coverage: {np.mean(eval_metrics['coverage']):.3f}")
    print(f"Average Efficiency: {np.mean(eval_metrics['efficiency']):.3f}")
    print(f"Average Steps: {np.mean(eval_metrics['time_steps']):.1f}")
    print(f"Average Reward: {np.mean([np.mean(r) for r in eval_metrics['total_reward']]):.1f}")
    print(f"Average Collision Rate: {np.mean(eval_metrics['collision_rate']):.3f}")
