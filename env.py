import numpy as np
import gym
from gym import spaces
import random

import collections
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Grid(gym.Env):
    metadata = {'render.modes': ['console']}
    # Orientations
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    # Actions
    TURN_LEFT = 0
    TURN_RIGHT = 1
    GO_STRAIGHT = 2
    TURN_BACK = 3

    def __init__(self, x_list, agents_list, fov_x=3, fov_y=3):
        super(Grid, self).__init__()

        self.x_list = x_list
        self.agents_list = agents_list

        self.max_x_size = max(x_list)
        self.max_agents = max(agents_list)

        self.x_size = random.choice(x_list)
        self.n_agents = random.choice(agents_list)
        self.y_size = self.x_size
        self.orient_values = [0, 1, 2, 3]

        self.fov_x = fov_x
        self.fov_y = fov_y

        self.setup_spaces()

        self.init_grid()
        self.init_agent()

    def setup_spaces(self):
        """Setup action and observation spaces to handle variable number of agents"""
        n_actions = 4

        self.action_space = spaces.Tuple([
            spaces.Discrete(n_actions) for _ in range(self.max_agents)
        ])

        self.obs_low = -np.ones(4) * 2
        self.obs_high = np.ones(4)
        self.observation_space = spaces.Dict({
            'local_obs': spaces.Box(
                low=-2,
                high=1,
                shape=(self.max_agents, 4),
                dtype=np.float32
            ),
            'global_state': spaces.Dict({
                'drone_positions': spaces.Box(
                    low=0,
                    high=self.max_x_size - 1,
                    shape=(self.max_agents, 2),
                    dtype=np.int32
                ),
                'drone_orientations': spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.max_agents,),
                    dtype=np.int32
                ),
                'grid_status': spaces.Box(
                    low=-2,
                    high=1,
                    shape=(self.max_x_size, self.max_x_size),
                    dtype=np.int32
                )
            })
        })
      

    def init_agent(self, initial_pos=None):
        """Initialize agent positions and orientations"""
        self.agent_pos = []
        self.agent_orientations = []

        if initial_pos is not None and len(initial_pos) == self.n_agents:
            self.agent_pos = initial_pos
            for i in range(self.n_agents):
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 0
                orientation = random.choice(self.orient_values)
                self.agent_orientations.append(orientation)
        else:
            for i in range(self.n_agents):
                while True:
                    agent_pos_x = random.randrange(0, self.x_size)
                    agent_pos_y = random.randrange(0, self.x_size)
                    if not any(pos == [agent_pos_x, agent_pos_y] for pos in self.agent_pos):
                        break
                orientation = random.choice(self.orient_values)
                self.agent_orientations.append(orientation)
                self.agent_pos.append([agent_pos_x, agent_pos_y])
                self.grid_status[agent_pos_x, agent_pos_y] = 0

        self.stuck_counts = [0] * self.n_agents

    def init_grid(self):
        # initialize the mapping status
        ## -1: out of the grid or obstacle
        ## 0: POI that is covered
        ## 1: POI that is unmapped
        self.grid_status = np.ones([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        # area of the environment
        self.n_poi = self.x_size * self.y_size

    def get_coverage(self):
        mapped_poi = (self.grid_status == 0).sum()
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        """Get observations padded to maximum possible agents"""
        local_obs = self.get_local_obs()

        padded_local_obs = np.zeros((self.max_agents, 10))
        padded_local_obs[:self.n_agents] = local_obs

        # Pad positions and orientations
        padded_positions = np.zeros((self.max_agents, 2), dtype=np.int32)
        padded_positions[:self.n_agents] = self.agent_pos

        padded_orientations = np.zeros(self.max_agents, dtype=np.int32)
        padded_orientations[:self.n_agents] = self.agent_orientations

        padded_grid = np.zeros((self.max_x_size, self.max_x_size))-1
        padded_grid[:self.x_size, :self.x_size] = self.grid_status

        return {
            'local_obs': padded_local_obs,
            'global_state': {
                'drone_positions': padded_positions,
                'drone_orientations': padded_orientations,
                'grid_status': padded_grid
            }
        }

    def get_local_obs(self):
        # Pad the grid_status with -1 for boundary cases
        padded_grid = np.pad(self.grid_status, pad_width=1, mode='constant', constant_values=-1)

        self.agent_obs = []

        # Get 3*3 local view for each agent
        for agent in range(self.n_agents):

            x, y = self.agent_pos[agent][0] + 1, self.agent_pos[agent][1] + 1
  
            local_obs = padded_grid[x - 1:x + 2, y - 1:y + 2].flatten()
            local_obs = np.append(self.agent_orientations[agent], local_obs)
            self.agent_obs.append(local_obs)

        return self.agent_obs

    def reset(self, initial_pos=None):
        """Reset environment"""

        self.x_size = random.choice(self.x_list)
        self.y_size = self.x_size
        self.n_agents = random.choice(self.agents_list)

        self.init_grid()
        self.init_agent(initial_pos)

        return self.get_agent_obs()


    def get_new_position(self, x, y, i):
        """Calculate new position based on orientation"""
        if self.agent_orientations[i] == self.NORTH:
            return x, y + 1
        elif self.agent_orientations[i] == self.SOUTH:
            return x, y - 1
        elif self.agent_orientations[i] == self.EAST:
            return x + 1, y
        elif self.agent_orientations[i] == self.WEST:
            return x - 1, y
        return x, y

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

        current_orientation = copy.deepcopy(self.agent_orientations[i])
        reward = 0
      
        # move the agent
        if action == self.TURN_LEFT:
            reward -= 0.1
            self.agent_orientations[i] = (current_orientation - 1) % 4
        elif action == self.TURN_RIGHT:
            reward -= 0.1
            self.agent_orientations[i] = (current_orientation + 1) % 4
        elif action == self.TURN_BACK:
            reward -= 0.1
            self.agent_orientations[i] = (current_orientation + 2) % 4
        else:
            self.agent_orientations[i] = current_orientation

        self.agent_pos[i][0], self.agent_pos[i][1] = self.get_new_position(org_x, org_y, i)

        if self.check_collision(i):
            collision = True
            reward -= 1
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y
            self.agent_orientations[i] = current_orientation
        else:
            collision = False

        # account for the boundaries
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][
            1] > self.y_size - 1 or self.agent_pos[i][1] < 0:
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y
            self.agent_orientations[i] = current_orientation

            self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
            reward -= 1
        else:
            # previous status of the cell
            prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
            if prev_status == -1:  # Obstacles
                # go back
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.agent_orientations[i] = current_orientation
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward -= 1
            elif prev_status == 1:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 0
                reward += 5
            elif prev_status == 0:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward -= 0.1

        # update the stuck count
        if org_x == self.agent_pos[i][0] and org_y == self.agent_pos[i][1]:  # stuck
            self.stuck_counts[i] += 1
        else:
            self.stuck_counts[i] = 0

        mapped_poi = (self.grid_status == 0).sum()
        done = bool(mapped_poi == self.n_poi)

        if done:
            reward += 50

        return self.get_agent_obs(), reward, done, collision

    def close(self):
        pass
