import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import os
import multiprocessing
import time

class GridWorldEnv:
    EMPTY = 0
    WALL = 1
    APPLE = 2
    PREDATOR = 3
    AGENT = 4  # Added for visualization
    APPLE_TREE = 5  # New constant for apple tree tiles

    def __init__(self, grid_size=20, view_size=5, max_hunger=100, num_predators=1, num_trees=1):
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_hunger = max_hunger
        self.num_predators = num_predators
        self.num_trees = num_trees
        self.previous_predator_distance = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        # Initialize the grid with walls (1) around the borders
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = self.WALL

        # Generate multiple apple trees
        max_tree_start = self.grid_size - 5 - 1  # -1 to account for the walls
        self.apple_trees = []
        occupied_positions = set()
        for _ in range(self.num_trees):
            overlap = True
            attempts = 0
            while overlap:
                tree_x = np.random.randint(1, max_tree_start + 1)
                tree_y = np.random.randint(1, max_tree_start + 1)
                apple_tree_positions = []
                for i in range(tree_x, tree_x + 5):
                    for j in range(tree_y, tree_y + 5):
                        apple_tree_positions.append((i, j))
                overlap = any(pos in occupied_positions for pos in apple_tree_positions)
                attempts += 1
                if attempts > 100:  # Prevent infinite loops
                    print("Could not place all apple trees without overlap.")
                    break
            if attempts > 100:
                break
            occupied_positions.update(apple_tree_positions)
            self.apple_trees.append(apple_tree_positions)
            for pos in apple_tree_positions:
                self.grid[pos[0], pos[1]] = self.APPLE_TREE

        # Remove apple tree positions from empty cells
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        # apple_tree_set = set(self.apple_tree_positions)
        apple_tree_set = set(pos for tree in self.apple_trees for pos in tree)
        empty_cells = [cell for cell in empty_cells if tuple(cell) not in apple_tree_set]
        empty_cells = np.array(empty_cells)

        # Place the agent randomly in empty cells (excluding apple tree positions)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]

        # Remove the agent's position from empty_cells
        empty_cells = empty_cells[~np.all(empty_cells == self.agent_pos, axis=1)]

        # Place predators
        self.predator_positions = []
        self.predator_underlying_cells = []  # New list to store underlying cells
        for _ in range(self.num_predators):
            if len(empty_cells) > 0:
                predator_pos = empty_cells[np.random.choice(len(empty_cells))]
                self.predator_positions.append(predator_pos)
                underlying_cell = self.grid[predator_pos[0], predator_pos[1]]
                self.predator_underlying_cells.append(underlying_cell)
                self.grid[predator_pos[0], predator_pos[1]] = self.PREDATOR
                # Remove predator position from empty_cells
                empty_cells = empty_cells[~np.all(empty_cells == predator_pos, axis=1)]
            else:
                break  # No empty cell to place a predator

        # Place an apple randomly within the apple tree
        # Ensure the apple doesn't spawn on the agent or predators
        occupied_positions = [tuple(self.agent_pos)] + [tuple(pos) for pos in self.predator_positions]

        self.apple_timer = 0
        self.apple_positions = []
        self.generate_apples()

        self.hunger = 0
        self.done = False
        self.steps = 0
        self.previous_predator_distance = -1

        return self._get_observation()
    
    def generate_apples(self):
        occupied_positions = [tuple(self.agent_pos)] + [tuple(pos) for pos in self.predator_positions]
        for tree in self.apple_trees:
            # Exclude positions occupied by agent or predators
            available_positions = [pos for pos in tree if pos not in occupied_positions]
            if available_positions:
                apple_pos = random.choice(available_positions)
                self.apple_positions.append(apple_pos)
                self.grid[apple_pos[0], apple_pos[1]] = self.APPLE

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done

        reward = 0

        # Map action to movement
        movement = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = movement[action]
        next_pos = self.agent_pos + np.array(move)

        # Check for wall collision
        if self.grid[next_pos[0], next_pos[1]] != self.WALL:
            self.agent_pos = next_pos

        # Check for apple consumption
        #print(tuple(self.agent_pos))
        #print(self.apple_positions)
        if tuple(self.agent_pos) in self.apple_positions:
            reward += 5
            self.hunger = 0
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.APPLE_TREE
            self.apple_positions.remove(tuple(self.agent_pos))
        else: 
            self.hunger += 1

        # Check if the agent dies due to hunger
        if self.hunger >= self.max_hunger:
            reward += -10  # Negative reward for dying
            self.done = True

        # Move predators (if any)
        # Remove predators from their old positions in the grid, restoring the underlying cell
        for idx, pos in enumerate(self.predator_positions):
            self.grid[pos[0], pos[1]] = self.predator_underlying_cells[idx]

        new_predator_positions = []
        new_predator_underlying_cells = []

        for idx, pos in enumerate(self.predator_positions):
            # Decide movement for predator
            # Check if the agent is within 10 tiles (Manhattan distance)
            distance_to_agent = np.abs(pos - self.agent_pos).sum()
            if distance_to_agent <= 10:
                # 70% chance to move towards the agent
                if np.random.rand() < 0.7:
                    # Move towards the agent
                    delta = self.agent_pos - pos
                    move_options = []
                    if delta[0] > 0:
                        move_options.append((1, 0))  # Down
                    elif delta[0] < 0:
                        move_options.append((-1, 0))  # Up
                    if delta[1] > 0:
                        move_options.append((0, 1))  # Right
                    elif delta[1] < 0:
                        move_options.append((0, -1))  # Left
                    if move_options:
                        move = random.choice(move_options)
                    else:
                        move = (0, 0)  # Predator is on the agent's position
                else:
                    # Move randomly
                    move = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            else:
                # Move randomly
                move = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

            next_pos = pos + np.array(move)
            # Check for wall collision, other predators, and apple
            if self.grid[next_pos[0], next_pos[1]] not in [self.WALL, self.PREDATOR, self.APPLE]:
                # Before moving, record the underlying cell at the next position
                underlying_cell = self.grid[next_pos[0], next_pos[1]]
                pos = next_pos
                # Update underlying cell
                new_predator_underlying_cells.append(underlying_cell)
            else:
                # Position remains the same, underlying cell remains the same
                new_predator_underlying_cells.append(self.predator_underlying_cells[idx])

            new_predator_positions.append(pos)

        # Update predator positions and underlying cells
        self.predator_positions = new_predator_positions
        self.predator_underlying_cells = new_predator_underlying_cells

        # Place predators in the grid
        for pos in self.predator_positions:
            self.grid[pos[0], pos[1]] = self.PREDATOR

        # Check for collision with agent
        for pos in self.predator_positions:
            # Check if adjacent (distance <= 1)
            distance_to_agent = np.abs(pos - self.agent_pos).sum()
            if distance_to_agent <= 1:
                # Agent is adjacent to predator
                reward += -10
                self.done = True
                break
            elif distance_to_agent <= 3:
                reward += -2*(4 - distance_to_agent)  # Negative reward for being close to a predator

            if (self.previous_predator_distance != -1):
                if distance_to_agent > self.previous_predator_distance:
                    reward += 2
            if distance_to_agent > (self.view_size - 1):
                self.previous_predator_distance = -1
            else:
                self.previous_predator_distance = distance_to_agent

        if self.done:
            obs = self._get_observation()
            return obs, reward, self.done, {}

        # Increment apple timer and generate apples if needed
        self.apple_timer += 1
        if self.apple_timer >= 20:
            self.generate_apples()
            self.apple_timer = 0

        self.steps += 1
        obs = self._get_observation()

        info = {
            'agent_pos': self.agent_pos.copy(),
            'apple_positions': self.apple_positions.copy()
        }
        # print("Reward: " + str(reward))
        return obs, reward, self.done, info


    def _get_observation(self):
        # Extract a 5x5 observation around the agent, excluding the agent's position
        x, y = self.agent_pos
        half_view = self.view_size // 2
        min_x, max_x = x - half_view, x + half_view + 1
        min_y, max_y = y - half_view, y + half_view + 1

        # Handle edge cases with padding
        pad_min_x = max(0, -min_x)
        pad_min_y = max(0, -min_y)
        pad_max_x = max(0, max_x - self.grid_size)
        pad_max_y = max(0, max_y - self.grid_size)

        # Convert the grid to a PyTorch tensor if not already
        grid_tensor = torch.tensor(self.grid, device=self.device, dtype=torch.float32)

        # Extract the observation window
        obs = grid_tensor[
            max(0, min_x):min(max_x, self.grid_size),
            max(0, min_y):min(max_y, self.grid_size)
        ]

        # Apply padding
        obs = F.pad(obs, (pad_min_y, pad_max_y, pad_min_x, pad_max_x), value=self.WALL)

        # Flatten the observation and remove the agent's position
        obs_flat = obs.flatten()
        agent_idx = (self.view_size * self.view_size) // 2  # Index of the agent's position
        obs_flat = torch.cat([obs_flat[:agent_idx], obs_flat[agent_idx+1:]])  # Remove agent position
        return obs_flat  # Returns a tensor of length 24

        #### OLD CODE
        # obs = self.grid[
        #     max(0, min_x):min(max_x, self.grid_size),
        #     max(0, min_y):min(max_y, self.grid_size)
        # ]

        # obs = np.pad(obs, ((pad_min_x, pad_max_x), (pad_min_y, pad_max_y)), 'constant', constant_values=self.WALL)
        # obs_flat = obs.flatten()
        # agent_idx = (self.view_size * self.view_size) // 2  # Index of the agent's position
        # obs_flat = np.delete(obs_flat, agent_idx)  # Remove the agent's own position
        # return obs_flat  # Returns an array of length 24

