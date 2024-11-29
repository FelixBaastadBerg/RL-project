# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

class GridWorldEnv:
    EMPTY = 0
    WALL = 1
    APPLE = 2
    PREDATOR = 3
    AGENT = 4  # Added for visualization
    APPLE_TREE = 5  # New constant for apple tree tiles

    def __init__(self, grid_size=20, view_size=5, max_hunger=100):
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_hunger = max_hunger
        self.reset()

    def reset(self):
        # Initialize the grid with walls (1) around the borders
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = self.WALL

        # Generate the apple tree
        max_tree_start = self.grid_size - 5 - 1  # -1 to account for the walls
        tree_x = np.random.randint(1, max_tree_start + 1)
        tree_y = np.random.randint(1, max_tree_start + 1)
        self.apple_tree_positions = []
        for i in range(tree_x, tree_x + 5):
            for j in range(tree_y, tree_y + 5):
                self.apple_tree_positions.append((i, j))
                self.grid[i, j] = self.APPLE_TREE  # Mark the apple tree on the grid

        # Remove apple tree positions from empty cells
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        apple_tree_set = set(self.apple_tree_positions)
        empty_cells = [cell for cell in empty_cells if tuple(cell) not in apple_tree_set]
        empty_cells = np.array(empty_cells)

        # Place the agent randomly in empty cells (excluding apple tree positions)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]

        # Remove the agent's position from empty_cells
        empty_cells = empty_cells[~np.all(empty_cells == self.agent_pos, axis=1)]

        # Place predators
        self.predator_positions = []
        self.predator_underlying_cells = []  # New list to store underlying cells
        num_predators = 1  # Adjust the number of predators as needed
        for _ in range(num_predators):
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
        available_apple_positions = [pos for pos in self.apple_tree_positions if pos not in occupied_positions]
        if available_apple_positions:
            self.apple_pos = available_apple_positions[np.random.choice(len(available_apple_positions))]
            self.grid[self.apple_pos[0], self.apple_pos[1]] = self.APPLE
        else:
            self.apple_pos = None  # No available position in the apple tree

        self.hunger = 0
        self.done = False
        self.steps = 0
        return self._get_observation()

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
        if self.apple_pos is not None and np.array_equal(self.agent_pos, self.apple_pos):
            reward = 1  # Reward for eating an apple
            self.hunger = 0  # Reset hunger
            self.grid[self.apple_pos[0], self.apple_pos[1]] = self.APPLE_TREE  # Reset to apple tree tile

            # Place a new apple within the apple tree
            occupied_positions = [tuple(self.agent_pos)] + [tuple(pos) for pos in self.predator_positions]
            available_apple_positions = [pos for pos in self.apple_tree_positions if pos not in occupied_positions]
            if available_apple_positions:
                self.apple_pos = available_apple_positions[np.random.choice(len(available_apple_positions))]
                self.grid[self.apple_pos[0], self.apple_pos[1]] = self.APPLE
            else:
                self.apple_pos = None  # No available position in the apple tree
        else:
            self.hunger += 1

        # Check if the agent dies due to hunger
        if self.hunger >= self.max_hunger:
            reward = -1  # Negative reward for dying
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
                # 50% chance to move towards the agent
                if np.random.rand() < 0.5:
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
            if np.array_equal(pos, self.agent_pos):
                # Predator is at the same position as the agent
                reward = -1  # Negative reward similar to dying of hunger
                self.done = True
                break
            else:
                # Check if adjacent (distance <= 1)
                distance_to_agent = np.abs(pos - self.agent_pos).sum()
                if distance_to_agent <= 1:
                    # Agent is adjacent to predator
                    reward = -1
                    self.done = True
                    break

        if self.done:
            obs = self._get_observation()
            return obs, reward, self.done

        self.steps += 1
        obs = self._get_observation()
        return obs, reward, self.done

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

        obs = self.grid[
            max(0, min_x):min(max_x, self.grid_size),
            max(0, min_y):min(max_y, self.grid_size)
        ]

        obs = np.pad(obs, ((pad_min_x, pad_max_x), (pad_min_y, pad_max_y)), 'constant', constant_values=self.WALL)
        obs_flat = obs.flatten()
        agent_idx = (self.view_size * self.view_size) // 2  # Index of the agent's position
        obs_flat = np.delete(obs_flat, agent_idx)  # Remove the agent's own position
        return obs_flat  # Returns an array of length 24


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128):
        super(PolicyValueNetwork, self).__init__()
        self.hidden_size = hidden_size

        # First hidden layer after the input
        self.fc1 = nn.Linear(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Actor (policy) head with two hidden layers
        self.actor_fc1 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

        # Critic (value) head with two hidden layers
        self.critic_fc1 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hx, cx):
        x = x.float() / 5.0  # Normalize input (max value is 5)
        x = F.relu(self.fc1(x))  # First hidden layer

        x = x.unsqueeze(1)  # Add time dimension for LSTM: (batch_size, seq_len=1, hidden_size)
        x, (hx, cx) = self.lstm(x, (hx, cx))  # LSTM layer
        x = x.squeeze(1)  # Remove time dimension: (batch_size, hidden_size)

        # Actor (policy) head
        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        policy_logits = self.policy_head(actor_x)

        # Critic (value) head
        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value = self.value_head(critic_x)

        return policy_logits, value.squeeze(-1), (hx, cx)

class PPOAgent:
    def __init__(self, num_envs=100, num_steps=128, num_updates=2000, hidden_size = 128, grid_size=20, view_size=5, max_hunger=100):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_hunger = max_hunger

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.learning_rate = 2.5e-4
        self.eps = 1e-5

        self.envs = [GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger) for _ in range(num_envs)]
        self.input_size = 24  # 5x5 grid minus the agent's own position
        self.num_actions = 4
        self.hidden_size = hidden_size  # Hidden size for LSTM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyValueNetwork(self.input_size, self.num_actions, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=self.eps)

        self.all_rewards = []
        self.agent_positions = []  # To track positions of the first agent
        self.apple_positions = []  # To track positions of the apple

        # Initialize LSTM hidden states (num_layers=1)
        self.hx = torch.zeros(1, self.num_envs, self.hidden_size, device=self.device)
        self.cx = torch.zeros(1, self.num_envs, self.hidden_size, device=self.device)

    def collect_rollouts(self, track_positions=False):
        obs_list, actions_list, log_probs_list = [], [], []
        values_list, rewards_list, dones_list = [], [], []
        hxs_list, cxs_list = [], []

        obs = np.array([env.reset() for env in self.envs])
        obs = torch.tensor(obs, device=self.device)

        # Initialize position tracking if needed
        if track_positions:
            positions = []
            apple_positions = []

        for step in range(self.num_steps):
            with torch.no_grad():
                # Pass observations and hidden states to the network
                policy_logits, value, (hx, cx) = self.policy(obs, self.hx, self.cx)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Store the collected data
            obs_list.append(obs.cpu())
            actions_list.append(action.cpu())
            log_probs_list.append(log_prob.cpu())
            values_list.append(value.cpu())
            hxs_list.append(self.hx.squeeze(0).cpu())
            cxs_list.append(self.cx.squeeze(0).cpu())

            # Update hidden states
            self.hx = hx.detach()
            self.cx = cx.detach()

            obs_np = []
            rewards = []
            dones = []

            for i, env in enumerate(self.envs):
                ob, reward, done = env.step(action[i].item())

                # Track positions of the first agent and the apple
                if track_positions and i == 0:
                    positions.append(tuple(env.agent_pos))
                    if env.apple_pos is not None:
                        apple_positions.append(env.apple_pos)
                    else:
                        apple_positions.append(None)

                if done:
                    ob = env.reset()
                    # Reset hidden states for this environment
                    self.hx[:, i, :] = torch.zeros_like(self.hx[:, i, :])
                    self.cx[:, i, :] = torch.zeros_like(self.cx[:, i, :])

                obs_np.append(ob)
                rewards.append(reward)
                dones.append(done)

            obs = torch.tensor(obs_np, device=self.device)
            rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
            dones_list.append(torch.tensor(dones, dtype=torch.float32))

        # Collect the last value estimation for GAE computation
        with torch.no_grad():
            _, next_value, _ = self.policy(obs, self.hx, self.cx)
        next_value = next_value.cpu()

        # Store positions if tracking
        if track_positions:
            self.agent_positions = positions
            self.apple_positions = apple_positions

        return (obs_list, actions_list, log_probs_list, values_list,
                rewards_list, dones_list, next_value, hxs_list, cxs_list)

    def compute_gae(self, rewards_list, values_list, dones_list, next_value):
        rewards = torch.stack(rewards_list)
        values = torch.stack(values_list)
        dones = torch.stack(dones_list)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    def update_policy(self, obs_batch, actions_batch, log_probs_old_batch,
                      returns_batch, advantages_batch, hxs_batch, cxs_batch):
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Perform PPO update
        batch_size = self.num_envs * self.num_steps
        minibatch_size = 256
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            obs_mb = obs_batch[mb_indices].to(self.device)
            actions_mb = actions_batch[mb_indices].to(self.device)
            old_log_probs_mb = log_probs_old_batch[mb_indices].to(self.device)
            returns_mb = returns_batch[mb_indices].to(self.device)
            advantages_mb = advantages_batch[mb_indices].to(self.device)
            hxs_mb = hxs_batch[mb_indices].unsqueeze(0).to(self.device)  # Add num_layers dimension
            cxs_mb = cxs_batch[mb_indices].unsqueeze(0).to(self.device)

            # Forward pass with LSTM hidden states
            policy_logits, value, _ = self.policy(obs_mb, hxs_mb, cxs_mb)
            dist = torch.distributions.Categorical(logits=policy_logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions_mb)

            ratio = torch.exp(new_log_probs - old_log_probs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, returns_mb)
            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss.item(), actor_loss, value_loss, entropy

    def train(self):
        for update in range(self.num_updates):
            # Set track_positions=True during the last rollout
            track_positions = (update == self.num_updates - 1)

            (obs_list, actions_list, log_probs_list, values_list,
             rewards_list, dones_list, next_value, hxs_list, cxs_list) = self.collect_rollouts(track_positions=track_positions)

            advantages, returns = self.compute_gae(rewards_list, values_list, dones_list, next_value)

            # Flatten the batch
            obs_batch = torch.stack(obs_list).view(-1, self.input_size)
            actions_batch = torch.stack(actions_list).view(-1)
            log_probs_old_batch = torch.stack(log_probs_list).view(-1)
            returns_batch = returns.view(-1)
            advantages_batch = advantages.view(-1)
            hxs_batch = torch.stack(hxs_list).view(-1, self.hidden_size)
            cxs_batch = torch.stack(cxs_list).view(-1, self.hidden_size)

            loss, actor_loss, value_loss, entropy = self.update_policy(obs_batch, actions_batch, log_probs_old_batch,
                                      returns_batch, advantages_batch, hxs_batch, cxs_batch)

            # Tracking average reward
            avg_reward = torch.stack(rewards_list).sum(0).mean().item()
            self.all_rewards.append(avg_reward)

            if update % 10 == 0:
                print(f'Update {update}, Loss: {loss:.4f}, Avg Reward: {avg_reward:.2f}')
                print(f"Actor Loss: {actor_loss.item()}, Value Loss: {value_loss.item()}, Entropy: {entropy.item()}")


        print("Training completed!")
        self.plot_rewards()
        self.plot_agent_positions()  # Plot the agent's positions
        # Save the trained model
        torch.save(self.policy.state_dict(), 'trained_policy.pth')
        # Run the test environment
        self.test_trained_model()

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.all_rewards)
        plt.title('Average Reward per Update')
        plt.xlabel('Update')
        plt.ylabel('Average Reward')
        plt.grid()
        plt.show()

    def plot_agent_positions(self):
        if not self.agent_positions:
            print("No agent positions to plot.")
            return

        positions = np.array(self.agent_positions)
        x_positions = positions[:, 1]
        y_positions = positions[:, 0]

        # Prepare the figure
        plt.figure(figsize=(8, 8))

        # Plot the agent's positions
        plt.scatter(x_positions, y_positions, c=range(len(x_positions)), cmap='viridis', marker='o', label='Agent')

        # Plot the apple's positions
        apple_positions_filtered = [pos for pos in self.apple_positions if pos is not None]
        if apple_positions_filtered:
            apple_positions_array = np.array(apple_positions_filtered)
            apple_x_positions = apple_positions_array[:, 1]
            apple_y_positions = apple_positions_array[:, 0]
            apple_timesteps = [i for i, pos in enumerate(self.apple_positions) if pos is not None]
            plt.scatter(apple_x_positions, apple_y_positions, c=apple_timesteps, cmap='cool', marker='x', label='Apple')

        # Plot the apple tree positions
        apple_tree_positions = np.array(self.envs[0].apple_tree_positions)
        if len(apple_tree_positions) > 0:
            plt.scatter(apple_tree_positions[:, 1], apple_tree_positions[:, 0], color='brown', marker='s', label='Apple Tree')

        plt.colorbar(label='Timestep')
        plt.title('Agent, Apple, and Apple Tree Movement in the Last Rollout')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim(0, self.envs[0].grid_size)
        plt.ylim(0, self.envs[0].grid_size)
        plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_trained_model(self):
        # Initialize a new environment
        test_env = GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger)
        obs = test_env.reset()
        obs = torch.tensor(obs, device=self.device).unsqueeze(0)
        hx = torch.zeros(1, 1, self.hidden_size, device=self.device)
        cx = torch.zeros(1, 1, self.hidden_size, device=self.device)

        # Load the trained model
        self.policy.load_state_dict(torch.load('trained_policy.pth'))
        self.policy.eval()

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()

        done = False
        step = 0

        def update_plot():
            ax.clear()
            grid = test_env.grid.copy()  # Use test_env.grid instead of test_env.terrain_grid
            grid[test_env.agent_pos[0], test_env.agent_pos[1]] = self.envs[0].AGENT  # Represent agent with AGENT constant

            cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue', 'brown'])
            bounds = [0, 1, 2, 3, 4, 5, 6]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            ax.imshow(grid, cmap=cmap, norm=norm)

            # Create custom legends
            legend_elements = [
                mpatches.Patch(color='white', label='Empty'),
                mpatches.Patch(color='black', label='Wall'),
                mpatches.Patch(color='green', label='Apple'),
                mpatches.Patch(color='red', label='Predator'),
                mpatches.Patch(color='blue', label='Agent'),
                mpatches.Patch(color='brown', label='Apple Tree'),
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Step: {step}')

            plt.draw()
            plt.pause(0.001)

        update_plot()

if __name__ == "__main__":
    agent = PPOAgent(num_envs=100, num_steps=128, num_updates=500, hidden_size=128,
                     grid_size=20, view_size=5, max_hunger=100)
    agent.train()
