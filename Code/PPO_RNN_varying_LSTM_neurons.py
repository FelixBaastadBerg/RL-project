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

class GridWorldEnv:
    EMPTY = 0
    WALL = 1
    APPLE = 2
    PREDATOR = 3
    AGENT = 4
    APPLE_TREE = 5

    def __init__(self, grid_size=20, view_size=5, max_hunger=100, num_predators=1, num_trees=1):
        self.grid_size = grid_size
        self.view_size = view_size
        self.view_size_predator = 10
        self.max_hunger = max_hunger
        self.num_predators = num_predators
        self.num_trees = num_trees
        self.previous_predator_distance = -1
        self.observe_hunger = True  # Whether to include hunger in the observation
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = self.WALL

        max_tree_start = self.grid_size - 5 - 1
        self.apple_trees = []
        occupied_positions = set()

        for _ in range(self.num_trees):
            overlap = True
            attempts = 0
            while overlap:
                tree_x = self.grid_size // 2 - 2
                tree_y = self.grid_size // 2 - 2
                apple_tree_positions = []
                for i in range(tree_x, tree_x + 5):
                    for j in range(tree_y, tree_y + 5):
                        if (i, j) in [
                            (tree_x, tree_y),
                            (tree_x + 4, tree_y),
                            (tree_x, tree_y + 4),
                            (tree_x + 4, tree_y + 4),
                        ]:
                            continue
                        apple_tree_positions.append((i, j))
                overlap = any(pos in occupied_positions for pos in apple_tree_positions)
                attempts += 1
                if attempts > 100:
                    print("Could not place all apple trees without overlap.")
                    break
            if attempts > 100:
                break
            occupied_positions.update(apple_tree_positions)
            self.apple_trees.append(apple_tree_positions)
            for pos in apple_tree_positions:
                self.grid[pos[0], pos[1]] = self.APPLE_TREE

        empty_cells = np.argwhere(self.grid == self.EMPTY)
        apple_tree_set = set(pos for tree in self.apple_trees for pos in tree)
        empty_cells = [cell for cell in empty_cells if tuple(cell) not in apple_tree_set]
        empty_cells = np.array(empty_cells)

        random_tree = self.apple_trees[np.random.choice(len(self.apple_trees))]
        random_tree_part = random_tree[np.random.choice(len(random_tree))]

        offset_x = np.random.randint(-3, 4)
        offset_y = np.random.randint(-3, 4)
        self.agent_pos = [random_tree_part[0] + offset_x, random_tree_part[1] + offset_y]

        self.agent_pos[0] = max(1, min(self.grid_size - 2, self.agent_pos[0]))
        self.agent_pos[1] = max(1, min(self.grid_size - 2, self.agent_pos[1]))

        empty_cells = empty_cells[~np.all(empty_cells == self.agent_pos, axis=1)]
        predator_radius = 15
        middle = self.grid_size // 2
        empty_cells = empty_cells[np.abs(empty_cells[:, 0] - middle) <= predator_radius]
        empty_cells = empty_cells[np.abs(empty_cells[:, 1] - middle) <= predator_radius]

        self.predator_positions = []
        self.predator_underlying_cells = []
        for _ in range(self.num_predators):
            if len(empty_cells) > 0:
                predator_pos = empty_cells[np.random.choice(len(empty_cells))]
                self.predator_positions.append(predator_pos)
                underlying_cell = self.grid[predator_pos[0], predator_pos[1]]
                self.predator_underlying_cells.append(underlying_cell)
                self.grid[predator_pos[0], predator_pos[1]] = self.PREDATOR
                empty_cells = empty_cells[~np.all(empty_cells == predator_pos, axis=1)]
            else:
                break

        self.apple_timer = 0
        self.apple_positions = []
        self.generate_apples()

        self.hunger = 0
        self.done = False
        self.steps = 0
        self.previous_predator_distance = -1
        return self._get_observation()

    def generate_apples(self):
        occupied_positions = [tuple(self.agent_pos)] + [tuple(pos) for pos in self.predator_positions] + [tuple(pos) for pos in self.apple_positions]
        for tree in self.apple_trees:
            max_apple_count = 5
            current_apple_count = sum(self.grid[pos[0], pos[1]] == self.APPLE for pos in tree)
            if current_apple_count >= max_apple_count:
                continue

            available_positions = [pos for pos in tree if pos not in occupied_positions]
            max_apples = 5
            tree_size = 25 - 4
            if available_positions and (len(available_positions) > (tree_size - max_apples)):
                apple_pos = random.choice(available_positions)
                self.apple_positions.append(apple_pos)
                self.grid[apple_pos[0], apple_pos[1]] = self.APPLE

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done

        reward = 0
        movement = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = movement[action]
        next_pos = self.agent_pos + np.array(move)

        if self.grid[next_pos[0], next_pos[1]] != self.WALL:
            self.agent_pos = next_pos

        if tuple(self.agent_pos) in self.apple_positions:
            if self.observe_hunger:
                min_eating_reward = 1.0
                max_eating_reward = 10.0
                scaled_reward = min_eating_reward + (max_eating_reward - min_eating_reward) / self.max_hunger * self.hunger
                reward += scaled_reward
            else:
                reward += 2
            self.hunger = 0
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.APPLE_TREE
            self.apple_positions.remove(tuple(self.agent_pos))
        else:
            self.hunger += 1

        if self.hunger >= self.max_hunger:
            reward += -10
            self.done = True

        for idx, pos in enumerate(self.predator_positions):
            self.grid[pos[0], pos[1]] = self.predator_underlying_cells[idx]

        new_predator_positions = []
        new_predator_underlying_cells = []

        for idx, pos in enumerate(self.predator_positions):
            distance_to_agent = np.abs(pos - self.agent_pos).sum()
            if distance_to_agent <= self.view_size_predator:
                if np.random.rand() < 0.7:
                    delta = self.agent_pos - pos
                    move_options = []
                    if delta[0] > 0:
                        move_options.append((1, 0))
                    elif delta[0] < 0:
                        move_options.append((-1, 0))
                    if delta[1] > 0:
                        move_options.append((0, 1))
                    elif delta[1] < 0:
                        move_options.append((0, -1))
                    if move_options:
                        move = random.choice(move_options)
                    else:
                        move = (0, 0)
                else:
                    move = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            else:
                if np.random.rand() < 0.1:
                    delta = np.array([self.grid_size // 2, self.grid_size // 2]) - pos
                    move_options = []
                    if delta[0] > 0:
                        move_options.append((1, 0))
                    elif delta[0] < 0:
                        move_options.append((-1, 0))
                    if delta[1] > 0:
                        move_options.append((0, 1))
                    elif delta[1] < 0:
                        move_options.append((0, -1))
                    if move_options:
                        move = random.choice(move_options)
                    else:
                        move = (0, 0)
                else:
                    move = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

            next_pos = pos + np.array(move)
            if self.grid[next_pos[0], next_pos[1]] not in [self.WALL, self.PREDATOR, self.APPLE]:
                underlying_cell = self.grid[next_pos[0], next_pos[1]]
                pos = next_pos
                new_predator_underlying_cells.append(underlying_cell)
            else:
                new_predator_underlying_cells.append(self.predator_underlying_cells[idx])

            new_predator_positions.append(pos)

        self.predator_positions = new_predator_positions
        self.predator_underlying_cells = new_predator_underlying_cells

        for pos in self.predator_positions:
            self.grid[pos[0], pos[1]] = self.PREDATOR

        for pos in self.predator_positions:
            if np.array_equal(pos, self.agent_pos):
                reward += -10
                self.done = True
                break
            else:
                distance_to_agent = np.abs(pos - self.agent_pos).sum()
                if distance_to_agent <= 1:
                    reward += -10
                    self.done = True
                    break

        if self.done:
            obs = self._get_observation()
            return obs, reward, self.done

        self.apple_timer += 1
        if self.apple_timer >= 20:
            self.generate_apples()
            self.apple_timer = 0

        self.steps += 1
        obs = self._get_observation()
        return obs, reward, self.done

    def _get_observation(self):
        x, y = self.agent_pos
        half_view = self.view_size // 2
        min_x, max_x = x - half_view, x + half_view + 1
        min_y, max_y = y - half_view, y + half_view + 1

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
        agent_idx = (self.view_size * self.view_size) // 2
        obs_flat = np.delete(obs_flat, agent_idx)
        if self.observe_hunger:
            normalized_hunger = self.hunger / self.max_hunger
            obs_flat = np.append(obs_flat, normalized_hunger)
        return obs_flat


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, lstm_percentage=1.0):
        super(PolicyValueNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_neurons = int(lstm_percentage * hidden_size)
        self.num_ff_neurons = hidden_size - self.num_lstm_neurons

        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # LSTM part if needed
        if self.num_lstm_neurons > 0:
            self.lstm = nn.LSTM(hidden_size, self.num_lstm_neurons, batch_first=True)

        # Feed-forward part if needed
        if self.num_ff_neurons > 0:
            self.fc_ff = nn.Linear(hidden_size, self.num_ff_neurons)

        self.actor_fc1 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)

        self.critic_fc1 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None, cx=None):
        x = x.float() / 5.0
        x = F.relu(self.fc1(x))

        lstm_out = None
        ff_out = None

        if self.num_lstm_neurons > 0:
            x_lstm = x.unsqueeze(1)
            x_lstm, (hx, cx) = self.lstm(x_lstm, (hx, cx))
            x_lstm = x_lstm.squeeze(1)
            lstm_out = x_lstm

        if self.num_ff_neurons > 0:
            ff_out = F.relu(self.fc_ff(x))

        if self.num_lstm_neurons > 0 and self.num_ff_neurons > 0:
            x = torch.cat([lstm_out, ff_out], dim=-1)
        elif self.num_lstm_neurons > 0:
            x = lstm_out
        else:
            x = ff_out

        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        policy_logits = self.policy_head(actor_x)

        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value = self.value_head(critic_x)

        return policy_logits, value.squeeze(-1), (hx, cx)


class PPOAgent:
    def __init__(self, num_envs=100, num_steps=128, num_updates=2000, hidden_size=128,
                 grid_size=20, view_size=5, max_hunger=100, num_trees=1, num_predators=1,
                 results_path=None, lstm_percentage=1.0):
        self.config_string = (f"envs_{num_envs}-steps_{num_steps}-updates_{num_updates}-"
                              f"hidden_{hidden_size}-grid_{grid_size}-view_{view_size}-hunger_{max_hunger}-"
                              f"trees_{num_trees}-predators_{num_predators}-lstm_percentage_{lstm_percentage}")

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_hunger = max_hunger
        self.num_trees = num_trees
        self.num_predators = num_predators
        self.lstm_percentage = lstm_percentage

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.learning_rate = 2.5e-4
        self.eps = 1e-5

        self.envs = [GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger, num_predators=self.num_predators, num_trees=self.num_trees) for _ in range(num_envs)]
        self.input_size = self.view_size * self.view_size - 1
        if self.envs[0].observe_hunger:
            self.input_size += 1
        self.num_actions = 4
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyValueNetwork(self.input_size, self.num_actions, self.hidden_size, lstm_percentage=self.lstm_percentage).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=self.eps)

        self.num_lstm_neurons = int(self.lstm_percentage * self.hidden_size)
        if self.num_lstm_neurons > 0:
            self.hx = torch.zeros(1, self.num_envs, self.num_lstm_neurons, device=self.device)
            self.cx = torch.zeros(1, self.num_envs, self.num_lstm_neurons, device=self.device)
        else:
            self.hx = None
            self.cx = None

        self.all_rewards = []
        self.agent_positions = []
        self.apple_positions = []
        self.results_path = results_path
        self.temp_plot_initialized = False

    def collect_rollouts(self, track_positions=False):
        obs_list, actions_list, log_probs_list = [], [], []
        values_list, rewards_list, dones_list = [], [], []
        hxs_list, cxs_list = [], []

        obs = np.array([env.reset() for env in self.envs])
        obs = torch.tensor(obs, device=self.device)

        if track_positions:
            positions = []
            apple_positions = []

        for step in range(self.num_steps):
            with torch.no_grad():
                if self.num_lstm_neurons > 0:
                    policy_logits, value, (hx, cx) = self.policy(obs, self.hx, self.cx)
                else:
                    policy_logits, value, _ = self.policy(obs, None, None)
                    hx, cx = None, None

                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            obs_list.append(obs.cpu())
            actions_list.append(action.cpu())
            log_probs_list.append(log_prob.cpu())
            values_list.append(value.cpu())

            if self.num_lstm_neurons > 0:
                hxs_list.append(self.hx.squeeze(0).cpu())
                cxs_list.append(self.cx.squeeze(0).cpu())
                self.hx = hx.detach()
                self.cx = cx.detach()
            else:
                hxs_list.append(None)
                cxs_list.append(None)

            obs_np = []
            rewards = []
            dones = []
            for i, env in enumerate(self.envs):
                ob, reward, done = env.step(action[i].item())
                if track_positions and i == 0:
                    positions.append(tuple(env.agent_pos))
                    apple_positions.append(env.apple_positions if env.apple_positions else None)

                if done:
                    ob = env.reset()
                    if self.num_lstm_neurons > 0:
                        self.hx[:, i, :] = torch.zeros_like(self.hx[:, i, :])
                        self.cx[:, i, :] = torch.zeros_like(self.cx[:, i, :])

                obs_np.append(ob)
                rewards.append(reward)
                dones.append(done)

            obs = torch.tensor(np.array(obs_np), device=self.device)
            rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
            dones_list.append(torch.tensor(dones, dtype=torch.float32))

        with torch.no_grad():
            if self.num_lstm_neurons > 0:
                _, next_value, _ = self.policy(obs, self.hx, self.cx)
            else:
                _, next_value, _ = self.policy(obs, None, None)
        next_value = next_value.cpu()

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
                      returns_batch, advantages_batch, hxs_batch, cxs_batch, old_values_batch):
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

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
            old_values_mb = old_values_batch[mb_indices].to(self.device)

            if self.num_lstm_neurons > 0:
                hxs_mb = hxs_batch[mb_indices].unsqueeze(0).to(self.device)
                cxs_mb = cxs_batch[mb_indices].unsqueeze(0).to(self.device)
                policy_logits, value, _ = self.policy(obs_mb, hxs_mb, cxs_mb)
            else:
                policy_logits, value, _ = self.policy(obs_mb, None, None)

            dist = torch.distributions.Categorical(logits=policy_logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions_mb)

            ratio = torch.exp(new_log_probs - old_log_probs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            value_pred = value.squeeze(-1)
            value_pred_clipped = old_values_mb + (value_pred - old_values_mb).clamp(-self.clip_epsilon, self.clip_epsilon)
            value_loss_unclipped = (value_pred - returns_mb).pow(2)
            value_loss_clipped = (value_pred_clipped - returns_mb).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss.item(), actor_loss, value_loss, entropy

    def train(self):
        for update in range(self.num_updates):
            track_positions = (update == self.num_updates - 1)
            (obs_list, actions_list, log_probs_list, values_list,
             rewards_list, dones_list, next_value, hxs_list, cxs_list) = self.collect_rollouts(track_positions=track_positions)

            advantages, returns = self.compute_gae(rewards_list, values_list, dones_list, next_value)

            obs_batch = torch.stack(obs_list).view(-1, self.input_size)
            actions_batch = torch.stack(actions_list).view(-1)
            log_probs_old_batch = torch.stack(log_probs_list).view(-1)
            returns_batch = returns.view(-1)
            advantages_batch = advantages.view(-1)
            old_values_batch = torch.stack(values_list).view(-1).detach()

            if self.num_lstm_neurons > 0:
                hxs_batch = torch.stack(hxs_list).view(-1, self.num_lstm_neurons)
                cxs_batch = torch.stack(cxs_list).view(-1, self.num_lstm_neurons)
            else:
                hxs_batch = None
                cxs_batch = None

            loss, actor_loss, value_loss, entropy = self.update_policy(
                obs_batch, actions_batch, log_probs_old_batch,
                returns_batch, advantages_batch, hxs_batch, cxs_batch, old_values_batch)

            avg_reward = torch.stack(rewards_list).sum(0).mean().item()
            self.all_rewards.append(avg_reward)

            if update % 10 == 0:
                print(f'Update {update}, Loss: {loss:.4f}, Avg Reward: {avg_reward:.2f}')
                print(f"Actor Loss: {actor_loss.item()}, Value Loss: {value_loss.item()}, Entropy: {entropy.item()}")

            if update % 100 == 0 and update > 0:
                self.plot_rewards_temp()
                torch.save(self.policy.state_dict(), 'trained_policy_temp.pth')

        print("Training completed!")
        if self.temp_plot_initialized:
            plt.close(self.fig)
            plt.close('all')
            self.temp_plot_initialized = False

        if self.results_path:
            torch.save(self.policy.state_dict(), f'{self.results_path}/{self.config_string}.pth')
        else:
            self.plot_rewards()
            torch.save(self.policy.state_dict(), self.config_string + '.pth')
            self.test_trained_model(self.config_string + '.pth')

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.all_rewards)
        plt.xlabel('Update')
        plt.ylabel('Average Reward')
        plt.grid()
        plt.show()

    def plot_rewards_temp(self):
        if not self.temp_plot_initialized:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.line, = self.ax.plot([], [], label="Average Reward")
            self.ax.set_xlabel('Update')
            self.ax.set_ylabel('Average Reward')
            self.ax.grid()
            self.ax.legend()
            self.temp_plot_initialized = True
        
        self.line.set_xdata(range(len(self.all_rewards)))
        self.line.set_ydata(self.all_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def plot_agent_positions(self):
        if not self.agent_positions:
            print("No agent positions to plot.")
            return

        positions = np.array(self.agent_positions)
        x_positions = positions[:, 1]
        y_positions = positions[:, 0]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_positions, y_positions, c=range(len(x_positions)), cmap='viridis', marker='o', label='Agent')

        apple_positions_filtered = [pos for pos in self.apple_positions if pos is not None]
        if apple_positions_filtered:
            flat_apple_positions = []
            for item in apple_positions_filtered:
                if isinstance(item, list):
                    flat_apple_positions.extend(item)
                elif isinstance(item, tuple):
                    flat_apple_positions.append(item)
            apple_positions_array = np.array(flat_apple_positions)
            apple_x_positions = apple_positions_array[:, 1]
            apple_y_positions = apple_positions_array[:, 0]
            apple_timesteps = [i for i, pos in enumerate(self.apple_positions) if pos is not None]
            plt.scatter(apple_x_positions, apple_y_positions, c=apple_timesteps, cmap='cool', marker='x', label='Apple')

        apple_tree_positions = np.array([pos for tree in self.envs[0].apple_trees for pos in tree])
        if len(apple_tree_positions) > 0:
            plt.scatter(apple_tree_positions[:, 1], apple_tree_positions[:, 0], color='brown', marker='s', label='Apple Tree')

        plt.colorbar(label='Timestep')
        plt.title('Agent, Apple, and Apple Tree Movement in the Last Rollout')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim(0, self.envs[0].grid_size)
        plt.ylim(0, self.envs[0].grid_size)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_trained_model(self, model_path="trained_policy.pth"):
        test_env = GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger, num_predators=self.num_predators, num_trees=self.num_trees)
        obs = test_env.reset()
        obs = torch.tensor(obs, device=self.device).unsqueeze(0)

        if self.num_lstm_neurons > 0:
            hx = torch.zeros(1, 1, self.num_lstm_neurons, device=self.device)
            cx = torch.zeros(1, 1, self.num_lstm_neurons, device=self.device)
        else:
            hx, cx = None, None

        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()

        done = False
        step = 0

        def update_plot():
            ax.clear()
            grid = test_env.grid.copy()
            grid[test_env.agent_pos[0], test_env.agent_pos[1]] = test_env.AGENT

            cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue', 'brown'])
            bounds = [0, 1, 2, 3, 4, 5, 6]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            ax.imshow(grid, cmap=cmap, norm=norm)

            legend_elements = [
                mpatches.Patch(color='white', label='Empty'),
                mpatches.Patch(color='black', label='Wall'),
                mpatches.Patch(color='green', label='Apple'),
                mpatches.Patch(color='red', label='Predator'),
                mpatches.Patch(color='blue', label='Agent'),
                mpatches.Patch(color='brown', label='Apple Tree'),
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Step: {step}')

            plt.draw()
            plt.pause(0.001)

        update_plot()

        def on_key(event):
            nonlocal obs, hx, cx, done, step
            if event.key == 'right' and not done:
                with torch.no_grad():
                    if self.num_lstm_neurons > 0:
                        policy_logits, _, (hx, cx) = self.policy(obs, hx, cx)
                    else:
                        policy_logits, _, _ = self.policy(obs, None, None)

                    dist = torch.distributions.Categorical(logits=policy_logits)
                    action = dist.sample()
                ob, reward, done = test_env.step(action.item())
                if done:
                    if self.num_lstm_neurons > 0:
                        hx = torch.zeros(1, 1, self.num_lstm_neurons, device=self.device)
                        cx = torch.zeros(1, 1, self.num_lstm_neurons, device=self.device)
                    print(f"Episode ended with reward: {reward}")
                else:
                    if self.num_lstm_neurons > 0:
                        hx = hx.detach()
                        cx = cx.detach()
                obs = torch.tensor(ob, device=self.device).unsqueeze(0)
                step += 1
                update_plot()
            elif event.key == 'q':
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"

    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    # Example usage:
    # lstm_percentage = 0.5 means half the second hidden layer is LSTM and half is feed-forward.
    agent = PPOAgent(num_envs=100, num_steps=256, num_updates=2000, hidden_size=256,
                     grid_size=100, view_size=7, max_hunger=100, num_trees=1, num_predators=1,
                     results_path=None, lstm_percentage=0.5)
    agent.train()
