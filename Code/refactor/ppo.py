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
from utils import *
from env import *

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
    # Create a function to generate new environments
    def make_env(self):
            return GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger,
                                num_predators=self.num_predators, num_trees=self.num_trees)
    
    def __init__(self, num_envs=100, num_steps=128, num_updates=2000, hidden_size = 128, grid_size=20, view_size=5, max_hunger=100, num_trees=1, num_predators=1, results_path=None):
        self.config_string = f"envs_{num_envs}-steps_{num_steps}-updates_{num_updates}-hidden_{hidden_size}-grid_{grid_size}-view_{view_size}-hunger_{max_hunger}-trees_{num_trees}-predators_{num_predators}"

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_hunger = max_hunger
        self.num_trees = num_trees
        self.num_predators = num_predators

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.learning_rate = 2.5e-4
        self.eps = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.envs = ParallelEnv(self.num_envs, self.make_env)
        self.input_size = self.view_size * self.view_size - 1 #The square of view size around the agent, minus its position
        self.num_actions = 4
        self.hidden_size = hidden_size  # Hidden size for LSTM

        self.policy = PolicyValueNetwork(self.input_size, self.num_actions, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=self.eps)

        self.all_rewards = []
        self.agent_positions = []  # To track positions of the first agent
        self.apple_positions = []  # To track positions of the apple

        self.results_path = results_path

        # Initialize LSTM hidden states (num_layers=1)
        self.hx = torch.zeros(1, self.num_envs, self.hidden_size, device=self.device)
        self.cx = torch.zeros(1, self.num_envs, self.hidden_size, device=self.device)

    def collect_rollouts(self, track_positions=False):
        obs_list, actions_list, log_probs_list = [], [], []
        values_list, rewards_list, dones_list = [], [], []
        hxs_list, cxs_list = [], []

        obs = self.envs.reset()
        # obs = torch.tensor(obs, device=self.device)

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
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)
            hxs_list.append(self.hx.squeeze(0))
            cxs_list.append(self.cx.squeeze(0))

            # obs_list.append(obs.cpu())
            # actions_list.append(action.cpu())
            # log_probs_list.append(log_prob.cpu())
            # values_list.append(value.cpu())


            # Update hidden states
            self.hx = hx.detach()
            self.cx = cx.detach()

            actions_np = action #.cpu().numpy()
            obs, rewards, dones, infos = self.envs.step(actions_np)

            # Convert observations and rewards to tensors
            # obs = torch.tensor(obs_np, device=self.device)
            # obs = torch.from_numpy(obs_np).to(self.device)

            # rewards = torch.tensor(rewards_np, dtype=torch.float32)
            # dones = torch.tensor(dones_np, dtype=torch.float32)

            # For environments that are done, reset hidden states
            for i in range(self.num_envs):
                if dones[i]:
                    self.hx[:, i, :] = torch.zeros_like(self.hx[:, i, :])
                    self.cx[:, i, :] = torch.zeros_like(self.cx[:, i, :])

            rewards_list.append(rewards)
            dones_list.append(dones)

            # Track positions of the first agent and the apple
            if track_positions:
                # Assuming the first environment corresponds to the first agent
                env_info = infos[0]
                positions.append(env_info.get('agent_pos', None))
                apple_positions.append(env_info.get('apple_positions', None))

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
        rewards = torch.stack(rewards_list).to(self.device)
        values = torch.stack(values_list).to(self.device)
        dones = torch.stack(dones_list).to(self.device)
        next_value = next_value.to(self.device)  # Ensure next_value is on the same device

        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
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
            old_values_mb = old_values_batch[mb_indices].to(self.device)

            # Forward pass with LSTM hidden states
            policy_logits, value, _ = self.policy(obs_mb, hxs_mb, cxs_mb)
            dist = torch.distributions.Categorical(logits=policy_logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions_mb)

            ratio = torch.exp(new_log_probs - old_log_probs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            # Clipped value function
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
            old_values_batch = torch.stack(values_list).view(-1)  # Flatten old values

            # Ensure old_values are detached from the computation graph
            old_values_batch = old_values_batch.detach()

            loss, actor_loss, value_loss, entropy = self.update_policy(
                obs_batch, actions_batch, log_probs_old_batch,
                returns_batch, advantages_batch, hxs_batch, cxs_batch, old_values_batch)

            # Tracking average reward
            avg_reward = torch.stack(rewards_list).sum(0).mean().item()
            self.all_rewards.append(avg_reward)

            if update % 10 == 0:
                print(f'Update {update}, Loss: {loss:.4f}, Avg Reward: {avg_reward:.2f}')
                print(f"Actor Loss: {actor_loss.item()}, Value Loss: {value_loss.item()}, Entropy: {entropy.item()}")

        print("Training completed!")
        self.envs.close()  # Close the parallel environments
        if self.results_path:
            torch.save(self.policy.state_dict(), f'{self.results_path}/{self.config_string}.pth')
        else:
            self.plot_rewards()
            # self.plot_agent_positions()  # Plot the agent's positions (needs adjustment)
            # Save the trained model
            torch.save(self.policy.state_dict(), 'trained_policy.pth')
            # Run the test environment
            self.test_trained_model()


    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.all_rewards)
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
            # Flatten the list if it contains nested sequences
            flat_apple_positions = []
            for item in apple_positions_filtered:
                if isinstance(item, list):  # If it's a nested list (e.g., [[(1, 2), (3, 4)], ...])
                    flat_apple_positions.extend(item)
                elif isinstance(item, tuple):  # If it's a tuple
                    flat_apple_positions.append(item)
                # Skip invalid types like `None` (already filtered)
            # apple_positions_array = np.array(apple_positions_filtered)
            apple_positions_array = np.array(flat_apple_positions)
            apple_x_positions = apple_positions_array[:, 1]
            apple_y_positions = apple_positions_array[:, 0]
            apple_timesteps = [i for i, pos in enumerate(self.apple_positions) if pos is not None]
            plt.scatter(apple_x_positions, apple_y_positions, c=apple_timesteps, cmap='cool', marker='x', label='Apple')

        # Plot the apple tree positions
        # apple_tree_positions = np.array(self.envs[0].apple_tree_positions)
        apple_tree_positions = np.array([pos for tree in self.envs[0].apple_trees for pos in tree])
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
        test_env = GridWorldEnv(grid_size=self.grid_size, view_size=self.view_size, max_hunger=self.max_hunger, num_predators=self.num_predators, num_trees=self.num_trees)
        obs = test_env.reset()
        # print(type(obs))
        obs = obs.unsqueeze(0)
        hx = torch.zeros(1, 1, self.hidden_size, device=self.device)
        cx = torch.zeros(1, 1, self.hidden_size, device=self.device)

        # Load the trained model
        self.policy.load_state_dict(torch.load('trained_policy.pth', weights_only=False))
        self.policy.eval()

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()

        done = False
        step = 0

        def update_plot():
            ax.clear()
            grid = test_env.grid.copy()  # Use test_env.grid instead of test_env.terrain_grid
            grid[test_env.agent_pos[0], test_env.agent_pos[1]] = 4  # Represent agent with AGENT constant

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

        def on_key(event):
            nonlocal obs, hx, cx, done, step
            if event.key == 'right' and not done:
                with torch.no_grad():
                    policy_logits, _, (hx, cx) = self.policy(obs, hx, cx)
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    action = dist.sample()
                ob, reward, done, info = test_env.step(action.item())
                # Reset hidden states if done
                if done:
                    hx = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    cx = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    print(f"Episode ended with reward: {reward}")
                else:
                    hx = hx.detach()
                    cx = cx.detach()
                obs = ob.unsqueeze(0)
                step += 1
                update_plot()
            elif event.key == 'q':
                plt.close()
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.ioff()
        plt.show()
