import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Constants
TILES_HORIZONTAL = 20
TILES_VERTICAL = 20

# --------------------------------------------------------
#                   Experience Buffer
# --------------------------------------------------------
class ExperienceBuffer:
    def __init__(self):
        self.buffer = deque()

    def add(self, experience):
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def get(self):
        return list(self.buffer)

# --------------------------------------------------------
#                   Neural Network
# --------------------------------------------------------
class AgentNN(nn.Module):
    def __init__(self, input_size=27, hidden_size1=128, output_size=4):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc_policy = nn.Linear(hidden_size1, output_size)  # For policy output
        self.fc_value = nn.Linear(hidden_size1, 1)             # For value estimate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Policy branch
        policy_logits = self.fc_policy(x)
        # Value branch
        value = self.fc_value(x)
        return policy_logits, value

# --------------------------------------------------------
#                   Agent Class
# --------------------------------------------------------
class Agent:
    def __init__(self, model):
        self.x, self.y = None, None
        self.load_agent()
        self.hunger = 10
        self.model = model  # Shared model

    def load_agent(self):
        # Initialize agent's position
        self.x, self.y = TILES_HORIZONTAL // 2, TILES_VERTICAL // 2

    def get_observed_state(self, tiles, monsters, apples):
        observed_tiles = []
        agent_x, agent_y = self.x, self.y
        grid_size = 5
        offset = grid_size // 2

        for dy in range(-offset, offset + 1):
            row = []
            for dx in range(-offset, offset + 1):
                x = agent_x + dx
                y = agent_y + dy

                if 0 <= x < TILES_HORIZONTAL and 0 <= y < TILES_VERTICAL:
                    tile = next((t for t in tiles.inner if t.x == x and t.y == y), None)
                    monster = next((m for m in monsters.inner if m.x == x and m.y == y), None)
                    apple = next((a for a in apples.inner if a.x == x and a.y == y), None)
                    cell = {
                        'x': x,
                        'y': y,
                        'tile': tile,
                        'monster': monster,
                        'apple': apple
                    }
                else:
                    # Position is out of bounds
                    cell = {
                        'x': x,
                        'y': y,
                        'tile': None,
                        'monster': None,
                        'apple': None,
                        'out_of_bounds': True
                    }
                row.append(cell)
            observed_tiles.append(row)
        return observed_tiles

    def process_observed_state(self, observed_state):
        input_vector = []
        for row in observed_state:
            for cell in row:
                if 'out_of_bounds' in cell and cell['out_of_bounds']:
                    input_vector.append(-2)  # Represent out-of-bounds
                elif cell['monster'] is not None:
                    input_vector.append(-1)
                elif cell['apple'] is not None:
                    input_vector.append(1)
                else:
                    input_vector.append(0)
        input_vector.append(self.x / TILES_HORIZONTAL)  # Normalize position
        input_vector.append(self.y / TILES_VERTICAL)
        return input_vector  # Length should be 27 (25 + 2)

    def move(self, action):
        if action == 'up' and self.y > 0:
            self.y -= 1
        elif action == 'down' and self.y < TILES_VERTICAL - 1:
            self.y += 1
        elif action == 'left' and self.x > 0:
            self.x -= 1
        elif action == 'right' and self.x < TILES_HORIZONTAL - 1:
            self.x += 1
        else:
            # Invalid move, agent stays in place
            pass

    def get_closest_monster_distance(self, observed_state):
        distances = []
        for row in observed_state:
            for cell in row:
                if cell['monster'] is not None:
                    # Manhattan distance from agent's position
                    distance = abs(cell['x'] - self.x) + abs(cell['y'] - self.y)
                    distances.append(distance)
        if distances:
            return min(distances)
        else:
            return None

    def get_closest_apple_distance(self, observed_state):
        distances = []
        for row in observed_state:
            for cell in row:
                if cell['apple'] is not None:
                    # Manhattan distance from agent's position
                    distance = abs(cell['x'] - self.x) + abs(cell['y'] - self.y)
                    distances.append(distance)
        if distances:
            return min(distances)
        else:
            return None

    def check_if_eat_apple(self, apples):
        # Check if the agent is on the same position as any apple
        for apple in apples.inner:
            if apple.x == self.x and apple.y == self.y:
                apples.inner.remove(apple)  # Remove the apple from the game
                return True
        return False

    def is_dead(self, monsters):
        # Check if a monster is on the same tile
        for monster in monsters.inner:
            if monster.x == self.x and monster.y == self.y:
                return True
        return False

# --------------------------------------------------------
#                   Apple Classes
# --------------------------------------------------------
class Apple:
    def __init__(self, id, x, y, apple_kind):
        self.id = id
        self.x, self.y = int(x), int(y)
        if apple_kind == "a":
            self.apple_image = "apple.png"  # Placeholder
        else:
            s = "Sorry, I don't recognize that: {}".format(apple_kind)
            raise ValueError(s)

class Apples:
    def __init__(self):
        self.inner = []
        self.load_apples()

    def load_apples(self):
        self.inner = []
        for i in range(5):  # Initialize with 5 apples at random positions
            x = random.randint(0, TILES_HORIZONTAL - 1)
            y = random.randint(0, TILES_VERTICAL - 1)
            apple = Apple(i, x, y, "a")
            self.inner.append(apple)

    def spawn_apple(self):
        # Spawn an apple at a random unoccupied position
        x = random.randint(0, TILES_HORIZONTAL - 1)
        y = random.randint(0, TILES_VERTICAL - 1)
        apple_id = max([apple.id for apple in self.inner], default=0) + 1
        apple = Apple(apple_id, x, y, "a")
        self.inner.append(apple)

# --------------------------------------------------------
#                   Monster Classes
# --------------------------------------------------------
class Monster:
    def __init__(self, id, x, y, monster_kind):
        self.id = id
        self.x, self.y = int(x), int(y)
        if monster_kind == "m":
            self.monster_image = "monster.png"  # Placeholder
        else:
            s = "Sorry, I don't recognize that: {}".format(monster_kind)
            raise ValueError(s)

class Monsters:
    def __init__(self):
        self.inner = []
        self.load_monsters()

    def load_monsters(self):
        self.inner = []
        for i in range(3):  # Initialize with 3 monsters at random positions
            x = random.randint(0, TILES_HORIZONTAL - 1)
            y = random.randint(0, TILES_VERTICAL - 1)
            monster = Monster(i, x, y, "m")
            self.inner.append(monster)

    def update(self, agent):
        for monster in self.inner:
            distance_x = abs(monster.x - agent.x)
            distance_y = abs(monster.y - agent.y)
            if distance_x <= 7 and distance_y <= 7:
                # Monster is within s^{obs}
                p = random.random()
                if p < 0.7:
                    # 70% chance to move towards the agent
                    self.move_towards_agent(monster, agent)
                else:
                    # 30% chance to move randomly in other directions
                    self.move_random_except(monster, agent)
            else:
                # Monster is not within s^{obs}
                # Move randomly in any direction
                self.move_random(monster)

    def move_towards_agent(self, monster, agent):
        possible_moves = []
        min_distance = None

        directions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

        for (dx, dy) in directions.values():
            new_x = monster.x + dx
            new_y = monster.y + dy

            # Check if move is within bounds
            if 0 <= new_x < TILES_HORIZONTAL and 0 <= new_y < TILES_VERTICAL:
                # Compute Manhattan distance to agent from new position
                distance = abs(new_x - agent.x) + abs(new_y - agent.y)

                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    possible_moves = [(dx, dy)]
                elif distance == min_distance:
                    possible_moves.append((dx, dy))

        if possible_moves:
            # Choose one of the moves that reduce the distance the most
            move = random.choice(possible_moves)
            monster.x += move[0]
            monster.y += move[1]

    def move_random_except(self, monster, agent):
        # First, find the move(s) that reduce the distance the most
        best_moves = []
        min_distance = None

        directions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        for (dx, dy) in directions.values():
            new_x = monster.x + dx
            new_y = monster.y + dy
            if 0 <= new_x < TILES_HORIZONTAL and 0 <= new_y < TILES_VERTICAL:
                distance = abs(new_x - agent.x) + abs(new_y - agent.y)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_moves = [(dx, dy)]
                elif distance == min_distance:
                    best_moves.append((dx, dy))

        # Exclude the best moves
        other_moves = []
        for (dx, dy) in directions.values():
            if (dx, dy) not in best_moves:
                new_x = monster.x + dx
                new_y = monster.y + dy
                if 0 <= new_x < TILES_HORIZONTAL and 0 <= new_y < TILES_VERTICAL:
                    other_moves.append((dx, dy))

        if other_moves:
            # Move randomly among the other moves
            move = random.choice(other_moves)
            monster.x += move[0]
            monster.y += move[1]
        else:
            # If no other moves are possible, stay in place
            pass

    def move_random(self, monster):
        directions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        possible_moves = []
        for (dx, dy) in directions.values():
            new_x = monster.x + dx
            new_y = monster.y + dy
            if 0 <= new_x < TILES_HORIZONTAL and 0 <= new_y < TILES_VERTICAL:
                possible_moves.append((dx, dy))
        if possible_moves:
            move = random.choice(possible_moves)
            monster.x += move[0]
            monster.y += move[1]
        else:
            # No possible moves, stay in place
            pass

# --------------------------------------------------------
#                   Tile Classes
# --------------------------------------------------------
class Tile:
    def __init__(self, id, x, y, kind_of_tile):
        self.id = id
        self.x = int(x)
        self.y = int(y)
        self.kind_of_tile = kind_of_tile

class Tiles:
    def __init__(self):
        self.inner = []
        self.load_tiles()

    def load_tiles(self):
        self.inner = []
        id = 0
        # For simplicity, create a grid of 'dirt' tiles
        for y in range(TILES_VERTICAL):
            for x in range(TILES_HORIZONTAL):
                new_tile = Tile(id, x, y, 'd')
                self.inner.append(new_tile)
                id += 1

# --------------------------------------------------------
#                   Environment Class
# --------------------------------------------------------
class Environment:
    def __init__(self, model):
        self.tiles = Tiles()
        self.monsters = Monsters()
        self.apples = Apples()
        self.agent = Agent(model)
        self.done = False
        self.total_reward = 0  # Initialize total reward

    def reset(self):
        self.tiles = Tiles()
        self.monsters = Monsters()
        self.apples = Apples()
        self.agent.load_agent()
        self.agent.hunger = 10
        self.done = False
        self.total_reward = 0  # Reset total reward
        return self.get_state()

    def step(self, action_str):
        # Agent performs action
        self.agent.move(action_str)
        self.agent.hunger -= 0.01
        self.agent.hunger = max(self.agent.hunger, 0)

        # Update monsters
        self.monsters.update(self.agent)

        # Check if agent eats an apple
        if self.agent.check_if_eat_apple(self.apples):
            self.agent.hunger = 10  # Reset hunger when apple is eaten

        # Spawn apples periodically
        if random.random() < 0.05:  # 5% chance to spawn an apple each step
            self.apples.spawn_apple()

        # Get reward and check if done
        observed_state = self.agent.get_observed_state(self.tiles, self.monsters, self.apples)
        reward = self.compute_reward(observed_state)
        self.total_reward += reward  # Accumulate reward
        self.done = self.check_if_done()

        # Get next state
        next_state = self.get_state()

        return next_state, reward, self.done

    def get_state(self):
        observed_state = self.agent.get_observed_state(self.tiles, self.monsters, self.apples)
        input_vector = self.agent.process_observed_state(observed_state)
        state = torch.tensor(input_vector, dtype=torch.float32)
        return state

    def compute_reward(self, observed_state):
        reward = 0

        # Reward for eating an apple
        if self.agent.check_if_eat_apple(self.apples):
            reward += 100

        # Penalty for being eaten by a monster
        if self.check_if_done():
            reward -= 100

        # Reward inversely proportional to distance to the closest apple
        closest_apple_distance = self.agent.get_closest_apple_distance(observed_state)
        if closest_apple_distance is not None and closest_apple_distance > 0:
            apple_reward = (1 / closest_apple_distance) * 10
            reward += apple_reward

        # Penalty proportional to proximity to the closest monster
        #closest_monster_distance = self.agent.get_closest_monster_distance(observed_state)
        #if closest_monster_distance is not None and closest_monster_distance > 0:
        #    monster_penalty = (1 / closest_monster_distance) * 10
        #    reward -= monster_penalty

        # Small penalty for each time step to encourage efficiency
        reward -= 0.1

        # Penalty for hunger
        reward -= (10 - self.agent.hunger) * 0.1

        return reward

    def check_if_done(self):
        if self.agent.hunger <= 0:
            return True
        if self.agent.is_dead(self.monsters):
            return True
        return False

# --------------------------------------------------------
#                   Game Class
# --------------------------------------------------------
class Game:
    def __init__(self, num_envs=10):
        self.num_envs = num_envs
        self.environments = []
        self.agent_model = AgentNN()
        self.agent_model.train()
        self.optimizer = optim.Adam(self.agent_model.parameters(), lr=1e-4)
        # PPO Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 4
        self.clip_param = 0.2
        self.mini_batch_size = 1024

        for _ in range(self.num_envs):
            env = Environment(self.agent_model)
            self.environments.append(env)

        self.keep_looping = True
        self.experience_buffer = ExperienceBuffer()
        self.episode_rewards = []  # List to store average rewards per iteration
        self.loss_values = []      # List to store loss values

    def compute_ppo_loss(self, states, actions, old_log_probs, returns, advantages, clip_param=0.2):
        policy_logits, values = self.agent_model(states)
        values = values.squeeze(-1)  # values shape: [batch_size]

        # Calculate the new log probabilities
        action_probs = F.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)

        # Calculate the entropy of the action distribution
        entropy = dist.entropy().mean()

        # Calculate new log probabilities
        new_log_probs = dist.log_prob(actions)

        # Calculate ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # Adjust coefficients as needed

        return loss

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return returns, advantages

    def update_policy(self):
        # Convert experiences to tensors
        experiences = self.experience_buffer.get()
        states = torch.stack([e['state'] for e in experiences])
        actions = torch.tensor([e['action'] for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e['reward'] for e in experiences], dtype=torch.float32)
        log_probs = torch.stack([e['log_prob'] for e in experiences])
        values = torch.tensor([e['value'] for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e['done'] for e in experiences], dtype=torch.float32)

        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Perform PPO updates
        for _ in range(self.ppo_epochs):
            loss = self.compute_ppo_loss(states, actions, log_probs, returns, advantages, self.clip_param)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Record the loss value
        self.loss_values.append(loss.item())

    def main(self):
        i = 0
        MAX_TRAINING_STEPS = 10000
        while self.keep_looping:
            i += 1
            # Collect experiences from all environments
            episode_rewards = []  # Collect rewards from all environments for this iteration
            for env in self.environments:
                state = env.get_state()
                done = False
                #while not done:
                for i in range(10000):
                    # Get action and value from the agent
                    policy_logits, value = self.agent_model(state.unsqueeze(0))
                    action_probs = F.softmax(policy_logits, dim=1)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    # Map action index to action string
                    action_str = ['up', 'down', 'left', 'right'][action.item()]

                    # Environment step
                    next_state, reward, done = env.step(action_str)

                    # Store experience
                    experience = {
                        'state': state,
                        'action': action.item(),
                        'reward': reward,
                        'log_prob': log_prob.detach(),
                        'value': value.detach().squeeze(),
                        'done': done
                    }
                    self.experience_buffer.add(experience)

                    state = next_state

                    # Update policy if enough experiences collected
                    if len(self.experience_buffer.buffer) >= self.mini_batch_size:
                        self.update_policy()
                        self.experience_buffer.clear()

                    if done:
                        # Record the total reward for this episode
                        episode_rewards.append(env.total_reward)
                        env.reset()
                        break  # Exit the while loop for this environment

            # Compute average reward for this iteration
            if episode_rewards:
                average_reward = sum(episode_rewards) / len(episode_rewards)
                self.episode_rewards.append(average_reward)
                print(f"Iteration {i}, Average Reward: {average_reward:.2f}")

            # Termination condition for the main loop
            if i >= MAX_TRAINING_STEPS:
                self.keep_looping = False

        # After training is complete, plot the average episode rewards
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.episode_rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Average Total Reward')
        plt.title('Agent Performance Over Time')
        plt.show()

        # Plot the loss values
        plt.figure()
        plt.plot(self.loss_values)
        plt.xlabel('Policy Update Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.show()

if __name__ == "__main__":
    print("Starting the game with PPO training...")
    game = Game(num_envs=10)
    game.main()
