"""
Draws a window filled with tiles AND MONSTERS!
"""
import pygame
import constants as con
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TITLE = "beasties"
TILES_HORIZONTAL = 20
TILES_VERTICAL = 20
TILESIZE = 32
WINDOW_WIDTH = TILESIZE * TILES_HORIZONTAL
WINDOW_HEIGHT = TILESIZE * TILES_VERTICAL

# In a separate file or within your script



class AgentNN(nn.Module):
    def __init__(self, input_size=25, hidden_size1=128, lstm_hidden_size=64, hidden_size2=64, output_size=4):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lstm = nn.LSTM(input_size=hidden_size1, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)    # Remove sequence dimension
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden

# --------------------------------------------------------
#                   class Agent
# --------------------------------------------------------
class Agent:
    def __init__(self, surface):
        self.surface = surface
        self.x, self.y = None, None
        self.agent_image = ""
        self.load_agent()
        self.hunger = 10
        # ---------------------
        image_path = os.path.join("data", "images")
        self.image = pygame.image.load(os.path.join(image_path, self.agent_image)).convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILESIZE, TILESIZE))
        # ---------------------
        self.model = AgentNN()
        self.hidden = None  # LSTM hidden state

        # Set the model to evaluation mode
        self.model.eval()

    def load_agent(self):
        filepath = os.path.join("data", "agent_map.txt")
        with open(filepath, "r") as f:
            mylines = f.readlines()
            mylines = [line.strip() for line in mylines if len(line.strip()) > 0]
        agent_found = False
        for count_i, line in enumerate(mylines):
            for count_j, elem in enumerate(line):
                if elem == "a":  # Assuming 'a' represents the agent in agent_map.txt
                    self.x, self.y = count_j, count_i
                    agent_found = True
                    break  # Exit the inner loop
            if agent_found:
                break  # Exit the outer loop
        if not agent_found:
            raise ValueError("Agent not found in agent_map.txt")
        # Set the agent image after confirming the agent is found
        self.agent_image = con.AGENT  # Ensure con.AGENT is defined with the agent's image filename

    def draw(self, surface):
        if self.x is None or self.y is None:
            raise ValueError("Agent position is not set.")
        myrect = pygame.Rect(self.x * TILESIZE, self.y * TILESIZE, TILESIZE, TILESIZE)
        self.surface.blit(self.image, myrect)

    def debug_print(self):
        s = "Agent position - x: {}, y: {}".format(self.x, self.y)
        print(s)

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
        return input_vector  # Length should be 25
    
    def agent_action(self, observed_state):
        input_vector = self.process_observed_state(observed_state)
        
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 25]

        # Initialize hidden state if it's None
        if self.hidden is None:
            self.hidden = (torch.zeros(1, 1, self.model.lstm.hidden_size),
                           torch.zeros(1, 1, self.model.lstm.hidden_size))

        with torch.no_grad():
            output, self.hidden = self.model(input_tensor, self.hidden)
            probabilities = F.softmax(output, dim=1).numpy()[0]  # Convert to numpy array

        # Decide on an action based on the probabilities
        action = np.random.choice(['up', 'down', 'left', 'right'], p=probabilities)

        # Move the agent based on the action
        self.move(action)

        self.hunger -= 0.1
        self.hunger = max(self.hunger, 0)  # Ensure hunger doesn't go below zero
        print(f"Agent hunger level: {self.hunger:.2f}")

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
        
    def check_if_eat_apple(self, apples):
        # Check if the agent is on the same position as any apple
        for apple in apples.inner:
            if apple.x == self.x and apple.y == self.y:
                apples.inner.remove(apple)  # Remove the apple from the game
                return True
        return False



# --------------------------------------------------------
#                   class Apple
# --------------------------------------------------------
class Apple:
    def __init__(self, id, x, y, apple_kind):
        self.id = id
        self.x, self.y = int(x), int(y)
        self.myinc = .05
        self.apple_image = ""
        if apple_kind == "a":
            self.apple_image = con.APPLE
        else:
            s = "Sorry, I don't recognize that: {}".format(apple_kind)
            raise ValueError(s)
        # ---------------------
        image_path = os.path.join("data", "images")
        self.image = pygame.image.load(os.path.join(image_path, self.apple_image)).convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILESIZE, TILESIZE))
        # ---------------------

    def debug_print(self):
        s = "id: {}, x: {}, y: {}".format(self.id, self.x, self.y)
        print(s)


# --------------------------------------------------------
#                   class Apples
# --------------------------------------------------------
class Apples:
    def __init__(self, surface):
        self.surface = surface
        self.inner = []
        self.current_apple = None
        # ------------------------------------
        id = 0
        filepath = os.path.join("data", "apple_map.txt")
        with open(filepath, "r") as f:
            mylines = f.readlines()
            mylines = [i.strip() for i in mylines if len(i.strip()) > 0]
        for count_i, line in enumerate(mylines):
            for count_j, elem in enumerate(line):
                if elem == "a":
                    new_apple = Apple(id, count_j, count_i, elem)
                    self.inner.append(new_apple)
                    id += 1

    def draw(self, surface):
        if len(self.inner) == 0:
            raise ValueError("Doh! There are no tiles to display. 😕")
        for elem in self.inner:
            myrect = pygame.Rect(elem.x * TILESIZE, elem.y * TILESIZE, TILESIZE, TILESIZE)
            self.surface.blit(elem.image, myrect)

    def debug_print(self):
        for elem in self.inner:
            elem.debug_print()

# --------------------------------------------------------
#                   class Monster
# --------------------------------------------------------
class Monster:
    def __init__(self, id, x, y, monster_kind):
        self.id = id
        self.x, self.y = int(x), int(y)
        self.myinc = .05
        self.monster_image = ""
        if monster_kind == "m":
            self.monster_image = con.DOG
        else:
            s = "Sorry, I don't recognize that: {}".format(monster_kind)
            raise ValueError(s)
        # ---------------------
        image_path = os.path.join("data", "images")
        self.image = pygame.image.load(os.path.join(image_path, self.monster_image)).convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILESIZE, TILESIZE))
        # ---------------------

    def debug_print(self):
        s = "id: {}, x: {}, y: {}".format(self.id, self.x, self.y)
        print(s)

    

# --------------------------------------------------------
#                   class Monsters
# --------------------------------------------------------
class Monsters:
    def __init__(self, surface):
        self.surface = surface
        self.inner = []
        self.current_monster = None
        # ------------------------------------
        id = 0
        filepath = os.path.join("data", "monster_map.txt")
        with open(filepath, "r") as f:
            mylines = f.readlines()
            mylines = [i.strip() for i in mylines if len(i.strip()) > 0]
        for count_i, line in enumerate(mylines):
            for count_j, elem in enumerate(line):
                if elem == "m":
                    new_monster = Monster(id, count_j, count_i, elem)
                    self.inner.append(new_monster)
                    id += 1

    def draw(self, surface):
        if len(self.inner) == 0:
            raise ValueError("Doh! There are no tiles to display. 😕")
        for elem in self.inner:
            myrect = pygame.Rect(elem.x * TILESIZE, elem.y * TILESIZE, TILESIZE, TILESIZE)
            self.surface.blit(elem.image, myrect)

    def debug_print(self):
        for elem in self.inner:
            elem.debug_print()

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
#                   class Tile
# --------------------------------------------------------

class Tile:
    def __init__(self, id, x, y, kind_of_tile):
        filename = ""
        self.id = id
        self.x = int(x)
        self.y = int(y)
        self.kind_of_tile = kind_of_tile
        # ----
        if kind_of_tile == "g": filename = con.GRASS
        elif kind_of_tile == "d" : filename = con.DIRT
        else: raise ValueError("Error! kind of tile: ", kind_of_tile)
        # ---------------------
        self.rect = pygame.Rect(self.x * TILESIZE, self.y * TILESIZE, TILESIZE, TILESIZE)
        image_path = os.path.join("data", "images")
        self.image = pygame.image.load(os.path.join(image_path, filename)).convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILESIZE, TILESIZE))

    def debug_print(self):
        s = "id: {}, x: {}, y: {}, kind: {}"
        s = s.format(self.id, self.x, self.y, self.kind_of_tile)
        print(s)

# --------------------------------------------------------
#                   class Tiles
# --------------------------------------------------------

class Tiles:
    def __init__(self, screen):
        self.screen = screen
        self.inner = []
        self.current_tile = None
        self._load_data()

    def _load_data(self):
        self.inner = []
        filepath = os.path.join("data", "animal_map.txt")
        with open(filepath, "r") as f:
            mylines = f.readlines()
            mylines = [i.strip() for i in mylines if len(i.strip()) > 0]
        id = 0
        for count_i, myline in enumerate(mylines):
            temp_list = myline.split(";")
            temp_list = [i.strip() for i in temp_list if len(i.strip()) > 0]
            for count_j, elem in enumerate(temp_list):
                new_tile = Tile(id, count_j, count_i, elem)
                self.inner.append(new_tile)
                id += 1

    def draw(self, surface):
        if len(self.inner) == 0:
            raise ValueError("Doh! There are no tiles to display. 😕")
        for elem in self.inner:
            self.screen.blit(elem.image, elem.rect)

    def debug_print(self):
        for elem in self.inner:
            elem.debug_print()

# --------------------------------------------------------
#                   class Game
# --------------------------------------------------------

class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(TITLE)
        self.surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.BG_COLOR = con.LIGHTGREY
        self.keep_looping = True
        # ----
        self.tiles = Tiles(self.surface)
        self.monsters = Monsters(self.surface)
        self.apples = Apples(self.surface)
        self.agent = Agent(self.surface)

    def monster_in_view(self, observed_state):
        for row in observed_state:
            for cell in row:
                if cell['monster'] is not None:
                    return True
        return False   


    def compute_reward(self, observed_state):
        reward = 0
        # Penalty proportional to hunger
        hunger_penalty = (10-self.agent.hunger) * 0.1
        reward -= hunger_penalty
        print(f"Hunger penalty: {-hunger_penalty:.2f}")

        # Small reward for surviving
        survival_reward = 1
        reward += survival_reward
        print(f"Survival reward: {survival_reward}")

        # Check if agent observed a monster
        closest_monster_distance = self.agent.get_closest_monster_distance(observed_state)
        if closest_monster_distance is not None and closest_monster_distance > 0:
            monster_penalty = (1 / closest_monster_distance) * 10  # Adjust scaling factor as needed
            reward -= monster_penalty
            print(f"Monster proximity penalty: {-monster_penalty:.2f}")
        elif closest_monster_distance == 0:
            reward -= 100
            print(f'Monster ate you')
        else:
            print("No monster observed.")

        # Check if agent eats an apple
        if self.agent.check_if_eat_apple(self.apples):
            apple_reward = 50  # Positive reward for eating an apple
            reward += apple_reward
            self.agent.hunger = 10
            print(f"Apple consumed! Reward: {apple_reward}")
        else:
            print("No apple consumed.")

        print(f"Total reward: {reward:.2f}")
        return reward

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.keep_looping = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.keep_looping = False
                elif event.key == pygame.K_RIGHT:
                    self.agent_action()
                    self.monsters.update(self.agent)

    def agent_action(self):
        observed_state = self.agent.get_observed_state(self.tiles, self.monsters, self.apples)
        self.agent.agent_action(observed_state)
        reward = self.compute_reward(observed_state)
        """
        observed_state = self.agent.get_observed_state(self.tiles, self.monsters, self.apples)
        if self.monster_in_view(observed_state):
            # Move the agent to the right if possible
            if self.agent.x + 1 < TILES_HORIZONTAL:
                self.agent.x += 1
        """

    def update(self):
        pass  # No automatic updates needed

    def draw(self):
        self.surface.fill(self.BG_COLOR)
        self.tiles.draw(self.surface)
        self.monsters.draw(self.surface)
        self.apples.draw(self.surface)
        self.agent.draw(self.surface)
        pygame.display.update()

    def main(self):
        while self.keep_looping:
            self.events()
            self.update()
            self.draw()

if __name__ == "__main__":
    print("Hei")
    mygame = Game()
    mygame.main()
