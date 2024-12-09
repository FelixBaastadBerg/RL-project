from env import *
from ppo import *
from utils import *


if __name__ == "__main__":
    #parameters
    num_envs = 8 # number of environments / simulations used for training
    num_steps = 128 # number of steps per environment for
    num_updates = 100#1000 # number of updates to the policy??
    hidden_size = 256 # size of hidden lstm layer
    grid_size = 20 # size of the grid world, that is it's a grid_size x grid_size grid
    view_size = 7 # size of the agent's view in each direction
    max_hunger = 100 # maximum hunger value for the agent before it diess
    num_trees = 2 # number of trees in the grid
    num_predators = 1 # number of predators in the grid

    agent = PPOAgent(num_envs=num_envs, num_steps=num_steps, num_updates=num_updates, hidden_size=hidden_size,
                     grid_size=grid_size, view_size=view_size, max_hunger=max_hunger, num_trees=num_trees, num_predators=num_predators, results_path=None)
    agent.train()
    
    # agent.test_trained_model()