# import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

from PPO_RNN_2 import GridWorldEnv, PolicyValueNetwork, PPOAgent

if __name__ == "__main__":
    agent = PPOAgent(num_envs=100, num_steps=128, num_updates=3000, hidden_size=256,
                     grid_size=50, view_size=5, max_hunger=100, num_trees=5, num_predators=0, results_path=None)
    agent.test_trained_model()