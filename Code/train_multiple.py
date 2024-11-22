import os
from datetime import datetime
from itertools import product
import PPO_RNN

test = 10

now = datetime.now()
date_time_string = now.strftime("%y%m%d-%H%M%S")
results_path = f'Results/' + date_time_string
os.makedirs(results_path, exist_ok=True)

variable_ranges = {
        "num_envs": (10, 50, 10),  # Test 10, 20, 30, 40
        "num_steps": (64, 257, 64),  # Test 64, 128, 192, 256
        "learning_rate": (1e-4, 6e-4, 1e-4),  # Test 1e-4, 2e-4, ..., 5e-4
        "gamma": (0.9, 1.01, 0.05)  # Gamma in steps of 0.05
    }


if __name__ == "__main__":
    # for i in range(test):
    #     agent = PPO_RNN.PPOAgent(num_updates=100*(i+1))
    #     agent.train(results_path, i)
    for variable in variable_ranges:
        print(variable)