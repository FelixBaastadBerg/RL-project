import os
from datetime import datetime
import PPO_RNN_2 as PPO_RNN
import numpy as np


now = datetime.now()
date_time_string = now.strftime("%y%m%d-%H%M%S")
results_path_root = f'Results/' + date_time_string
os.makedirs(results_path_root, exist_ok=True)

# Test variable ranges:
#      "variable_name": (min_value, max_value, step) --> [min, min+step, min+2*step, ..., max]
variable_ranges = {
        # "num_envs": (10, 100, 10),
        # "num_steps": (64, 256, 64), 
        "num_updates": (2000, 8100, 500), 
        # "hidden_size": (64, 256, 64), 
        # "grid_size": (10, 50, 10),  
        # "view_size": (3, 7, 2), 
        # "max_hunger": (50, 150, 20),
        # "num_trees": (1, 10, 1), 
        # "num_predators": (0, 10, 1)  
    }


if __name__ == "__main__":
    total_runs = 0
    for variable in variable_ranges:
        value_min, value_max, value_step = variable_ranges[variable]
        total_runs += (value_max - value_min) // value_step + 1
    # Ask for confirmation
    print(f"You are about to start {total_runs} runs. Press any key to continue or 'q' to quit.")
    response = input()
    if response == 'q':
        exit()

    for variable in variable_ranges:
        print(variable)
        results_path = results_path_root + f'/{variable}'
        os.makedirs(results_path, exist_ok=True)

        value_min, value_max, value_step = variable_ranges[variable]
        values = np.arange(value_min, value_max+1, value_step)

        for value in values:
            value = int(value)
            print(f"Testing {variable} = {value}")
            if variable == "num_envs":
                agent = PPO_RNN.PPOAgent(num_envs=value, results_path=results_path)
            elif variable == "num_steps":
                agent = PPO_RNN.PPOAgent(num_steps=value, results_path=results_path)
            elif variable == "num_updates":
                agent = PPO_RNN.PPOAgent(num_updates=value, results_path=results_path)
            elif variable == "hidden_size":
                agent = PPO_RNN.PPOAgent(hidden_size=value, results_path=results_path)
            elif variable == "grid_size":
                agent = PPO_RNN.PPOAgent(grid_size=value, results_path=results_path)
            elif variable == "view_size":
                agent = PPO_RNN.PPOAgent(view_size=value, results_path=results_path)
            elif variable == "max_hunger":
                agent = PPO_RNN.PPOAgent(max_hunger=value, results_path=results_path)
            elif variable == "num_trees":
                agent = PPO_RNN.PPOAgent(num_trees=value, results_path=results_path)
            elif variable == "num_predators":
                agent = PPO_RNN.PPOAgent(num_predators=value, results_path=results_path)
            agent.train()
    


