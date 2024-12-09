import torch
import numpy as np
from PPO_RNN_2 import GridWorldEnv, PolicyValueNetwork, PPOAgent
import matplotlib.pyplot as plt
import os

def calculate_distances(agent_positions, predator_positions):
    """Calculate distances between agent and each predator over time."""
    distances = []
    for predator_trace in predator_positions:
        distances_to_predator = [
            np.abs(np.array(agent_pos) - np.array(predator_pos)).sum()
            for agent_pos, predator_pos in zip(agent_positions, predator_trace)
        ]
        distances.append(distances_to_predator)
    return distances


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "12"  # Number of threads for OpenMP
    os.environ["MKL_NUM_THREADS"] = "12"  # Number of threads for Intel MKL

    # Configure PyTorch threading
    torch.set_num_threads(12)  # Number of threads for intra-op parallelism
    torch.set_num_interop_threads(12)  # Number of threads for inter-op parallelism
    # Initialize the PPOAgent with the same settings as the training
    agent = PPOAgent(
        num_envs=1,             # Single environment for easier tracking
        num_steps=100,          # Not relevant for testing, just required for initialization
        num_updates=1,          # Not relevant for testing
        hidden_size=256,        # Hidden layer size
        grid_size=120,           # Grid size
        view_size=7,            # Agent view size
        max_hunger=200,         # Hunger limit
        num_trees=8,            # Number of apple trees
        num_predators=8         # Number of predators
    )

    # Load the trained policy
    agent.policy.load_state_dict(torch.load("trained_policy.pth"))
    agent.policy.eval()

    # To store all attempts
    attempts = []

    for attempt in range(10):  # Run 100 independent attempts
        print(f"Attempt {attempt + 1}")
        # Initialize environment and tracking
        env = agent.envs[0]  # Single environment
        env.reset()
        agent.agent_positions = []  # To track the agent's positions
        agent.apple_positions = []  # To track apple positions over time
        agent.predator_positions = [[] for _ in range(env.num_predators)]  # To track each predator's positions

        # Initialize the hidden states for LSTM
        hx = torch.zeros(1, 1, agent.hidden_size, device=agent.device)
        cx = torch.zeros(1, 1, agent.hidden_size, device=agent.device)

        obs = torch.tensor(env._get_observation(), device=agent.device).unsqueeze(0)  # Initial observation

        # Perform rollout until the agent dies
        done = False
        steps_survived = 0
        total_reward = 0
        rewards_all = []
        while not done:
            with torch.no_grad():
                policy_logits, _, (hx, cx) = agent.policy(obs, hx, cx)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()

            # Perform the action in the environment
            obs_np, reward, done = env.step(action.item())
            obs = torch.tensor(obs_np, device=agent.device).unsqueeze(0)

            # Track positions
            agent.agent_positions.append(tuple(env.agent_pos))
            if env.apple_positions:
                agent.apple_positions.append(env.apple_positions[:])
            else:
                agent.apple_positions.append(None)

            # Track predators' positions
            for i, pos in enumerate(env.predator_positions):
                agent.predator_positions[i].append(tuple(pos))

            steps_survived += 1
            total_reward += reward
            rewards_all.append(reward)
            
            
        distances_to_predators = calculate_distances(agent.agent_positions, agent.predator_positions)
        predator_view_size = env.view_size_predator
        proximity_count = sum(
            sum(distances[t] <= predator_view_size for distances in distances_to_predators)
            for t in range(len(agent.agent_positions))
        )
            
        

        # Store the attempt's data
        attempts.append({
            "steps_survived": steps_survived,
            "total_reward": total_reward,
            "rewards_all": rewards_all,
            "agent_positions": agent.agent_positions,
            "apple_positions": agent.apple_positions,
            "tree_positions": env.apple_trees,
            "predator_positions": agent.predator_positions,
            "time_in_proximity_to_predator": proximity_count,
            "env": env  # Store the environment for plotting tree positions
        })

    print("Average Attempt:")
    print(f"Survival: {np.mean([a['steps_survived'] for a in attempts])} steps")
    print(f"Total reward: {np.mean([a['total_reward'] for a in attempts])}")
    
    # Find the attempt with the longest survival
    best_attempt = max(attempts, key=lambda x: x["time_in_proximity_to_predator"])
    # best_attempt = max(attempts, key=lambda x: x["steps_survived"])
    # best_attempt = max(attempts, key=lambda x: x["total_reward"])
    print("Best Attempt:")
    print(f"Longest survival: {best_attempt['steps_survived']} steps")
    print(f"Total reward: {best_attempt['total_reward']}")

    # Plot the best attempt
    def plot_best_attempt():
        env = best_attempt["env"]
        agent_positions = np.array(best_attempt["agent_positions"])
        apple_positions = best_attempt["apple_positions"]
        tree_positions = best_attempt["tree_positions"]
        predator_positions = best_attempt["predator_positions"]

        plt.figure(figsize=(8, 8))

        # Plot apple tree positions first
        apple_tree_positions = np.array([pos for tree in tree_positions for pos in tree])
        if len(apple_tree_positions) > 0:
            plt.scatter(
                apple_tree_positions[:, 1],
                apple_tree_positions[:, 0],
                color='brown',
                marker='s',
                label='Apple Tree'
            )

        # Plot apple positions on top of trees
        apple_positions_filtered = [pos for pos in apple_positions if pos is not None]
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
            plt.scatter(
                apple_x_positions,
                apple_y_positions,
                color='orange',
                marker='x',
                label='Apples'
            )

        # Plot predator positions
        for i, predator_trace in enumerate(predator_positions):
            predator_trace = np.array(predator_trace)
            plt.plot(
                predator_trace[:, 1],
                predator_trace[:, 0],
                '--',
                label=f'Predator {i + 1}',
                alpha=0.7
            )

        # Plot agent positions
        x_positions = agent_positions[:, 1]
        y_positions = agent_positions[:, 0]
        plt.scatter(
            x_positions,
            y_positions,
            c=range(len(x_positions)),
            cmap='viridis',
            marker='o',
            label='Agent'
        )

        # Finalize plot
        plt.colorbar(label='Timestep')
        plt.title('Best Attempt: Agent, Predator, and Apple Movement')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim(0, env.grid_size)
        plt.ylim(0, env.grid_size)
        plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
        plt.legend()
        plt.grid(True)
        plt.show()

    def moving_average(data, window_size):
        """Calculate the moving average for a list of 2D positions."""
        if len(data) < window_size:
            return np.array(data)  # If not enough data points, return the data as-is

        data = np.array(data)
        cumsum = np.cumsum(data, axis=0)
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    def plot_best_attempt_with_moving_average(window_size=5):
        """Plot the best attempt with a moving average applied to positions."""
        env = best_attempt["env"]
        agent_positions = np.array(best_attempt["agent_positions"])
        apple_positions = best_attempt["apple_positions"]
        tree_positions = best_attempt["tree_positions"]
        predator_positions = best_attempt["predator_positions"]

        # Calculate moving averages
        agent_positions_smoothed = moving_average(agent_positions, window_size)
        predator_positions_smoothed = [
            moving_average(trace, window_size) for trace in predator_positions
        ]

        plt.figure(figsize=(8, 8))

        # Plot apple tree positions first
        apple_tree_positions = np.array([pos for tree in tree_positions for pos in tree])
        if len(apple_tree_positions) > 0:
            plt.scatter(
                apple_tree_positions[:, 1],
                apple_tree_positions[:, 0],
                color='brown',
                marker='s',
                label='Apple Tree'
            )

        # Plot apple positions on top of trees
        apple_positions_filtered = [pos for pos in apple_positions if pos is not None]
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
            plt.scatter(
                apple_x_positions,
                apple_y_positions,
                color='orange',
                marker='x',
                label='Apples'
            )

        # Plot smoothed predator positions
        for i, predator_trace_smoothed in enumerate(predator_positions_smoothed):
            plt.plot(
                predator_trace_smoothed[:, 1],
                predator_trace_smoothed[:, 0],
                '--',
                label=f'Predator {i + 1}',
                alpha=0.7
            )

        # Plot smoothed agent positions
        x_positions = agent_positions_smoothed[:, 1]
        y_positions = agent_positions_smoothed[:, 0]
        plt.scatter(
            x_positions,
            y_positions,
            c=range(len(x_positions)),
            cmap='viridis',
            marker='o',
            label='Agent (Smoothed)'
        )

        # Finalize plot
        plt.colorbar(label='Timestep')
        plt.title('Best Attempt (Smoothed): Agent, Predator, and Apple Movement')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim(0, env.grid_size)
        plt.ylim(0, env.grid_size)
        plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
        plt.legend()
        plt.grid(True)
        # plt.show()

    def plot_accumulated_reward_and_distances(best_attempt):
        """Plot accumulated reward and distances to predators over time with proper color gradient."""
        # Extract relevant data
        agent_positions = np.array(best_attempt["agent_positions"])
        predator_positions = best_attempt["predator_positions"]
        rewards_all = best_attempt["rewards_all"]
        view_size = best_attempt["env"].view_size
        view_size = view_size // 2
        view_size_predator = best_attempt["env"].view_size_predator

        # Calculate accumulated reward over timesteps
        timesteps = len(agent_positions)
        accumulated_rewards = np.cumsum(rewards_all)

        # Calculate distances to predators
        distances_to_predators = calculate_distances(agent_positions, predator_positions)

        # Generate a colormap for timesteps
        colormap = plt.cm.viridis
        colors = colormap(np.linspace(0, 1, timesteps))

        # Plot setup
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Adjust the plot size to make room for the colorbar and legend
        ax1.set_position([0.1, 0.25, 0.75, 0.65])  # [left, bottom, width, height]

        # Plot accumulated reward with color gradient
        for i in range(1, timesteps):
            ax1.plot([i-1, i], [accumulated_rewards[i-1], accumulated_rewards[i]],
                    color=colors[i], linewidth=2)

        ax1.set_xlabel("Timesteps")
        pad = timesteps // 200
        ax1.set_xlim(-pad, timesteps + pad)
        ax1.set_ylabel("Accumulated Reward")
        ax1.set_title("Accumulated Reward and Distances to Predators Over Time")

        # Add a second y-axis for distances
        ax2 = ax1.twinx()
        predator_colors = plt.cm.tab10.colors  # Use distinct colors for predators
        for i, distances in enumerate(distances_to_predators):
            ax2.plot(range(timesteps), distances, color=predator_colors[i], linestyle='--', linewidth=2,
                    label=f"Distance to Predator {i + 1}")
        # Plot view sizes
        ax2.axhline(y=view_size, color='green', linestyle=':', label='Agent View Size')
        ax2.axhline(y=view_size_predator, color='red', linestyle=':', label='Predator View Size')

        ax2.set_ylabel("Distance to Predators")

        # Add a horizontal colorbar at the bottom
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=timesteps))
        sm.set_array([])  # Required for colorbar
        cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.05, aspect=30)
        cbar.set_label('Timestep')

        # Place the legend horizontally at the bottom
        num_predators = len(distances_to_predators)
        fig.legend(
            loc="upper left",
            bbox_to_anchor=(0.125, 0.875),  # Centered horizontally below the plot
            # ncol=num_predators + 1,      # Arrange predator lines and reward label horizontally
        )

        # Finalize the plot
        plt.grid(True)
        # plt.show()





    print(best_attempt["time_in_proximity_to_predator"])
    plot_best_attempt_with_moving_average(window_size=10)
    plot_accumulated_reward_and_distances(best_attempt)
    plt.show()
    # plot_best_attempt()
