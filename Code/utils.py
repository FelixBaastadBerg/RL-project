import numpy as np
import matplotlib.pyplot as plt

def visualize_states(file_path='longest_run_states.csv'):
    # Load the saved hidden states
    # states = np.load(file_path)
    states = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip header row

    # Ensure it's a 2D array: (time_steps, hidden_size)
    all_states = states.squeeze()  # Remove extra dimensions if shape is (128, 1, 122)
    pos_list = all_states[:, -2:]  # Extract agent position
    hidden_states = all_states[:, :-2]  # Extract hidden states

    if len(hidden_states.shape) == 3:  # Handle unexpected shapes
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

    # Plot each hidden unit's activation over time (imshow)
    plt.figure(figsize=(12, 6))
    plt.imshow(hidden_states.T, aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.title("Hidden States Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Unit")
    plt.show()

    # Plot x, y position as a function of time
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(pos_list.shape[0])  # Generate time step indices
    plt.plot(time_steps, pos_list[:, 0], label='X Position', marker='o')
    plt.plot(time_steps, pos_list[:, 1], label='Y Position', marker='x')
    plt.title("Agent Position Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.show()


