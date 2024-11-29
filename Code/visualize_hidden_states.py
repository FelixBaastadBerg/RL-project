import numpy as np
import matplotlib.pyplot as plt

def visualize_hidden_states(file_path='longest_run_hidden_states.npy'):
    # Load the saved hidden states
    hidden_states = np.load(file_path)

    # Ensure it's a 2D array: (time_steps, hidden_size)
    hidden_states = hidden_states.squeeze()  # Remove extra dimensions if shape is (128, 1, 122)

    if len(hidden_states.shape) == 3:  # Handle unexpected shapes
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

    # Plot each hidden unit's activation over time
    plt.figure(figsize=(12, 6))
    plt.imshow(hidden_states.T, aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.title("Hidden States Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Unit")
    plt.show()

if __name__ == "__main__":
    visualize_hidden_states()
