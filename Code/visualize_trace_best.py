import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load the saved best run data
load_path = "best_run_data.pkl"

with open(load_path, "rb") as f:
    best_attempt = pickle.load(f)

# Input variables for time range
time_start = 3920  # Start time
length = 60        # Total length of the trace
time_end = min(time_start + length, len(best_attempt["agent_positions"]))
mv_avg_window = 5  # Window size for moving average

# Extract relevant data for plotting
agent_positions = np.array(best_attempt["agent_positions"])[time_start:time_end]
predator_positions = [np.array(trace)[time_start:time_end] for trace in best_attempt["predator_positions"]]
tree_positions = [pos for tree in best_attempt["tree_positions"] for pos in tree]

def moving_average(data, window_size):
    """Calculate the moving average for a list of 2D positions."""
    if len(data) < window_size:
        return np.array(data)  # If not enough data points, return the data as-is

    data = np.array(data)
    cumsum = np.cumsum(data, axis=0)
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# Smooth the positions
agent_positions = moving_average(agent_positions, mv_avg_window)
predator_positions = [moving_average(trace, mv_avg_window) for trace in predator_positions]

# Function to calculate opacity gradient
def calculate_opacity(num_points):
    return np.linspace(0.1, 1, num_points)  # Opacity from 10% to 100%

# Split the range into thirds
split1 = time_start
split2 = time_start + length // 3
split3 = time_start + 2 * length // 3
split4 = time_end

time_ranges = [
    (split1, split2),
    (split2, split3),
    (split3, split4)
]

# Plot function
def plot_trace(ax, agent_positions, predator_positions, tree_positions, time_range, opacity_values):
    tile_size = 1  # Tree tile size
    start, end = time_range

    # Plot tree positions as green squares
    if tree_positions:
        for tree_pos in tree_positions:
            rect = patches.Rectangle(
                (tree_pos[1] - tile_size / 2, tree_pos[0] - tile_size / 2),
                tile_size, tile_size,
                linewidth=0, edgecolor='none', facecolor='green', label='Tree' if tree_pos == tree_positions[0] else None
            )
            ax.add_patch(rect)

    # Plot agent positions
    for i, (x, y) in enumerate(agent_positions[start - time_start:end - time_start]):
        ax.scatter(
            y, x, color=(0, 0, 1, opacity_values[i]), s=50, label='Agent' if i == 0 else None
        )

    # Plot predator positions
    for predator_idx, predator_trace in enumerate(predator_positions):
        for i, (x, y) in enumerate(predator_trace[start - time_start:end - time_start]):
            ax.scatter(
                y, x, color=(1, 0, 0, opacity_values[i]), s=50, 
                label=f'Predator' if i == 0 else None
            )

    # Plot formatting
    ax.set_title(f"Timestep {start} to {end}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim(33, 55)
    ax.set_ylim(43, 65)
    ax.invert_yaxis()  # Align grid axes
    ax.legend()

# Generate opacity values
opacity_values = calculate_opacity(length//3)

# Plot the three time ranges
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, time_range in zip(axes, time_ranges):
    plot_trace(ax, agent_positions, predator_positions, tree_positions, time_range, opacity_values)

# Adjust layout
plt.tight_layout()
plt.show()
