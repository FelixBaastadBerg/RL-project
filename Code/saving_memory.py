import torch
import numpy as np
from PPO_RNN_Experiment_1 import GridWorldEnv
from PPO_RNN_Experiment_2 import PPOAgent
from rich.progress import track

def test_agent_longest_run(agent, num_tests=100):
    #print("#")
    max_duration = 0
    best_hidden_states = []
    device = agent.device

    for _ in range(num_tests):
        env = GridWorldEnv(grid_size=agent.grid_size, view_size=agent.view_size, max_hunger=agent.max_hunger)
        obs = torch.tensor(env.reset(), device=device).unsqueeze(0)
        hx = torch.zeros(1, 1, 256, device=device)
        cx = torch.zeros(1, 1, 256, device=device)
        states = []
        states = []


        done = False
        duration = 0

        while not done:
            with torch.no_grad():
                policy_logits, _, (hx, cx) = agent.policy(obs, hx, cx)
                #print("#")
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()

            obs, _, done = env.step(action.item())
            pos = np.array(env.agent_pos, dtype=np.float32)
            obs = torch.tensor(obs, device=device).unsqueeze(0)
            state = np.concatenate((hx.clone().squeeze(0).cpu().numpy()[0], pos))
            states.append(state)
            duration += 1

        if duration > max_duration:
            max_duration = duration
            best_states = states

    return max_duration, best_states


if __name__ == "__main__":
    #print("1")
    agent = PPOAgent(num_envs=1, num_steps=128, num_updates=0, view_size=5)  # Training not needed
    #print("2")
    agent.policy.load_state_dict(torch.load('policy_lstm_1_apple.pth'))
    #print("3")

    B = 50  # Number of test iterations
    all_states = []  # To store all states across runs
    max_durations = []  # To store durations of each run
    L = 50
    # Loop over multiple runs and collect states
    for _ in track(range(L)):
        duration, states = test_agent_longest_run(agent, num_tests=B)  # Single test run
        all_states.extend(states[50:])  # Add states to the larger 2D list
        max_durations.append(duration)

    # Prepare the CSV header
    header = ""
    for i in range(len(all_states[0]) - 2):  # Assuming all_states[0] has hidden states + x, y
        header += f"hidden_{i},"
    header += "x,y"

    # Save the combined states to a CSV file
    print(f"Longest Durations from Each Run: {max_durations}")
    np.savetxt(f'CSV_files/hidden_data{L}.csv', all_states, delimiter=',', header=header, comments='')

    print(f"Total States Collected: {len(all_states)}")


    # np.save('longest_run_states.npy', states)  # Save hidden states
