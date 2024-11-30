import torch
import numpy as np
from PPO_RNN_2 import PPOAgent, GridWorldEnv

def test_agent_longest_run(agent, num_tests=100):
    max_duration = 0
    best_hidden_states = []
    device = agent.device

    for _ in range(num_tests):
        env = GridWorldEnv(grid_size=agent.grid_size, view_size=agent.view_size, max_hunger=agent.max_hunger)
        obs = torch.tensor(env.reset(), device=device).unsqueeze(0)
        hx = torch.zeros(1, 1, agent.hidden_size, device=device)
        cx = torch.zeros(1, 1, agent.hidden_size, device=device)
        hidden_states = []


        done = False
        duration = 0

        while not done:
            with torch.no_grad():
                policy_logits, _, (hx, cx) = agent.policy(obs, hx, cx)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()

            obs, _, done = env.step(action.item())
            obs = torch.tensor(obs, device=device).unsqueeze(0)
            hidden_states.append(hx.clone().squeeze(0).cpu().numpy())
            duration += 1

        if duration > max_duration:
            max_duration = duration
            best_hidden_states = hidden_states

    return max_duration, best_hidden_states


if __name__ == "__main__":
    # Load the trained agent
    agent = PPOAgent(num_envs=1, num_steps=128, num_updates=0)  # Training not needed
    agent.policy.load_state_dict(torch.load('../trained_policy.pth', weights_only=True))

    B = 10  # Number of test iterations
    duration, hidden_states = test_agent_longest_run(agent, num_tests=B)

    print(f"Longest Duration: {duration} steps")
    np.save('longest_run_hidden_states.npy', hidden_states)  # Save hidden states
