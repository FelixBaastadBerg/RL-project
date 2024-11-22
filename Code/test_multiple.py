import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PPO_RNN import PPOAgent, GridWorldEnv, PolicyValueNetwork

def test_policy(policy_path, num_episodes=100):
    """
    Test a single policy and return the average survival time.
    """
    # Initialize the test environment
    test_env = GridWorldEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a new instance of the policy network
    input_size = 24  # Same as env._get_observation().shape[0]
    num_actions = 4
    hidden_size = 128
    policy = PolicyValueNetwork(input_size, num_actions, hidden_size).to(device)
    
    # Load the policy
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    # Test the policy
    total_steps = []
    for _ in range(num_episodes):
        obs = test_env.reset()
        obs = torch.tensor(obs, device=device).unsqueeze(0)
        hx = torch.zeros(1, 1, hidden_size, device=device)
        cx = torch.zeros(1, 1, hidden_size, device=device)

        steps = 0
        done = False
        while not done:
            with torch.no_grad():
                policy_logits, _, (hx, cx) = policy(obs, hx, cx)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()

            ob, reward, done = test_env.step(action.item())
            obs = torch.tensor(ob, device=device).unsqueeze(0)
            hx, cx = hx.detach(), cx.detach()
            steps += 1

        total_steps.append(steps)

    return np.mean(total_steps)


def test_multiple_policies(results_path, num_episodes=100):
    """
    Test multiple policies and plot the average survival times.
    """
    policy_files = []
    for root, _, files in os.walk(results_path):
        for file in files:
            if file.endswith(".pth"):
                policy_files.append(os.path.join(root, file))

    policy_files.sort()  # Ensure policies are tested in order

    average_survival_times = []
    for policy_path in policy_files:
        print(f"Testing policy: {policy_path}")
        avg_survival = test_policy(policy_path, num_episodes=num_episodes)
        average_survival_times.append(avg_survival)
        print(f"Average survival time: {avg_survival}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(policy_files)), average_survival_times, tick_label=[f"Policy {i}" for i in range(len(policy_files))])
    plt.title("Average Survival Times of Trained Policies")
    plt.xlabel("Policy")
    plt.ylabel("Average Survival Time")
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_path = "Results/241122-122207"  # Folder containing trained policies
    num_episodes = 1  # Number of test episodes per policy
    test_multiple_policies(results_path, num_episodes)
