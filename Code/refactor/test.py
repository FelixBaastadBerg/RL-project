from train import *


if __name__ == "__main__":
    pass
    # Load the trained agent
    # agent = PPOAgent(num_envs=1, num_steps=128, num_updates=0)  # Training not needed
    # agent.policy.load_state_dict(torch.load('./trained_policy.pth', weights_only=True))

    # agent.test_trained_model()

    # np.save('longest_run_states.npy', states)  # Save hidden states