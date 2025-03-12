import gymnasium as gym
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

def select_action(policy_net, state, temperature=0.8):
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_probs = policy_net(state)
    action_probs = action_probs.pow(1 / temperature)  # Temperature scaling for exploration
    action_probs /= action_probs.sum()  # Normalize
    action = torch.multinomial(action_probs, 1).item()
    return action

def train_reinforcement_learning(env_name='LunarLander-v3', episodes=2000, learning_rate=0.0005, gamma=0.99):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_net = PolicyNetwork(input_dim, output_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    for episode in range(episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            action = select_action(policy_net, state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            log_probs.append(torch.log(policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0, action]))
            rewards.append(np.clip(reward, -1, 1))  # Clip rewards to stabilize training
        
        # Compute discounted rewards
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute policy loss with entropy bonus
        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards) - 0.01 * torch.sum(torch.stack(log_probs).exp() * torch.stack(log_probs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Loss: {loss.item():.4f}")
    
    torch.save(policy_net.state_dict(), "best_policy.pth")
    print("Policy trained and saved.")

def load_policy(filename, env_name='LunarLander-v3'):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_net = PolicyNetwork(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load(filename, map_location=device))
    policy_net.eval()
    return policy_net

def play_policy(filename, episodes=5, env_name='LunarLander-v3'):
    policy_net = load_policy(filename, env_name)
    env = gym.make(env_name, render_mode='human')
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = select_action(policy_net, state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using Reinforcement Learning.")
    parser.add_argument("--train", action="store_true", help="Train the policy using RL and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.pth", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        train_reinforcement_learning()
    elif args.play:
        play_policy(args.filename)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
