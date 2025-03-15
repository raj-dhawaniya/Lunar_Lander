import gymnasium as gym
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor_fc1 = nn.Linear(input_dim, 256)
        self.actor_fc2 = nn.Linear(256, 128)
        self.actor_fc3 = nn.Linear(128, output_dim)
        # Critic
        self.critic_fc1 = nn.Linear(input_dim, 256)
        self.critic_fc2 = nn.Linear(256, 128)
        self.critic_fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # Actor
        a = torch.relu(self.actor_fc1(x))
        a = self.dropout(a)
        a = torch.relu(self.actor_fc2(a))
        action_probs = F.softmax(self.actor_fc3(a), dim=-1)
        # Critic
        c = torch.relu(self.critic_fc1(x))
        c = self.dropout(c)
        c = torch.relu(self.critic_fc2(c))
        value = self.critic_fc3(c)
        return action_probs, value

def select_action(model, state, episode=None, total_episodes=None, temp_start=1.0, temp_end=0.5, greedy=False):
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_probs, _ = model(state)
    
    if greedy:
        # Greedy selection for evaluation
        action = torch.argmax(action_probs).item()
    else:
        # Temperature-based sampling for training
        temperature = temp_start - (temp_start - temp_end) * (episode / total_episodes)
        action_probs = action_probs.pow(1 / temperature)
        action_probs /= action_probs.sum()
        action = torch.multinomial(action_probs, 1).item()
    return action

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def train_reinforcement_learning(env_name='LunarLander-v3', episodes=4000, learning_rate=0.0001, gamma=0.995):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = ActorCritic(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for episode in range(episodes):
        state, info = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        
        while not done:
            action = select_action(model, state, episode, episodes, temp_start=1.0, temp_end=0.5)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action_probs, value = model(state_tensor)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            log_probs.append(torch.log(action_probs[0, action]))
            values.append(value)
            rewards.append(np.clip(reward, -2, 2))
            state = next_state
        
        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values = torch.cat(values).squeeze()
        advantages = returns - values
        
        # Compute losses
        actor_loss = -torch.mean(torch.stack(log_probs) * advantages.detach())
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "best_policy.pth")
    print("Policy trained and saved as 'best_policy.pth'.")

def load_policy(filename, env_name='LunarLander-v3'):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = ActorCritic(input_dim, output_dim).to(device)  # Changed from PolicyNetwork to ActorCritic
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def play_policy(filename, episodes=5, env_name='LunarLander-v3'):
    model = load_policy(filename, env_name)
    env = gym.make(env_name, render_mode='human')
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = select_action(model, state, greedy=True)  # Use greedy selection for evaluation
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