import gymnasium as gym
import numpy as np
import argparse
import os
<<<<<<< HEAD
=======
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
>>>>>>> dcd26ed35d234175f5b43c2e3b688ab19abb7e02

def policy_action(params, observation):
    # The policy is a linear mapping from the 8-dimensional observation to 4 action scores.
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

<<<<<<< HEAD
def evaluate_policy(params, episodes=3, render=False):
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make('LunarLander-v3', render_mode='human' if render else 'rgb_array')
        observation, info = env.reset()
        episode_reward = 0.0
=======
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
>>>>>>> dcd26ed35d234175f5b43c2e3b688ab19abb7e02
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
<<<<<<< HEAD
        env.close()
        total_reward += episode_reward
    return total_reward / episodes

def simulated_binary_crossover(parent1, parent2, eta_c=15):
    gene_size = parent1.shape[0]
    child = np.empty(gene_size)
    for i in range(gene_size):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta_c + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
        child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
    return child

def polynomial_mutation(child, mutation_rate=0.1, eta_m=20, lower_bound=-5, upper_bound=5):
    gene_size = child.shape[0]
    for i in range(gene_size):
        if np.random.rand() < mutation_rate:
            x = child[i]
            # Compute normalized distances from the bounds.
            delta1 = (x - lower_bound) / (upper_bound - lower_bound)
            delta2 = (upper_bound - x) / (upper_bound - lower_bound)
            r = np.random.rand()
            if r < 0.5:
                delta_q = (2 * r + (1 - 2 * r) * ((1 - delta1) ** (eta_m + 1))) ** (1 / (eta_m + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - r) + 2 * (r - 0.5) * ((1 - delta2) ** (eta_m + 1))) ** (1 / (eta_m + 1))
            child[i] = x + delta_q * (upper_bound - lower_bound)
            # Ensure the gene remains within bounds.
            child[i] = np.clip(child[i], lower_bound, upper_bound)
    return child

def genetic_algorithm(population_size=50, num_generations=50, elite_frac=0.2,
                      mutation_rate=0.1, lower_bound=-5, upper_bound=5):
    gene_size = 8 * 4 + 4  # 8 inputs x 4 outputs + 4 biases = 36 parameters
    population = np.random.randn(population_size, gene_size)
    
    num_elites = int(population_size * elite_frac)
    best_reward = -np.inf
    best_params = None
=======
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
>>>>>>> dcd26ed35d234175f5b43c2e3b688ab19abb7e02

    for generation in range(num_generations):
        fitness = np.array([evaluate_policy(individual, episodes=3) for individual in population])
        elite_indices = fitness.argsort()[::-1][:num_elites]
        elites = population[elite_indices]
        
<<<<<<< HEAD
        if fitness[elite_indices[0]] > best_reward:
            best_reward = fitness[elite_indices[0]]
            best_params = population[elite_indices[0]].copy()
=======
        while not done:
            action = select_action(model, state, greedy=True)  # Use greedy selection for evaluation
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
>>>>>>> dcd26ed35d234175f5b43c2e3b688ab19abb7e02
        
        print(f"Generation {generation+1}: Best Average Reward = {best_reward:.2f}")
        
        # Create new population using elitism, simulated binary crossover, and polynomial mutation.
        new_population = []
        new_population.extend(elites)
        while len(new_population) < population_size:
            parents = elites[np.random.choice(num_elites, 2, replace=False)]
            child = simulated_binary_crossover(parents[0], parents[1], eta_c=15)
            child = polynomial_mutation(child, mutation_rate=mutation_rate, eta_m=20,
                                          lower_bound=lower_bound, upper_bound=upper_bound)
            new_population.append(child)
        
        population = np.array(new_population)
    
    return best_params

def train_and_save(filename, population_size=50, num_generations=50, elite_frac=0.2,
                   mutation_rate=0.1, lower_bound=-5, upper_bound=5):
    best_params = genetic_algorithm(population_size, num_generations, elite_frac,
                                    mutation_rate, lower_bound, upper_bound)
    np.save(filename, best_params)
    print(f"Best policy saved to {filename}")
    return best_params

def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params

def play_policy(best_params, episodes=5):
    test_reward = evaluate_policy(best_params, episodes=episodes, render=True)
    print(f"Average reward of the best policy over {episodes} episodes: {test_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using GA with SBX and polynomial mutation.")
    parser.add_argument("--train", action="store_true", help="Train the policy using GA and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        # Train and save the best policy
        best_params = train_and_save(
            args.filename,
            population_size=100,
            num_generations=100,
            elite_frac=0.2,
            mutation_rate=0.1,
            lower_bound=-5,
            upper_bound=5
        )
    elif args.play:
        # Load and play with the best policy
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
