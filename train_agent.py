import gymnasium as gym
import numpy as np
import argparse
import os

def policy_action(params, observation):
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

def evaluate_policy(params, episodes=10, render=False):  # Increased evaluation episodes for stability
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make('LunarLander-v3', render_mode='human' if render else 'rgb_array')
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / episodes

def simulated_binary_crossover(parent1, parent2, eta_c=10):  # Lower eta_c for more exploration
    gene_size = parent1.shape[0]
    child = np.empty(gene_size)
    for i in range(gene_size):
        u = np.random.rand()
        beta = (2 * u) ** (1 / (eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
        child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
    return child

def polynomial_mutation(child, mutation_rate=0.2, eta_m=15, lower_bound=-5, upper_bound=5):
    gene_size = child.shape[0]
    for i in range(gene_size):
        if np.random.rand() < mutation_rate:
            x = child[i]
            delta = np.random.normal(0, 0.2)  # Gaussian mutation for better search
            child[i] = np.clip(x + delta, lower_bound, upper_bound)
    return child

def genetic_algorithm(population_size=200, num_generations=200, elite_frac=0.3,
                      mutation_rate=0.2, lower_bound=-5, upper_bound=5):
    gene_size = 8 * 4 + 4  # 8 inputs x 4 outputs + 4 biases = 36 parameters
    population = np.random.randn(population_size, gene_size)
    num_elites = int(population_size * elite_frac)
    best_reward = -np.inf
    best_params = None

    for generation in range(num_generations):
        fitness = np.array([evaluate_policy(individual, episodes=10) for individual in population])
        elite_indices = fitness.argsort()[::-1][:num_elites]
        elites = population[elite_indices]
        
        if fitness[elite_indices[0]] > best_reward:
            best_reward = fitness[elite_indices[0]]
            best_params = population[elite_indices[0]].copy()
        
        print(f"Generation {generation+1}: Best Average Reward = {best_reward:.2f}")
        
        new_population = list(elites)
        while len(new_population) < population_size:
            parents = elites[np.random.choice(num_elites, 2, replace=False)]
            child = simulated_binary_crossover(parents[0], parents[1], eta_c=10)
            child = polynomial_mutation(child, mutation_rate=mutation_rate)
            new_population.append(child)
        
        population = np.array(new_population)
    
    return best_params

def train_and_save(filename, population_size=200, num_generations=200, elite_frac=0.3,
                   mutation_rate=0.2, lower_bound=-5, upper_bound=5):
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
        best_params = train_and_save(
            args.filename,
            population_size=200,
            num_generations=200,
            elite_frac=0.3,
            mutation_rate=0.2,
            lower_bound=-5,
            upper_bound=5
        )
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
