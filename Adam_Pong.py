# Import necessary libraries
import numpy as np  # For numerical operations
import pickle  # For saving and loading Python objects
import gym  # To interact with the gym environment (Pong game)
import matplotlib.pyplot as plt  # For plotting purposes

# Hyperparameters: These are the settings that can be tuned to optimize performance
H = 200  # Number of neurons in the first hidden layer of the neural network
H2 = 100  # Number of neurons in the second hidden layer of the neural network
batch_size = 20  # Number of episodes to play before updating the network weights
learning_rate = 1e-4  # Speed at which the network learns during training
gamma = 0.99  # Discount factor for future rewards, to prioritize immediate rewards over distant ones
learning_rate_decay = 0.99  # Factor by which the learning rate decreases over time
lr_decay_every = 1000  # Number of episodes after which to apply learning rate decay
resume = True  # Whether to resume training from a saved model
render = True  # Whether to display the game screen during training
epsilon = 1.0  # Initial exploration rate, determining the probability of taking a random action
epsilon_final = 0.01  # Final value of epsilon, after decay
epsilon_decay = 0.995  # Factor by which epsilon decreases over time, making the agent act more greedily

# Adam Optimizer parameters: These settings help adjust how the optimizer updates the network weights
beta1 = 0.9
beta2 = 0.999
epsilon_adam = 1e-8

# Model initialization: Define the structure of the neural network
D = 80 * 80  # Input dimensionality: size of a preprocessed game frame

# Activation and utility functions
def sigmoid(x):
    """ Compute the sigmoid function, used for calculating probabilities """
    return 1.0 / (1.0 + np.exp(-x))

def leaky_relu(x, alpha=0.01):
    """ Compute the leaky ReLU activation, allowing for a small gradient when the unit is not active """
    return np.where(x > 0, x, x * alpha)

def prepro(I):
    """ Preprocess game frames for input into the network: crop, downsample, erase background, and binarize """
    I = I[35:195]  # Crop to play area
    I = I[::2, ::2, 0]  # Downsample by factor of 2 and take the red channel (simplification)
    I[I == 144] = 0  # Erase background type 1
    I[I == 109] = 0  # Erase background type 2
    I[I != 0] = 1  # Set everything else (paddles, ball) to 1
    return I.astype(np.float64).ravel()  # Flatten into vector

def discount_rewards(r):
    """ Compute discounted rewards over an episode: rewards that come later are worth less than immediate rewards """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # Reset the sum at the end of an episode
        running_add = running_add * gamma + r[t]  # Apply discount factor
        discounted_r[t] = running_add
    return discounted_r

def layer_norm(x):
    """ Normalize the activations of a layer by subtracting the mean and dividing by the standard deviation """
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / (std + 1e-5)  # Prevent division by zero

# Neural network functions
def policy_forward(x):
    """ Forward pass through the policy network: from inputs to action probabilities """
    z1 = np.dot(model['W1'], x)  # First layer multiplication
    norm_z1 = layer_norm(z1)  # Normalize first layer's activations
    h = leaky_relu(norm_z1)  # Apply activation function

    z2 = np.dot(model['W2'], h)  # Second layer multiplication
    norm_z2 = layer_norm(z2)  # Normalize second layer's activations
    h2 = leaky_relu(norm_z2)  # Apply activation function

    logp = np.dot(model['W3'], h2)  # Output layer multiplication
    p = sigmoid(logp)  # Calculate probability of taking action 2 (up in Pong)
    return p, h, h2  # Return probability and hidden layer values for backprop

def policy_backward(eph, eph2, epdlogp):
    """ Backward pass (gradient descent): compute gradients for updating network weights """
    dW3 = np.dot(eph2.T, epdlogp).ravel()  # Gradient for output layer weights
    dh2 = np.outer(epdlogp, model['W3'])  # Backprop into second hidden layer
    dh2[eph2 <= 0] = 0  # Apply gradient of leaky ReLU
    dW2 = np.dot(dh2.T, eph)  # Gradient for second layer weights
    
    dh = np.dot(dh2, model['W2'])  # Backprop into first hidden layer
    dh[eph <= 0] = 0  # Apply gradient of leaky ReLU
    dW1 = np.dot(dh.T, epx)  # Gradient for first layer weights
    return {'W1': dW1, 'W2': dW2, 'W3': dW3}  # Return all gradients for update

# Persistence functions: save and load the training progress
def save_progress(file_path, model, grad_buffer, m_cache, v_cache, episode_number, running_rewards, episode_rewards, epsilon):
    progress = {
        'model': model,
        'grad_buffer': grad_buffer,
        'm_cache': m_cache,
        'v_cache': v_cache,
        'episode_number': episode_number,
        'running_rewards': running_rewards,
        'episode_rewards': episode_rewards,
        'epsilon': epsilon
    }
    with open(file_path, 'wb') as f:
        pickle.dump(progress, f)
    print(f"Progress saved to {file_path}")

def load_progress(file_path):
    try:
        with open(file_path, 'rb') as f:
            progress = pickle.load(f)
        print(f"Progress loaded from {file_path}")
        return progress
    except FileNotFoundError:
        print(f"No saved progress found at {file_path}. Starting from scratch.")
        return None

# Setup for plotting
def save_rewards_data(episode_rewards, running_rewards, save_path='rewards_and_running_mean.png'):
    if episode_rewards and running_rewards:
        plt.figure(figsize=(10, 5))
        
        # Plot individual episode rewards
        plt.scatter(range(len(episode_rewards)), episode_rewards, s=10, label='Reward per Episode', color='tab:blue', alpha=0.5)
        
        # Plot running mean of rewards to visualize the performance trend over time
        plt.plot(running_rewards, label='Running Mean', color='tab:red', linewidth=2)
        
        # Adding legend and labels
        plt.title('Episode Rewards and Running Mean')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc='upper left')
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()


save_file_path = 'training_progressV2.p'

if resume:
    # Attempt to load existing progress
    progress = load_progress(save_file_path)

    if progress:
        model = progress['model']
        grad_buffer = progress['grad_buffer']
        m_cache = progress['m_cache']
        v_cache = progress['v_cache']
        episode_number = progress['episode_number']
        running_rewards = progress['running_rewards']
        episode_rewards = progress['episode_rewards']
        epsilon = progress['epsilon']
        
    else:
        # Initialize everything as before if no saved progress
        episode_number = 0
        running_rewards = []
        episode_rewards = []
        model = {
            'W1': np.random.randn(H, D) / np.sqrt(D),
            'W2': np.random.randn(H2, H) / np.sqrt(H),
            'W3': np.random.randn(H2) / np.sqrt(H2)
        }
        grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
        m_cache = {k: np.zeros_like(v) for k, v in model.items()}
        v_cache = {k: np.zeros_like(v) for k, v in model.items()}
else:
    # Initialize for training from scratch
    episode_number = 0
    running_rewards = []
    episode_rewards = []
    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),
        'W2': np.random.randn(H2, H) / np.sqrt(H),
        'W3': np.random.randn(H2) / np.sqrt(H2)
    }
    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    m_cache = {k: np.zeros_like(v) for k, v in model.items()}
    v_cache = {k: np.zeros_like(v) for k, v in model.items()}

env = gym.make("Pong-v0", render_mode='human' if render else None)
observation = env.reset()
prev_x = None
xs, hs, h2s, dlogps, drs = [], [], [], [], []
running_reward = None
reward_sum = 0

plot_update_freq = 10  # Frequency of plot updates, e.g., every 100 episodes

training_episode = 20000 # stop after 1000 episodes

while episode_number <= training_episode:
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    h, h2 = np.zeros((H,)), np.zeros((H2,))

    # Initialize aprob to a default value
    aprob = 0.5  # Default probability when action is chosen randomly

    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        aprob, h, h2 = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

    # Record the intermediates needed for backprop
    xs.append(x)
    hs.append(h)
    h2s.append(h2)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        eph2 = np.vstack(h2s)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, h2s, dlogps, drs = [], [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, eph2, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                m = beta1 * m_cache[k] + (1 - beta1) * g  # Update biased first moment estimate
                v = beta2 * v_cache[k] + (1 - beta2) * (g ** 2)  # Update biased second raw moment estimate
                m_hat = m / (1 - beta1 ** episode_number)  # Compute bias-corrected first moment estimate
                v_hat = v / (1 - beta2 ** episode_number)  # Compute bias-corrected second raw moment estimate
                model[k] += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)  # Update parameters
                grad_buffer[k] = np.zeros_like(v)  # Reset gradients
                m_cache[k], v_cache[k] = m, v  # Update Adam caches
                
        if episode_number % lr_decay_every == 0:
            learning_rate *= learning_rate_decay

        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'resetting env. episode {episode_number} reward total was {reward_sum}. running mean: {running_reward}')

        episode_rewards.append(reward_sum)
        running_rewards.append(running_reward)
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

        if episode_number % 100 == 0:
            save_progress(save_file_path, model, grad_buffer, m_cache, v_cache, episode_number, running_rewards, episode_rewards, epsilon)

        if episode_number % plot_update_freq == 0 and episode_number != 0:
            save_rewards_data(episode_rewards, running_rewards)

    #if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        #print(f'ep {episode_number}: game finished, reward: {reward}' + (' !!!!!!!!' if reward == 1 else ''))