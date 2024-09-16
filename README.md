# NeuroPlay-AI-Project

<div style="overflow: hidden;">
  <img src="https://github.com/user-attachments/assets/0b8fe695-ebf3-4d80-aa07-275a2f05c4db" alt="NeuroPlay Logo" width="400" height="400" style="float: left; margin-right: 20px;"/>
  <img src="rewards_and_running_mean.png" alt="Rewards Chart" width="600" height="400" style="float: left; margin-left: 20px;"/>
</div>

<p>Built an AI agent using reinforcement learning to autonomously improve at Pong, with a focus on agile prompt engineering.</p>
<h2>Key Features:</h2>
<ul>
  <li><strong>AI Agent</strong>: The AI controls one paddle in the game and learns how to return the ball effectively through trial and error.</li>
  <li><strong>Reinforcement Learning</strong>: The agent is trained by observing its performance, where successful returns are rewarded, and missed returns are penalized. This helps the agent improve its decision-making over time.</li>
  <li><strong>Neural Network Architecture</strong>: The network consists of two hidden layers, with 200 neurons in the first layer and 100 neurons in the second layer. The input is a preprocessed version of the game screen, which is fed into the neural network to predict optimal actions.</li>
  <li><strong>Adam Optimizer</strong>: The Adam optimizer is used to update the weights of the neural network. This allows the AI agent to adjust its learning rate dynamically, helping it converge faster.</li>
  <li><strong>Exploration and Exploitation</strong>: The model starts with an epsilon-greedy approach, where it initially explores random actions, then gradually shifts to exploiting its learned behavior as epsilon decays.</li>
  <li><strong>Performance Tracking</strong>: The AI’s performance is tracked by plotting the episode rewards and running mean, allowing us to visualize the learning progress.</li>
</ul>

<h2>How It Works:</h2>
<ul>
  <li><strong>Preprocessing</strong>: The game screen is preprocessed to reduce complexity by cropping, downsampling, and converting it to a binary format (paddles and ball vs background).</li>
  <li><strong>Forward Pass</strong>: The preprocessed screen is fed into the neural network, which computes the probability of taking certain actions (move the paddle up or down).</li>
  <li><strong>Backward Pass</strong>: After each game, the AI agent learns from its mistakes by updating the neural network weights using <strong>policy gradients</strong>. Gradients are calculated for all layers using the <strong>leaky ReLU activation function</strong>.</li>
  <li><strong>Discounted Rewards</strong>: The agent calculates discounted rewards, prioritizing immediate rewards over future rewards. This helps focus on short-term gains and improves gameplay faster.</li>
  <li><strong>Checkpointing</strong>: The progress, including model weights and training data, is saved periodically, allowing training to resume from the last saved checkpoint.</li>
</ul>

<h2>Files in the Repository:</h2>
<ul>
  <li><strong>Adam_Pong.py</strong>: The main script for training the neural network using reinforcement learning and Adam optimizer. This script contains the full implementation of the neural network, preprocessing, and training loop.</li>
  <li><strong>rewards_and_running_mean.png</strong>: A visual representation of the agent’s learning progress, where the running mean of rewards is plotted against the episodes.</li>
  <li><strong>training_progressV2.p</strong>: A pickle file storing the model's progress, weights, and rewards, allowing training to be resumed or evaluated later.</li>
  <li><strong>LOGO.png</strong>: The project logo representing the blend of neural networks and the Pong game.</li>
</ul>

<h2>Project Structure:</h2>
<ul>
  <li><strong>Neural Network</strong>: Built with 2 hidden layers, it processes the input (game screen) and predicts the optimal action for the AI agent.</li>
  <li><strong>Reinforcement Learning</strong>: The agent uses a reward-based system to learn. Successful actions (paddle returns the ball) are rewarded, and failed actions (paddle misses) are penalized.</li>
  <li><strong>Optimization</strong>: The Adam optimizer adjusts the weights of the neural network to improve the AI agent's performance efficiently.</li>
</ul>

<h2>Future Enhancements:</h2>
<ul>
  <li><strong>Multi-Agent Gameplay</strong>: Expanding the game to feature two AI agents playing against each other using competitive reinforcement learning.</li>
  <li><strong>Improved Environment Interaction</strong>: Exploring more complex environments and introducing different variations of the Pong game to improve AI versatility.</li>
</ul>
