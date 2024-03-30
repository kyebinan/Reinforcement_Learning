# Classic Reinforcement Learning Algorithms

## 1- Q-Learning
This is a model-free algorithm that learns the value of an action in a particular state. It uses a Q-table and updates it using a simple formula based on the Bellman equation. 
Q-Learning is foundational for understanding how agents can learn to act optimally in Markov Decision Processes (MDPs).

<img src="https://github.com/kyebinan/Reinforcement_Learning/blob/main/Classics/images/qlearning_snake.png?raw=true" alt="Q-Learning Snake" width="600" height="400">

https://github.com/kyebinan/Reinforcement_Learning/assets/155234248/98dcac3a-bbc7-43a1-b3c9-8ae03ea50cbe



## 2- SARSA (State-Action-Reward-State-Action)
Similar to Q-Learning, SARSA is a model-free algorithm but it updates its Q-values using the action actually taken from the next state, rather than the maximum reward action. 
This makes SARSA an on-policy algorithm in contrast to the off-policy nature of Q-Learning.

<img src="https://github.com/kyebinan/Reinforcement_Learning/blob/main/Classics/images/sarsa_snake.png?raw=true" alt="Q-Learning Snake" width="600" height="400">


## 3- Deep Q-Networks (DQN)
Building on Q-Learning, DQN uses deep neural networks to approximate Q-values, allowing it to handle high-dimensional state spaces. Key innovations include experience replay and fixed Q-targets to stabilize training.

## 4- Policy Gradient Methods
These methods, including the REINFORCE algorithm, learn a parameterized policy that can select actions without consulting a value function. 
The approach is based on optimizing the expected return by adjusting the policy parameters in the direction of the gradient of the expected return.

## 5- Actor-Critic Methods 
 Combining ideas from policy gradients and value function methods, actor-critic methods maintain both a policy (actor) and a value function (critic) to update the policy gradient in a more stable and efficient manner.
