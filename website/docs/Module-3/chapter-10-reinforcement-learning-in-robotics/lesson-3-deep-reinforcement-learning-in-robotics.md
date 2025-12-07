---
title: 'Deep Reinforcement Learning in Robotics'
description: 'Exploring deep reinforcement learning for robotics applications, covering DRL algorithms (DQN, policy gradients, actor-critic), and their implementation in robotic systems'
chapter: 10
lesson: 3
module: 3
sidebar_label: 'Deep Reinforcement Learning in Robotics'
sidebar_position: 3
tags: ['Deep RL', 'Robotics', 'DQN', 'Policy Gradients', 'Actor-Critic', 'Robot Control']
keywords: ['deep reinforcement learning', 'robotics', 'DQN', 'policy gradients', 'actor-critic', 'robot control', 'neural networks', 'continuous control']
---

# Deep Reinforcement Learning in Robotics

## Overview

Deep Reinforcement Learning (DRL) has revolutionized robotics by enabling robots to learn complex behaviors directly from high-dimensional sensory inputs. By combining deep neural networks with reinforcement learning algorithms, DRL allows robots to learn policies for tasks ranging from manipulation and navigation to locomotion and multi-agent coordination. This lesson covers the fundamental DRL algorithms and their specific applications to robotics.

## Deep Q-Network (DQN) for Robotics

### DQN Fundamentals

Deep Q-Networks extend classical Q-learning to handle high-dimensional state spaces using neural networks to approximate the Q-function:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network periodically
        self.update_target_network()

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()

    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network_if_needed(self, step, update_freq=1000):
        """Update target network periodically"""
        if step % update_freq == 0:
            self.update_target_network()
```

### Double DQN for Robotics

Double DQN addresses overestimation bias in Q-learning:

```python
class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min)

    def replay(self):
        """Train using Double DQN update"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Double DQN: Use main network to select actions, target network to evaluate
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Select best actions using main network
        best_actions = self.q_network(next_states).max(1)[1]

        # Evaluate using target network
        next_q_values = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Dueling DQN for Robotics

Dueling DQN separates value and advantage estimation:

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        # Common feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network periodically
        self.update_target_network()

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
```

## Policy Gradient Methods

### REINFORCE Algorithm

REINFORCE is a foundational policy gradient algorithm:

```python
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = 0.99

        # For storing episode data
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        self.log_probs.append(log_prob)
        return action.item()

    def calculate_returns(self, rewards):
        """Calculate discounted returns for policy gradient"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        returns = self.calculate_returns(self.rewards)

        # Calculate policy gradient
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Reset episode data
        self.log_probs = []
        self.rewards = []

    def finish_episode(self):
        """Complete episode and update policy"""
        if len(self.rewards) > 0:
            self.update_policy()
```

### Actor-Critic Methods

Actor-critic methods combine policy and value learning:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)

        self.gamma = 0.99

    def act(self, state):
        """Sample action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

        return action.item(), action_dist.log_prob(action)

    def update(self, state, action, reward, next_state, done):
        """Update actor and critic networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Get current and next state values
        _, current_value = self.actor_critic(state_tensor)
        _, next_value = self.actor_critic(next_state_tensor)

        # Calculate advantage (TD error)
        target_value = reward_tensor + (1 - done) * self.gamma * next_value.squeeze()
        advantage = target_value - current_value.squeeze()

        # Calculate losses
        action_probs, _ = self.actor_critic(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action_tensor)

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
```

### Advantage Actor-Critic (A2C) and Asynchronous Advantage Actor-Critic (A3C)

```python
class A2CAgent:
    def __init__(self, state_dim, action_dim, n_envs=8, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_envs = n_envs

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99

    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0

        # Add bootstrap value for next state if not terminal
        next_values = values[1:] + [0 if done else self.get_value(next_state) for done, next_state in zip(dones, next_states)]

        for i in range(len(rewards)):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.append(gae)

        advantages = torch.tensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def update(self, states, actions, rewards, next_states, dones):
        """Update agent with batch of experiences"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # Get action probabilities and values
        action_probs, values = self.model(states_tensor)
        values = values.squeeze()

        # Compute advantages
        advantages = self.compute_advantages(rewards, values.detach().cpu().numpy(), dones)

        # Calculate losses
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = action_dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

## Deep Deterministic Policy Gradient (DDPG)

For continuous control tasks in robotics:

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 network
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_network(sa)
        return q1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter

        self.memory = deque(maxlen=100000)
        self.batch_size = 100

    def select_action(self, state, add_noise=True, noise_std=0.1):
        """Select action with optional exploration noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self):
        """Update actor and critic networks"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1.squeeze(), target_q) + F.mse_loss(current_q2.squeeze(), target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic.Q1(states, actor_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target_network, source_network):
        """Soft update target network parameters"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Twin Delayed DDPG (TD3)

TD3 addresses overestimation bias in actor-critic methods:

```python
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)

        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2  # Update policy every 2 critic updates

        self.total_it = 0

    def select_action(self, state, add_noise=True, noise_std=0.1):
        """Select action with optional exploration noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self):
        """Update TD3 networks"""
        if len(self.memory) < self.batch_size:
            return

        self.total_it += 1

        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(actions).data.normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values
        target_q1, target_q2 = self.critic_1_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q.detach()

        # Get current Q-values
        current_q1, current_q2 = self.critic_1(states, actions), self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1.Q1(states, self.actor(states)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_1_target, self.critic_1)
            self.soft_update(self.critic_2_target, self.critic_2)

    def soft_update(self, target_network, source_network):
        """Soft update target network parameters"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Soft Actor-Critic (SAC) for Robotics

SAC is particularly effective for robotics applications:

```python
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(SACActor, self).__init__()

        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Initialize log_std to be between reasonable bounds
        self.log_std_layer.weight.data.uniform_(-1e-3, 1e-3)
        self.log_std_layer.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        x = self.network(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()

        # Q1 network
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_network(sa)
        return q1

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, lr_alpha=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SACActor(state_dim, action_dim, max_action).to(self.device)
        self.critic_1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_1_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_2 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_2_target = SACCritic(state_dim, action_dim).to(self.device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Entropy temperature parameter
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.gamma = 0.99
        self.tau = 0.005

        self.memory = deque(maxlen=100000)
        self.batch_size = 256

    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.max_action
        else:
            action, _ = self.actor.sample(state_tensor)
            action = action.cpu().numpy()[0]

        return action

    def update(self):
        """Update SAC networks"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_1_target(next_states, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        # Update critics
        current_q1, current_q2 = self.critic_1(states, actions)
        critic_1_loss = F.mse_loss(current_q1.squeeze(), target_q)
        critic_2_loss = F.mse_loss(current_q2.squeeze(), target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update actor
        pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic_1(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy temperature)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self.soft_update(self.critic_1_target, self.critic_1)
        self.soft_update(self.critic_2_target, self.critic_2)

    def soft_update(self, target_network, source_network):
        """Soft update target network parameters"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Practical Robotics Applications

### Manipulation Tasks with DRL

```python
class RoboticManipulationEnvironment:
    def __init__(self):
        # Initialize robot simulation
        self.robot = self.initialize_robot()
        self.object = self.initialize_object()
        self.workspace_bounds = [[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.8]]  # x, y, z bounds
        self.gripper_open_width = 0.08  # meters
        self.gripper_closed_width = 0.01

    def reset(self):
        """Reset environment to initial state"""
        # Reset robot to home position
        self.robot.reset_to_home()

        # Reset object to random position
        self.object.reset_to_random_position(self.workspace_bounds)

        # Open gripper
        self.robot.set_gripper_width(self.gripper_open_width)

        return self.get_state()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Parse action (typically 4-7 DOF: 3 pos, 1-4 orientation, gripper width)
        ee_pos_delta = action[:3]  # End-effector position delta
        ee_orient_delta = action[3:7] if len(action) > 6 else [0, 0, 0, 1]  # Orientation delta (quaternion)
        gripper_width = action[7] if len(action) > 7 else self.get_current_gripper_width()

        # Apply action to robot
        current_ee_pos = self.robot.get_end_effector_position()
        current_ee_orient = self.robot.get_end_effector_orientation()

        target_pos = current_ee_pos + ee_pos_delta
        target_orient = self.integrate_orientation(current_ee_orient, ee_orient_delta)
        target_gripper = np.clip(gripper_width, self.gripper_closed_width, self.gripper_open_width)

        self.robot.move_to_position_orientation(target_pos, target_orient)
        self.robot.set_gripper_width(target_gripper)

        # Step simulation
        self.robot.step_simulation()

        # Get next state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_manipulation_reward(next_state, action)

        # Check termination
        done = self.is_episode_terminated()
        info = self.get_episode_info()

        return next_state, reward, done, info

    def get_state(self):
        """Get current state vector"""
        state = np.concatenate([
            self.robot.get_joint_positions(),           # Joint positions
            self.robot.get_joint_velocities(),          # Joint velocities
            self.robot.get_end_effector_position(),     # EE position
            self.robot.get_end_effector_orientation(),  # EE orientation
            self.robot.get_gripper_width(),             # Gripper width
            self.object.get_position(),                 # Object position
            self.object.get_orientation(),              # Object orientation
            self.get_relative_object_ee_pose()          # Relative pose
        ])
        return state

    def calculate_manipulation_reward(self, state, action):
        """Calculate reward for manipulation task"""
        ee_pos = state[6:9]  # End-effector position (assuming it's at index 6-8)
        obj_pos = state[13:16]  # Object position (assuming it's at index 13-15)

        # Distance-based reward
        dist_to_object = np.linalg.norm(ee_pos - obj_pos)
        distance_reward = -dist_to_object  # Negative distance encourages approach

        # Grasping reward
        grasp_success = self.check_grasp_success()
        grasp_reward = 100.0 if grasp_success else 0.0

        # Object lifting reward
        object_height = obj_pos[2]  # z-coordinate
        lift_reward = max(0, object_height - 0.1) * 50  # Reward for lifting above 10cm

        # Smoothness penalty
        action_smoothness = -0.01 * np.sum(np.abs(action))

        # Collision penalty
        collision_penalty = -10.0 if self.robot.has_collision() else 0.0

        total_reward = distance_reward + grasp_reward + lift_reward + action_smoothness + collision_penalty
        return total_reward

    def check_grasp_success(self):
        """Check if object is successfully grasped"""
        # Check if object is within gripper and not moving relative to gripper
        gripper_pos = self.robot.get_gripper_position()
        object_pos = self.object.get_position()
        gripper_width = self.robot.get_gripper_width()

        # Check if object is between fingers
        dist_to_object = np.linalg.norm(gripper_pos - object_pos)
        is_close = dist_to_object < 0.05  # 5cm threshold

        # Check if gripper is closed around object
        is_grasped = (gripper_width < 0.03) and is_close  # Gripper is closed and object is nearby

        return is_grasped

class ManipulationDRLAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        # Use SAC for continuous manipulation control
        self.agent = SACAgent(state_dim, action_dim, max_action)

        # For exploration in manipulation tasks
        self.exploration_noise = 0.1

    def train_manipulation_policy(self, env, num_episodes=10000):
        """Train manipulation policy using SAC"""
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Add exploration noise for manipulation
                action = self.agent.select_action(state)
                if episode < 1000:  # More exploration in early episodes
                    action += np.random.normal(0, self.exploration_noise, size=action.shape)
                    action = np.clip(action, -self.agent.max_action, self.agent.max_action)

                next_state, reward, done, info = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                # Update agent
                self.agent.update()

            episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        return episode_rewards
```

### Navigation with DRL

```python
class NavigationEnvironment:
    def __init__(self, map_size=10.0, resolution=0.1):
        self.map_size = map_size
        self.resolution = resolution
        self.map_grid = np.zeros((int(map_size/resolution), int(map_size/resolution)))

        # Robot properties
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_heading = 0.0  # Radians
        self.robot_radius = 0.2   # Robot radius for collision checking

        # Goal and obstacles
        self.goal_pos = np.array([map_size/2, map_size/2])
        self.obstacles = []

    def reset(self):
        """Reset navigation environment"""
        # Randomize robot and goal positions
        self.robot_pos = np.random.uniform(0, self.map_size, size=2)
        self.goal_pos = np.random.uniform(0, self.map_size, size=2)

        # Ensure robot and goal are not too close to obstacles
        while self.is_in_collision(self.robot_pos) or self.is_in_collision(self.goal_pos):
            self.robot_pos = np.random.uniform(0, self.map_size, size=2)
            self.goal_pos = np.random.uniform(0, self.map_size, size=2)

        return self.get_state()

    def get_state(self):
        """Get navigation state"""
        # State includes:
        # - Robot position (2D)
        # - Robot heading (1)
        # - Goal relative position (2D)
        # - Local map around robot (e.g., 21x21 grid = 441 values)
        # - Previous action (2D: linear, angular velocity)

        relative_goal = self.goal_pos - self.robot_pos

        # Get local map around robot
        local_map = self.get_local_map_around_robot()

        # Previous action (initialize to zeros for first step)
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros(2)

        state = np.concatenate([
            self.robot_pos,
            [self.robot_heading],
            relative_goal,
            local_map.flatten(),
            self.prev_action
        ])

        return state

    def get_local_map_around_robot(self, local_size=21):
        """Get local occupancy grid around robot"""
        robot_grid_x = int(self.robot_pos[0] / self.resolution)
        robot_grid_y = int(self.robot_pos[1] / self.resolution)

        half_size = local_size // 2
        local_map = np.zeros((local_size, local_size))

        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                x_idx = robot_grid_x + dx
                y_idx = robot_grid_y + dy

                if (0 <= x_idx < self.map_grid.shape[0] and
                    0 <= y_idx < self.map_grid.shape[1]):
                    local_map[dx + half_size, dy + half_size] = self.map_grid[x_idx, y_idx]
                else:
                    # Outside map boundary - treat as occupied
                    local_map[dx + half_size, dy + half_size] = 1.0

        return local_map

    def step(self, action):
        """Execute navigation action"""
        # Action: [linear_velocity, angular_velocity]
        linear_vel, angular_vel = action

        # Update robot state
        dt = 0.1  # Time step
        self.robot_heading += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_heading) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_heading) * dt

        # Check for collisions
        collision = self.is_in_collision(self.robot_pos)

        # Check if goal reached
        goal_reached = np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5  # 50cm threshold

        # Calculate reward
        reward = self.calculate_navigation_reward(linear_vel, angular_vel, collision, goal_reached)

        # Update previous action
        self.prev_action = np.array([linear_vel, angular_vel])

        done = collision or goal_reached

        return self.get_state(), reward, done, {}

    def calculate_navigation_reward(self, linear_vel, angular_vel, collision, goal_reached):
        """Calculate navigation reward"""
        # Distance to goal reward
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        distance_reward = -dist_to_goal * 0.1

        # Progress reward (positive if getting closer to goal)
        if hasattr(self, 'prev_dist_to_goal'):
            progress = self.prev_dist_to_goal - dist_to_goal
            progress_reward = max(0, progress) * 10
        else:
            progress_reward = 0
        self.prev_dist_to_goal = dist_to_goal

        # Goal reached bonus
        goal_reward = 100.0 if goal_reached else 0.0

        # Collision penalty
        collision_penalty = -100.0 if collision else 0.0

        # Smooth movement reward
        smoothness_reward = -0.01 * (abs(linear_vel) + abs(angular_vel))

        # Velocity efficiency (encourage forward movement toward goal)
        goal_direction = (self.goal_pos - self.robot_pos) / (dist_to_goal + 1e-6)
        robot_direction = np.array([np.cos(self.robot_heading), np.sin(self.robot_heading)])
        alignment = np.dot(goal_direction, robot_direction)
        velocity_reward = max(0, alignment) * linear_vel * 0.1

        total_reward = (distance_reward + progress_reward + goal_reward +
                       collision_penalty + smoothness_reward + velocity_reward)

        return total_reward

    def is_in_collision(self, pos):
        """Check if position is in collision with obstacles"""
        grid_x = int(pos[0] / self.resolution)
        grid_y = int(pos[1] / self.resolution)

        if (0 <= grid_x < self.map_grid.shape[0] and
            0 <= grid_y < self.map_grid.shape[1]):
            return self.map_grid[grid_x, grid_y] > 0.5
        else:
            return True  # Outside bounds is considered collision

class NavigationDRLAgent:
    def __init__(self, state_dim, action_dim, max_action=np.array([1.0, 1.0])):
        # Use TD3 for navigation (continuous action space)
        self.agent = TD3Agent(state_dim, action_dim, max_action)

    def train_navigation_policy(self, env, num_episodes=5000):
        """Train navigation policy using TD3"""
        episode_rewards = []
        success_count = 0

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)

                # Add exploration noise during training
                if episode < 1000:
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -self.agent.max_action, self.agent.max_action)

                next_state, reward, done, info = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                # Update agent
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.update()

            episode_rewards.append(episode_reward)

            if reward > 90:  # Rough success criterion
                success_count += 1

            if episode % 500 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = success_count / max(1, len(episode_rewards[-100:]))
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")

        return episode_rewards, success_count / num_episodes
```

## Multi-Agent Deep RL for Robotics

### Cooperative Multi-Agent Systems

```python
import torch.nn as nn
import torch.nn.functional as F

class MultiAgentDQN(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=256):
        super(MultiAgentDQN, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Individual agent networks
        self.agent_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(n_agents)
        ])

    def forward(self, states):
        """
        states: tensor of shape (batch_size, n_agents, state_dim)
        returns: tensor of shape (batch_size, n_agents, action_dim)
        """
        batch_size, n_agents, state_dim = states.shape

        # Reshape for processing
        states_flat = states.view(-1, state_dim)
        features = self.feature_extractor(states_flat)

        # Process each agent
        q_values = []
        for i, agent_net in enumerate(self.agent_networks):
            agent_states = features[i::self.n_agents]  # Extract features for agent i
            agent_q_values = agent_net(agent_states)
            q_values.append(agent_q_values)

        # Stack and reshape
        q_values = torch.stack(q_values, dim=1)  # (batch_size, n_agents, action_dim)
        return q_values

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient"""
    def __init__(self, state_dim, action_dim, n_agents, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.max_action = max_action

        # Actor networks (individual for each agent)
        self.actors = nn.ModuleList([
            Actor(state_dim, action_dim, max_action).to(self.device)
            for _ in range(n_agents)
        ])
        self.actor_targets = nn.ModuleList([
            Actor(state_dim, action_dim, max_action).to(self.device)
            for _ in range(n_agents)
        ])

        # Critic network (centralized, takes all states and actions)
        self.critic = Critic(state_dim * n_agents, action_dim * n_agents).to(self.device)
        self.critic_target = Critic(state_dim * n_agents, action_dim * n_agents).to(self.device)

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=lr_actor)
            for i in range(n_agents)
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize target networks
        for i in range(n_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.gamma = 0.99
        self.tau = 0.005

    def select_actions(self, states, add_noise=True, noise_std=0.1):
        """Select actions for all agents"""
        actions = []
        states_tensor = torch.FloatTensor(states).to(self.device)

        for i in range(self.n_agents):
            with torch.no_grad():
                agent_state = states_tensor[i]
                action = self.actors[i](agent_state.unsqueeze(0)).cpu().numpy()[0]

                if add_noise:
                    noise = np.random.normal(0, noise_std, size=action.shape)
                    action = np.clip(action + noise, -self.max_action, self.max_action)

                actions.append(action)

        return actions

    def update(self, states, actions, rewards, next_states, dones):
        """Update MADDPG networks"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)

        # Flatten all states and actions for centralized critic
        all_states = states_tensor.view(1, -1)  # (1, n_agents * state_dim)
        all_actions = actions_tensor.view(1, -1)  # (1, n_agents * action_dim)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                next_action = self.actor_targets[i](next_states_tensor[i].unsqueeze(0))
                next_actions.append(next_action)

            next_actions_flat = torch.cat(next_actions, dim=1)  # (1, n_agents * action_dim)
            next_states_flat = next_states_tensor.view(1, -1)  # (1, n_agents * state_dim)

            next_q1, next_q2 = self.critic_target(next_states_flat, next_actions_flat)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards_tensor.sum() + (1 - dones_tensor.all()) * self.gamma * next_q.squeeze()

        # Compute current Q-value
        current_q1, current_q2 = self.critic(all_states, all_actions)
        critic_loss = F.mse_loss(current_q1.squeeze(), target_q) + F.mse_loss(current_q2.squeeze(), target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actors
        for i in range(self.n_agents):
            # Recompute actions for this agent while keeping others fixed
            current_actions = []
            for j in range(self.n_agents):
                if j == i:
                    agent_action = self.actors[j](states_tensor[j].unsqueeze(0))
                else:
                    agent_action = actions_tensor[j].unsqueeze(0)
                current_actions.append(agent_action)

            all_agent_actions = torch.cat(current_actions, dim=1)

            # Actor loss (maximize Q-value)
            actor_loss = -self.critic.Q1(all_states, all_agent_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update target networks
        for i in range(self.n_agents):
            self.soft_update(self.actor_targets[i], self.actors[i])
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target_network, source_network):
        """Soft update target network parameters"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Implementation Considerations for Robotics

### Handling Continuous Action Spaces

```python
class ContinuousActionHandler:
    def __init__(self, action_bounds):
        """
        action_bounds: list of tuples [(min1, max1), (min2, max2), ...]
        """
        self.action_bounds = action_bounds
        self.action_dim = len(action_bounds)

    def normalize_actions(self, raw_actions):
        """Normalize actions to [-1, 1] range"""
        normalized = np.zeros_like(raw_actions)
        for i, (min_val, max_val) in enumerate(self.action_bounds):
            mid = (min_val + max_val) / 2
            range_half = (max_val - min_val) / 2
            normalized[i] = (raw_actions[i] - mid) / range_half
        return np.clip(normalized, -1, 1)

    def denormalize_actions(self, normalized_actions):
        """Denormalize actions from [-1, 1] to original range"""
        denormalized = np.zeros_like(normalized_actions)
        for i, (min_val, max_val) in enumerate(self.action_bounds):
            mid = (min_val + max_val) / 2
            range_half = (max_val - min_val) / 2
            denormalized[i] = normalized_actions[i] * range_half + mid
        return denormalized

    def apply_action_constraints(self, actions):
        """Apply action constraints"""
        constrained = np.copy(actions)
        for i, (min_val, max_val) in enumerate(self.action_bounds):
            constrained[i] = np.clip(constrained[i], min_val, max_val)
        return constrained
```

### Safety Considerations

```python
class SafeRLAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.safety_constraints = []
        self.critical_regions = []

    def safe_act(self, state):
        """Select action with safety considerations"""
        # Get proposed action from base agent
        proposed_action = self.base_agent.select_action(state)

        # Check safety constraints
        if self.is_action_safe(proposed_action, state):
            return proposed_action
        else:
            # Fall back to safe action
            safe_action = self.get_safe_fallback_action(state)
            return safe_action

    def is_action_safe(self, action, state):
        """Check if action is safe given current state"""
        # Check joint limits
        if self.would_exceed_joint_limits(action, state):
            return False

        # Check collision constraints
        if self.would_cause_collision(action, state):
            return False

        # Check velocity limits
        if self.would_exceed_velocity_limits(action, state):
            return False

        return True

    def would_exceed_joint_limits(self, action, state):
        """Check if action would cause joint limit violation"""
        current_joints = state[:7]  # Assuming first 7 are joint positions
        next_joints = current_joints + action[:7] * 0.1  # Assuming 0.1s time step

        joint_limits = self.get_joint_limits()
        for j, (pos, limit) in enumerate(zip(next_joints, joint_limits)):
            if pos < limit[0] or pos > limit[1]:
                return True

        return False

    def get_safe_fallback_action(self, state):
        """Get a safe fallback action"""
        # For navigation: stop the robot
        if self.task_type == "navigation":
            return np.array([0.0, 0.0])  # Stop linear and angular motion

        # For manipulation: move slowly toward home position
        elif self.task_type == "manipulation":
            current_pos = state[6:9]  # EE position
            home_pos = np.array([0.0, 0.0, 0.5])  # Home position
            direction = (home_pos - current_pos) * 0.01  # Small movement toward home
            return np.concatenate([direction, [0, 0, 0, 1], [0.04]])  # pos, orient, gripper

        # Default: return zeros
        return np.zeros_like(self.base_agent.select_action(state))
```

## Learning Objectives

By the end of this lesson, you should be able to:
- Understand the fundamental deep reinforcement learning algorithms (DQN, policy gradients, actor-critic)
- Implement DQN variants (Double DQN, Dueling DQN) for discrete robotic tasks
- Apply policy gradient methods (REINFORCE, Actor-Critic) to robotics problems
- Use continuous control algorithms (DDPG, TD3, SAC) for robotic manipulation and navigation
- Implement multi-agent DRL for cooperative robotics tasks
- Design appropriate neural network architectures for robotic state and action spaces
- Apply safety considerations when deploying DRL policies on real robots
- Understand the trade-offs between different DRL algorithms for robotics applications
- Implement experience replay and other stabilization techniques for DRL in robotics