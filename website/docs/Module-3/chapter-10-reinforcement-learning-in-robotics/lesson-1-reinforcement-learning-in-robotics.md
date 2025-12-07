---
title: "Reinforcement Learning in Robotics"
description: "Introduction to reinforcement learning for robotics, explaining RL fundamentals (states, actions, rewards, policies) and applications in robotics"
chapter: 10
lesson: 1
module: 3
sidebar_label: "Reinforcement Learning in Robotics"
sidebar_position: 1
tags:
  [
    "Reinforcement Learning",
    "Robotics",
    "RL Fundamentals",
    "States",
    "Actions",
    "Rewards",
    "Policies",
  ]
keywords:
  [
    "reinforcement learning",
    "robotics",
    "RL fundamentals",
    "states",
    "actions",
    "rewards",
    "policies",
    "Q-learning",
    "actor-critic",
    "robot control",
  ]
---

# Reinforcement Learning in Robotics

## Overview

Reinforcement Learning (RL) is a powerful machine learning paradigm that has revolutionized robotics by enabling robots to learn complex behaviors through interaction with their environment. Unlike supervised learning which requires labeled examples, or unsupervised learning which finds patterns in unlabeled data, RL learns through trial and error by receiving rewards or penalties for actions taken. This lesson introduces the fundamental concepts of reinforcement learning and explores how these principles are applied to robotic systems.

## Fundamentals of Reinforcement Learning

### The RL Framework

Reinforcement Learning is based on the interaction between an agent and its environment. The core components of an RL system are:

- **Agent**: The learner/controller (in robotics, typically the robot itself)
- **Environment**: The world in which the agent operates
- **State (s)**: The current situation of the environment
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback from the environment
- **Policy (π)**: Strategy for selecting actions
- **Value Function (V)**: Expected return from a state
- **Model**: Agent's representation of the environment

The RL process follows this cycle:

1. Agent observes state s_t
2. Agent selects action a_t based on policy π
3. Agent executes action a_t
4. Environment transitions to state s\_{t+1}
5. Environment provides reward r\_{t+1}
6. Repeat

### Markov Decision Processes (MDPs)

Most RL problems in robotics can be formulated as Markov Decision Processes, where the next state depends only on the current state and action, not on the history of previous states:

```
P(s\_{t+1}|s\_t, a\_t) = P(s\_{t+1}|s\_0, a\_0, ..., s\_t, a\_t)
```

This Markov property is crucial for efficient learning algorithms.

## Core RL Components in Robotics

### State Space

In robotics, the state space typically includes:

```python
class RobotState:
    def __init__(self):
        # Joint positions and velocities
        self.joint_positions = []      # e.g., [0.1, -0.5, 1.2, ...]
        self.joint_velocities = []     # e.g., [0.01, -0.02, 0.05, ...]

        # End-effector pose and velocity
        self.ee_position = [0.0, 0.0, 0.0]    # Cartesian position
        self.ee_orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion
        self.ee_velocity = [0.0, 0.0, 0.0]    # Linear velocity

        # Sensor readings
        self.camera_image = None       # RGB or depth image
        self.lidar_scan = []           # Distance measurements
        self.force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6-axis F/T sensor

        # Task-specific information
        self.target_position = [0.5, 0.5, 0.1]  # Where to go
        self.object_pose = [0.3, 0.4, 0.05, 0.0, 0.0, 0.0, 1.0]  # Object location and orientation
```

### Action Space

Robot actions can be discrete or continuous:

```python
# Discrete actions (for simpler tasks)
discrete_actions = [
    "move_forward",
    "turn_left",
    "turn_right",
    "move_backward",
    "stop"
]

# Continuous actions (for precise control)
class ContinuousActionSpace:
    def __init__(self):
        # Joint position control
        self.joint_positions = np.zeros(7)  # For 7-DOF manipulator

        # Cartesian velocity control
        self.linear_velocity = np.zeros(3)  # [vx, vy, vz]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]

        # Torque control
        self.joint_torques = np.zeros(7)

        # Gripper control
        self.gripper_position = 0.0  # [0.0 (closed), 1.0 (open)]
```

### Reward Functions

Designing appropriate reward functions is critical for successful RL in robotics:

```python
def navigation_reward(current_state, action, next_state, goal_position):
    """Reward function for navigation task"""
    current_pos = current_state['position']
    next_pos = next_state['position']

    # Distance to goal reward
    current_dist = np.linalg.norm(current_pos - goal_position)
    next_dist = np.linalg.norm(next_pos - goal_position)

    # Progress toward goal (positive if getting closer)
    progress_reward = current_dist - next_dist

    # Collision penalty
    collision_penalty = -10.0 if next_state['collision'] else 0.0

    # Goal reached bonus
    goal_bonus = 100.0 if next_dist < 0.1 else 0.0  # 10cm threshold

    # Energy efficiency (negative of action magnitude)
    energy_penalty = -0.01 * np.sum(np.abs(action))

    total_reward = progress_reward + collision_penalty + goal_bonus + energy_penalty
    return total_reward

def manipulation_reward(current_state, action, next_state, target_object):
    """Reward function for manipulation task"""
    ee_pos = next_state['end_effector_position']
    obj_pos = next_state['object_position']

    # Distance to object
    dist_to_obj = np.linalg.norm(ee_pos - obj_pos)

    # Object lifting reward
    obj_height = obj_pos[2]  # z-coordinate
    lift_reward = max(0, obj_height - 0.1) * 10  # Reward for lifting above 10cm

    # Grasping reward (based on contact sensors)
    grasp_reward = 10.0 if next_state['grasp_success'] else 0.0

    # Stability penalty (penalize excessive movement)
    stability_penalty = -0.1 * np.sum(np.abs(next_state['joint_velocities']))

    total_reward = (-dist_to_obj * 0.1) + lift_reward + grasp_reward + stability_penalty
    return total_reward
```

## Classic RL Algorithms

### Q-Learning

Q-Learning is a model-free RL algorithm that learns the value of state-action pairs:

```python
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table (for discrete states) or function approximator
        # For continuous states, we'd use neural networks instead
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Explore: random action
            return np.random.choice(self.action_size)

        # Exploit: best known action
        state_idx = self.discretize_state(state)
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-value using Bellman equation"""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)

        # Q-learning update rule
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        # Update Q-value
        self.q_table[state_idx, action] += self.lr * (target - self.q_table[state_idx, action])

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def discretize_state(self, continuous_state):
        """Convert continuous state to discrete index (simplified)"""
        # This is a simplified example - in practice, you'd have more sophisticated discretization
        # For continuous state spaces, use function approximation (neural networks)
        return hash(tuple(continuous_state)) % self.state_size
```

### Deep Q-Networks (DQN)

For complex robotic tasks with high-dimensional state spaces, Deep Q-Networks use neural networks to approximate the Q-function:

```python
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 1000

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.update_count = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
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

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

## Policy Gradient Methods

### REINFORCE Algorithm

Policy gradient methods directly optimize the policy rather than learning a value function:

```python
class PolicyGradientAgent:
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

    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate loss
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
```

### Actor-Critic Methods

Actor-critic methods combine policy-based and value-based approaches:

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
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ac_network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)

        self.gamma = 0.99

    def act(self, state):
        """Sample action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.ac_network(state_tensor)
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
        current_action_probs, current_value = self.ac_network(state_tensor)
        _, next_value = self.ac_network(next_state_tensor)

        # Calculate advantage
        target_value = reward_tensor + (1 - done) * self.gamma * next_value.squeeze()
        advantage = target_value - current_value.squeeze()

        # Calculate losses
        action_dist = torch.distributions.Categorical(current_action_probs)
        log_prob = action_dist.log_prob(action_tensor)

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        # Update networks
        self.optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        self.optimizer.step()
```

## Robotics-Specific RL Applications

### Robot Control with Deep RL

Applying deep RL to continuous control tasks in robotics:

```python
class RobotControlEnvironment:
    def __init__(self):
        # Initialize robot simulation
        self.robot = self.initialize_robot()
        self.max_steps = 1000
        self.current_step = 0

    def reset(self):
        """Reset environment to initial state"""
        self.robot.reset()
        self.current_step = 0
        return self.get_state()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action to robot
        self.robot.apply_action(action)

        # Step simulation
        self.robot.step_simulation()

        # Get next state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward(action)

        # Check if episode is done
        done = self.is_terminal_state() or self.current_step >= self.max_steps
        self.current_step += 1

        return next_state, reward, done, {}

    def get_state(self):
        """Get current state of the robot"""
        state = np.concatenate([
            self.robot.get_joint_positions(),
            self.robot.get_joint_velocities(),
            self.robot.get_end_effector_position(),
            self.robot.get_end_effector_orientation(),
            self.robot.get_force_torque_sensors()
        ])
        return state

    def calculate_reward(self, action):
        """Calculate reward based on current state and action"""
        # Example: Reaching task
        ee_pos = self.robot.get_end_effector_position()
        target_pos = self.get_target_position()

        distance = np.linalg.norm(ee_pos - target_pos)

        # Dense reward for progress
        reward = -distance  # Negative distance encourages moving toward target

        # Bonus for getting close
        if distance < 0.05:  # 5cm threshold
            reward += 10.0

        # Penalty for excessive joint velocities
        joint_velocities = self.robot.get_joint_velocities()
        velocity_penalty = -0.01 * np.sum(np.abs(joint_velocities))

        # Penalty for high action magnitudes (energy efficiency)
        action_penalty = -0.001 * np.sum(np.abs(action))

        total_reward = reward + velocity_penalty + action_penalty
        return total_reward

    def is_terminal_state(self):
        """Check if current state is terminal"""
        ee_pos = self.robot.get_end_effector_position()
        target_pos = self.get_target_position()

        distance = np.linalg.norm(ee_pos - target_pos)

        # Episode ends if robot reaches target or collides
        return (distance < 0.05) or self.robot.has_collided()

class ContinuousControlAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor network for continuous action space
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        ).to(self.device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        # Target networks for stability
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        self.noise_std = 0.1  # Exploration noise

    def select_action(self, state, add_noise=True):
        """Select action with optional exploration noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)

        if add_noise:
            noise = torch.randn_like(action) * self.noise_std
            action = torch.clamp(action + noise, -1, 1)

        return action.cpu().numpy()[0]

    def update_networks(self, replay_buffer, batch_size=100):
        """Update actor and critic networks using DDPG algorithm"""
        if len(replay_buffer) < batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_critic, self.critic, self.tau)
        self.soft_update(self.target_actor, self.actor, self.tau)

    def soft_update(self, target_network, source_network, tau):
        """Soft update target network parameters"""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Multi-Agent RL for Robotics

```python
class MultiRobotEnvironment:
    def __init__(self, num_robots=2):
        self.num_robots = num_robots
        self.robots = [self.initialize_robot(i) for i in range(num_robots)]
        self.shared_environment = self.initialize_environment()

    def reset(self):
        """Reset multi-robot environment"""
        for robot in self.robots:
            robot.reset()
        return [robot.get_state() for robot in self.robots]

    def step(self, actions):
        """Execute actions for all robots"""
        next_states = []
        rewards = []
        dones = []
        infos = []

        # Execute actions simultaneously
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            robot.apply_action(action)

        # Step simulation
        self.shared_environment.step()

        # Get results for each robot
        for robot in self.robots:
            next_states.append(robot.get_state())
            rewards.append(self.calculate_cooperative_reward(robot))
            dones.append(robot.is_terminal())
            infos.append({})

        return next_states, rewards, dones, infos

    def calculate_cooperative_reward(self, robot):
        """Calculate reward considering cooperation among robots"""
        robot_pos = robot.get_position()

        # Individual task completion
        task_reward = self.get_task_completion_reward(robot)

        # Collision avoidance with other robots
        collision_penalty = 0
        for other_robot in self.robots:
            if other_robot != robot:
                dist = np.linalg.norm(robot_pos - other_robot.get_position())
                if dist < 0.5:  # 50cm threshold
                    collision_penalty -= 10.0

        # Formation maintenance (if applicable)
        formation_reward = self.get_formation_reward(robot)

        total_reward = task_reward + collision_penalty + formation_reward
        return total_reward

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient"""
    def __init__(self, agent_id, state_dim, action_dim, num_agents):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Each agent has its own actor but shares critic information
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def build_actor(self):
        """Build actor network for this agent"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        ).to(self.device)

    def build_critic(self):
        """Build critic network that takes state and action of all agents"""
        # Input: concatenated states of all agents + concatenated actions of all agents
        total_state_dim = self.state_dim * self.num_agents
        total_action_dim = self.action_dim * self.num_agents

        return nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

    def select_action(self, state, add_noise=True):
        """Select action for this agent"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)

        if add_noise:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)

        return action.cpu().numpy()[0]

    def update(self, experiences, other_agents):
        """Update this agent's networks using experiences from all agents"""
        states = torch.FloatTensor(experiences['states']).to(self.device)
        actions = torch.FloatTensor(experiences['actions']).to(self.device)
        rewards = torch.FloatTensor(experiences['rewards'][:, self.agent_id]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(experiences['next_states']).to(self.device)
        dones = torch.BoolTensor(experiences['dones'][:, self.agent_id]).unsqueeze(1).to(self.device)

        # Get next actions from target actors of all agents
        next_actions = []
        for i, agent in enumerate(other_agents):
            if i == self.agent_id:
                next_action = self.target_actor(next_states[:, i, :])
            else:
                next_action = agent.target_actor(next_states[:, i, :])
            next_actions.append(next_action)

        next_actions = torch.cat(next_actions, dim=1)

        # Calculate target Q-value
        with torch.no_grad():
            next_q_values = self.target_critic(
                torch.cat([next_states.view(next_states.size(0), -1),
                          next_actions], dim=1)
            )
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values

        # Calculate current Q-value
        current_actions = actions.clone()
        current_actions[:, self.agent_id, :] = self.actor(states[:, self.agent_id, :])
        current_q_values = self.critic(
            torch.cat([states.view(states.size(0), -1),
                      current_actions.view(current_actions.size(0), -1)], dim=1)
        )

        # Update critic
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(states[:, self.agent_id, :])
        temp_actions = actions.clone()
        temp_actions[:, self.agent_id, :] = predicted_action
        actor_loss = -self.critic(
            torch.cat([states.view(states.size(0), -1),
                      temp_actions.view(temp_actions.size(0), -1)], dim=1)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

## Learning Objectives

By the end of this lesson, you should be able to:

- Understand the fundamental concepts of reinforcement learning and its application to robotics
- Identify the key components of RL systems (states, actions, rewards, policies)
- Implement basic RL algorithms like Q-Learning and Deep Q-Networks
- Apply policy gradient methods and actor-critic algorithms to robotic tasks
- Design appropriate reward functions for robotic control problems
- Understand the challenges of applying RL to real-world robotic systems
- Recognize the differences between discrete and continuous action spaces in robotics
- Evaluate the potential of RL for solving complex robotic manipulation and navigation tasks
