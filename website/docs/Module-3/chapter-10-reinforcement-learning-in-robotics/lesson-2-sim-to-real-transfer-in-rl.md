---
title: 'Sim-to-Real Transfer in RL'
description: 'Understanding sim-to-real transfer for reinforcement learning, discussing challenges and techniques like domain randomization and adaptation'
chapter: 10
lesson: 2
module: 3
sidebar_label: 'Sim-to-Real Transfer in RL'
sidebar_position: 2
tags: ['Sim-to-Real', 'Domain Randomization', 'Transfer Learning', 'RL', 'Robotics']
keywords: ['sim-to-real', 'domain randomization', 'transfer learning', 'reinforcement learning', 'robotics', 'domain adaptation', 'simulation']
---

# Sim-to-Real Transfer in RL

## Overview

Sim-to-real transfer in reinforcement learning refers to the process of transferring policies learned in simulation to real robotic systems. This approach is crucial for robotics applications as it allows for safe, cost-effective training in virtual environments before deployment on physical robots. However, the "reality gap" between simulation and real-world environments poses significant challenges that must be addressed through specialized techniques.

## The Reality Gap Problem

### Simulation vs. Reality Differences

The reality gap encompasses all discrepancies between simulated and real environments:

1. **Dynamics Mismatch**: Differences in friction, mass, inertial properties
2. **Sensor Noise**: Simulated sensors often have ideal characteristics compared to real sensors
3. **Actuator Dynamics**: Motor response times, gear backlash, and control delays
4. **Environmental Conditions**: Lighting, temperature, and atmospheric effects
5. **Geometric Imperfections**: Manufacturing tolerances and wear patterns

### Impact on RL Policies

These differences can severely impact policy performance:

```python
class RealityGapAnalyzer:
    def __init__(self):
        self.sim_performance = 0.0
        self.real_performance = 0.0
        self.transfer_efficiency = 0.0

    def calculate_reality_gap(self, sim_metrics, real_metrics):
        """Calculate the reality gap between simulation and real performance"""
        gap_metrics = {}

        # Performance gap
        gap_metrics['performance_gap'] = sim_metrics['success_rate'] - real_metrics['success_rate']

        # Behavioral discrepancy
        gap_metrics['behavior_discrepancy'] = self.calculate_behavior_difference(
            sim_metrics['trajectories'], real_metrics['trajectories']
        )

        # Control stability gap
        gap_metrics['control_stability'] = abs(
            sim_metrics['control_effort'] - real_metrics['control_effort']
        )

        return gap_metrics

    def calculate_behavior_difference(self, sim_traj, real_traj):
        """Calculate behavioral difference between simulation and real trajectories"""
        # Use Dynamic Time Warping or similar method to compare trajectories
        if len(sim_traj) != len(real_traj):
            # Interpolate to same length
            sim_interp = self.interpolate_trajectory(sim_traj, len(real_traj))
        else:
            sim_interp = sim_traj

        # Calculate mean squared difference
        diff = np.array(sim_interp) - np.array(real_traj)
        mse = np.mean(diff ** 2)

        return mse

    def assess_transfer_readiness(self, policy, sim_env, real_env):
        """Assess how ready a policy is for sim-to-real transfer"""
        readiness_score = 0.0

        # Evaluate on simulation (baseline)
        sim_success_rate = self.evaluate_policy(policy, sim_env)

        # Evaluate on simulation with added noise/disturbances
        noisy_sim_success_rate = self.evaluate_policy_with_noise(policy, sim_env)

        # Robustness to simulation variations
        robustness_score = sim_success_rate - noisy_sim_success_rate

        # If policy performs well with noise, it might be robust enough
        if robustness_score < 0.1:  # Arbitrary threshold
            readiness_score += 0.3

        # Check policy variance across multiple runs
        variance_score = self.calculate_policy_variance(policy, sim_env)
        if variance_score < 0.05:  # Low variance indicates consistent behavior
            readiness_score += 0.2

        # Evaluate generalization capability
        generalization_score = self.test_generalization(policy, sim_env)
        readiness_score += generalization_score * 0.5

        return min(readiness_score, 1.0)  # Clamp to [0, 1]
```

## Domain Randomization Techniques

### Basic Domain Randomization

Domain randomization involves randomizing simulation parameters to make policies robust:

```python
class DomainRandomizer:
    def __init__(self):
        self.randomization_ranges = {
            'friction': (0.1, 1.0),
            'mass': (0.8, 1.2),
            'restitution': (0.0, 0.5),
            'gravity': (9.5, 10.1),
            'sensor_noise': (0.0, 0.05),
            'actuator_delay': (0.0, 0.02),
            'lighting': (0.5, 2.0)
        }

        self.current_params = {}
        self.reset()

    def reset(self):
        """Reset to random domain parameters"""
        self.current_params = {}
        for param, (min_val, max_val) in self.randomization_ranges.items():
            self.current_params[param] = random.uniform(min_val, max_val)

    def randomize_environment(self, env):
        """Apply randomization to environment"""
        # Randomize physical properties
        env.set_friction(self.current_params['friction'])
        env.set_mass_multiplier(self.current_params['mass'])
        env.set_restitution(self.current_params['restitution'])
        env.set_gravity(self.current_params['gravity'])

        # Randomize sensor properties
        env.set_sensor_noise_level(self.current_params['sensor_noise'])
        env.set_actuator_delay(self.current_params['actuator_delay'])

        # Randomize visual properties
        env.set_lighting_condition(self.current_params['lighting'])

        return env

    def update_randomization_ranges(self, real_world_data):
        """Update randomization ranges based on real-world observations"""
        # This would analyze real-world data to refine randomization bounds
        # For example, if we observe that friction in real world is between 0.3 and 0.7,
        # we'd update the friction range accordingly
        pass

    def randomize_episode(self, env):
        """Randomize environment for new episode"""
        self.reset()
        return self.randomize_environment(env)
```

### Advanced Domain Randomization

```python
class AdvancedDomainRandomizer:
    def __init__(self):
        self.param_groups = {
            'dynamics': ['friction', 'mass', 'restitution', 'gravity'],
            'sensors': ['noise_level', 'delay', 'bias'],
            'visual': ['lighting', 'texture', 'camera_params']
        }

        # Correlation matrix for related parameters
        self.correlation_matrix = self.initialize_correlation_matrix()
        self.randomization_schedule = self.create_adaptive_schedule()

    def initialize_correlation_matrix(self):
        """Initialize parameter correlation matrix"""
        # In practice, this would be learned from data
        # For now, we'll create a simple example
        n_params = len(self.get_all_param_names())
        corr_matrix = np.eye(n_params)

        # Add some correlations between related parameters
        # Example: friction and restitution might be somewhat correlated
        friction_idx = self.get_param_index('friction')
        restitution_idx = self.get_param_index('restitution')
        if friction_idx is not None and restitution_idx is not None:
            corr_matrix[friction_idx, restitution_idx] = 0.3
            corr_matrix[restitution_idx, friction_idx] = 0.3

        return corr_matrix

    def get_all_param_names(self):
        """Get all parameter names"""
        all_names = []
        for group in self.param_groups.values():
            all_names.extend(group)
        return all_names

    def get_param_index(self, param_name):
        """Get index of parameter in correlation matrix"""
        all_names = self.get_all_param_names()
        try:
            return all_names.index(param_name)
        except ValueError:
            return None

    def randomize_with_correlations(self, base_params):
        """Randomize parameters considering correlations"""
        param_names = self.get_all_param_names()
        n_params = len(param_names)

        # Generate correlated random variables using Cholesky decomposition
        L = np.linalg.cholesky(self.correlation_matrix + 1e-6 * np.eye(n_params))  # Add small value for numerical stability
        uncorrelated_vars = np.random.randn(n_params)
        correlated_vars = L @ uncorrelated_vars

        randomized_params = base_params.copy()

        for i, param_name in enumerate(param_names):
            if param_name in randomized_params:
                # Use correlated variable to influence randomization
                base_min, base_max = self.get_param_range(param_name)
                correlation_influence = correlated_vars[i] * 0.1  # Influence factor

                # Adjust range based on correlation
                adjusted_min = max(base_min, base_min + correlation_influence)
                adjusted_max = min(base_max, base_max + correlation_influence)

                randomized_params[param_name] = np.random.uniform(adjusted_min, adjusted_max)

        return randomized_params

    def get_param_range(self, param_name):
        """Get base range for parameter"""
        ranges = {
            'friction': (0.1, 1.0),
            'mass': (0.8, 1.2),
            'restitution': (0.0, 0.5),
            'gravity': (9.5, 10.1),
            'noise_level': (0.0, 0.05),
            'delay': (0.0, 0.02),
            'bias': (-0.01, 0.01),
            'lighting': (0.5, 2.0),
            'texture': (0.0, 1.0),
            'camera_params': (0.9, 1.1)
        }
        return ranges.get(param_name, (0.0, 1.0))

    def adaptive_randomization(self, performance_history):
        """Adapt randomization based on performance feedback"""
        if len(performance_history) < 10:
            return  # Not enough data

        # Calculate recent performance trend
        recent_performance = performance_history[-5:]
        older_performance = performance_history[-10:-5]

        recent_avg = np.mean(recent_performance)
        older_avg = np.mean(older_performance)

        # If performance is degrading, reduce randomization range
        if recent_avg < older_avg * 0.9:  # Performance dropped by 10%
            self.reduce_randomization_range(0.9)
        # If performance is stable/good, increase randomization range
        elif recent_avg > older_avg * 1.05:  # Performance improved by 5%
            self.increase_randomization_range(1.1)

    def reduce_randomization_range(self, factor):
        """Reduce randomization ranges by factor"""
        for param, (min_val, max_val) in self.randomization_ranges.items():
            center = (min_val + max_val) / 2
            range_width = (max_val - min_val) * factor
            new_min = center - range_width / 2
            new_max = center + range_width / 2
            self.randomization_ranges[param] = (new_min, new_max)

    def increase_randomization_range(self, factor):
        """Increase randomization ranges by factor"""
        for param, (min_val, max_val) in self.randomization_ranges.items():
            center = (min_val + max_val) / 2
            range_width = (max_val - min_val) * factor
            new_min = max(0, center - range_width / 2)  # Ensure non-negative
            new_max = center + range_width / 2
            self.randomization_ranges[param] = (new_min, new_max)

class CurriculumDomainRandomizer:
    def __init__(self):
        self.curriculum_levels = [
            {
                'name': 'easy',
                'difficulty': 0.1,
                'param_ranges': {
                    'friction': (0.4, 0.6),
                    'mass': (0.95, 1.05),
                    'restitution': (0.1, 0.2),
                    'gravity': (9.7, 9.9),
                    'sensor_noise': (0.0, 0.01)
                },
                'duration': 1000,  # episodes
                'success_threshold': 0.8
            },
            {
                'name': 'medium',
                'difficulty': 0.3,
                'param_ranges': {
                    'friction': (0.2, 0.8),
                    'mass': (0.9, 1.1),
                    'restitution': (0.05, 0.3),
                    'gravity': (9.6, 10.0),
                    'sensor_noise': (0.0, 0.03)
                },
                'duration': 2000,
                'success_threshold': 0.7
            },
            {
                'name': 'hard',
                'difficulty': 0.6,
                'param_ranges': {
                    'friction': (0.1, 0.9),
                    'mass': (0.8, 1.2),
                    'restitution': (0.0, 0.4),
                    'gravity': (9.5, 10.1),
                    'sensor_noise': (0.0, 0.05)
                },
                'duration': 3000,
                'success_threshold': 0.6
            },
            {
                'name': 'extreme',
                'difficulty': 1.0,
                'param_ranges': {
                    'friction': (0.05, 0.95),
                    'mass': (0.7, 1.3),
                    'restitution': (0.0, 0.5),
                    'gravity': (9.3, 10.3),
                    'sensor_noise': (0.0, 0.1)
                },
                'duration': float('inf'),  # Continue indefinitely
                'success_threshold': 0.5
            }
        ]

        self.current_level = 0
        self.episode_count = 0
        self.performance_history = []

    def get_current_randomization_params(self):
        """Get randomization parameters for current curriculum level"""
        current_ranges = self.curriculum_levels[self.current_level]['param_ranges']

        randomized_params = {}
        for param, (min_val, max_val) in current_ranges.items():
            randomized_params[param] = random.uniform(min_val, max_val)

        return randomized_params

    def update_curriculum(self, episode_performance):
        """Update curriculum level based on performance"""
        self.performance_history.append(episode_performance)
        self.episode_count += 1

        # Check if we should advance to next level
        if (self.episode_count >= self.curriculum_levels[self.current_level]['duration'] and
            self.get_recent_performance() >= self.curriculum_levels[self.current_level]['success_threshold']):

            if self.current_level < len(self.curriculum_levels) - 1:
                self.current_level += 1
                self.episode_count = 0  # Reset counter for new level
                print(f"Advancing to curriculum level: {self.curriculum_levels[self.current_level]['name']}")

    def get_recent_performance(self, window=100):
        """Get average performance over recent episodes"""
        recent = self.performance_history[-window:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def get_curriculum_progress(self):
        """Get current curriculum progress"""
        return {
            'level': self.curriculum_levels[self.current_level]['name'],
            'progress': min(1.0, self.episode_count / self.curriculum_levels[self.current_level]['duration']),
            'performance': self.get_recent_performance(window=50),
            'param_ranges': self.curriculum_levels[self.current_level]['param_ranges']
        }
```

## Domain Adaptation Techniques

### Unsupervised Domain Adaptation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=10):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor - shared between domains
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Task-specific classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lambda_val=0.0):
        features = self.feature_extractor(x)

        # Extract task prediction
        task_output = self.task_classifier(features)

        # Apply gradient reversal for domain adaptation
        reversed_features = GradientReversalFunction.apply(features, lambda_val)
        domain_output = self.domain_classifier(reversed_features)

        return task_output, domain_output, features

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer
    """
    @staticmethod
    def forward(ctx, input, lambda_val):
        ctx.lambda_val = lambda_val
        return input

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.lambda_val
        return -lambda_val * grad_output, None

class DomainAdversarialTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Task-specific loss (e.g., cross-entropy for classification)
        self.task_criterion = nn.CrossEntropyLoss()

        # Domain loss (binary cross-entropy for domain classification)
        self.domain_criterion = nn.BCELoss()

    def train_step(self, sim_data, real_data, sim_labels, lambda_val=1.0):
        """
        Training step for domain adversarial training
        """
        batch_size = sim_data.size(0)

        # Create domain labels (0 for sim, 1 for real)
        sim_domain_labels = torch.zeros(batch_size, 1).to(sim_data.device)
        real_domain_labels = torch.ones(batch_size, 1).to(real_data.device)

        # Concatenate data
        all_data = torch.cat([sim_data, real_data], dim=0)
        all_domain_labels = torch.cat([sim_domain_labels, real_domain_labels], dim=0)

        # Labels for task classifier (only for simulated data with labels)
        all_task_labels = torch.cat([sim_labels, torch.zeros_like(sim_labels)], dim=0)

        self.optimizer.zero_grad()

        # Forward pass
        task_outputs, domain_outputs, features = self.model(all_data, lambda_val)

        # Task classification loss (only on simulated data with labels)
        task_loss = self.task_criterion(
            task_outputs[:batch_size],  # Only simulated data has labels
            sim_labels
        )

        # Domain classification loss
        domain_loss = self.domain_criterion(domain_outputs, all_domain_labels)

        # Total loss: minimize task loss, maximize domain confusion
        total_loss = task_loss - lambda_val * domain_loss

        total_loss.backward()
        self.optimizer.step()

        return task_loss.item(), domain_loss.item(), total_loss.item()
```

### Adversarial Domain Adaptation for Robotics

```python
class RobotDomainAdaptation:
    def __init__(self, robot_model):
        self.sim_model = robot_model  # Simulation model
        self.real_model = None        # Will be initialized with real robot data
        self.adaptation_network = DomainAdaptationNetwork(
            input_dim=robot_model.observation_space.shape[0],
            hidden_dim=256,
            num_classes=robot_model.action_space.n
        )
        self.domain_discriminator = self.create_domain_discriminator()

    def create_domain_discriminator(self):
        """Create domain discriminator network"""
        return nn.Sequential(
            nn.Linear(256, 128),  # Assuming feature dimension of 256
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def adapt_policy(self, policy, real_experience_buffer):
        """
        Adapt policy from simulation to real using domain adaptation
        """
        adaptation_losses = []

        for epoch in range(100):  # Adaptation epochs
            # Sample batch from real experience
            real_batch = real_experience_buffer.sample(batch_size=32)

            # Get corresponding simulation data
            sim_batch = self.generate_similar_data(real_batch)

            # Perform domain adaptation step
            task_loss, domain_loss, total_loss = self.adaptation_step(
                sim_batch, real_batch, lambda_val=self.get_adaptation_lambda(epoch)
            )

            adaptation_losses.append(total_loss)

        return policy, adaptation_losses

    def generate_similar_data(self, real_batch):
        """Generate simulation data similar to real data"""
        # This would involve adjusting simulation parameters
        # to generate data similar to real observations
        sim_data = []

        for real_obs in real_batch.observations:
            # Perturb simulation data to match real characteristics
            perturbed_obs = self.perturb_simulation_observation(real_obs)
            sim_data.append(perturbed_obs)

        return torch.stack(sim_data)

    def perturb_simulation_observation(self, real_obs):
        """Perturb simulation observation to match real characteristics"""
        # Estimate differences between sim and real
        sim_to_real_mapping = self.estimate_sim_to_real_mapping()

        # Apply learned transformation
        adapted_obs = self.apply_mapping(real_obs, sim_to_real_mapping)

        return adapted_obs

    def estimate_sim_to_real_mapping(self):
        """
        Estimate mapping between simulation and real observations
        This could use techniques like:
        - Optimal transport
        - Generative models
        - Regression models
        """
        # Placeholder for mapping estimation
        # In practice, this would use collected sim and real data pairs
        return {}

    def apply_mapping(self, observation, mapping):
        """Apply learned mapping to observation"""
        # Apply transformation based on learned mapping
        # This is a simplified example
        transformed_obs = observation.clone()

        if 'mean_shift' in mapping:
            transformed_obs += mapping['mean_shift']
        if 'scaling' in mapping:
            transformed_obs *= mapping['scaling']

        return transformed_obs

    def get_adaptation_lambda(self, epoch):
        """Get adaptation lambda that changes over time"""
        # Gradually increase domain adaptation strength
        return min(1.0, epoch / 50.0)  # Ramp up from 0 to 1 over 50 epochs
```

## System Identification for Transfer

### Parameter Estimation

```python
from scipy.optimize import minimize
import numpy as np

class SystemIdentifier:
    def __init__(self, robot_model_structure):
        self.model_structure = robot_model_structure
        self.estimated_params = {}
        self.param_bounds = {}

    def identify_parameters(self, input_output_data):
        """
        Identify system parameters from input-output data
        """
        # Define objective function to minimize
        def objective(params):
            # Set model parameters
            self.set_model_parameters(params)

            # Simulate model with inputs
            simulated_outputs = self.simulate_model(
                input_output_data['inputs'],
                input_output_data['initial_states']
            )

            # Calculate error between simulated and real outputs
            error = self.calculate_error(
                simulated_outputs,
                input_output_data['outputs']
            )

            return error

        # Initial parameter guess
        initial_guess = self.get_initial_parameter_guess()

        # Optimize parameters
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=self.param_bounds
        )

        self.estimated_params = result.x
        return self.estimated_params

    def get_initial_parameter_guess(self):
        """Get initial parameter guess"""
        # This would return initial guesses based on nominal values
        return np.array([1.0, 1.0, 0.1, 9.81])  # Example: mass, length, friction, gravity

    def set_model_parameters(self, params):
        """Set model parameters"""
        # Update the robot model with estimated parameters
        pass

    def simulate_model(self, inputs, initial_states):
        """Simulate the robot model"""
        # This would run forward simulation of the model
        outputs = []
        current_state = initial_states[0]

        for input_cmd in inputs:
            # Apply input and step model
            next_state = self.step_model(current_state, input_cmd)
            output = self.get_output_from_state(next_state)
            outputs.append(output)
            current_state = next_state

        return np.array(outputs)

    def calculate_error(self, simulated_outputs, real_outputs):
        """Calculate error between simulated and real outputs"""
        # Mean squared error
        mse = np.mean((simulated_outputs - real_outputs) ** 2)
        return mse

    def system_identification_pipeline(self, real_robot, simulation_env):
        """
        Complete system identification pipeline
        """
        # 1. Collect data from real robot
        real_data = self.collect_real_robot_data(real_robot)

        # 2. Identify system parameters
        estimated_params = self.identify_parameters(real_data)

        # 3. Update simulation with identified parameters
        updated_sim_env = self.update_simulation_parameters(
            simulation_env, estimated_params
        )

        # 4. Validate the updated simulation
        validation_results = self.validate_identification(
            updated_sim_env, real_data
        )

        return updated_sim_env, validation_results

    def collect_real_robot_data(self, robot):
        """Collect input-output data from real robot"""
        data = {
            'inputs': [],
            'outputs': [],
            'timestamps': [],
            'initial_states': []
        }

        # Apply various inputs and record responses
        test_inputs = self.generate_excitation_signals()

        for input_seq in test_inputs:
            initial_state = robot.get_state()

            outputs = []
            for cmd in input_seq:
                robot.apply_command(cmd)
                robot.step_simulation()
                output = robot.get_sensor_readings()

                data['inputs'].append(cmd)
                data['outputs'].append(output)
                data['timestamps'].append(robot.get_time())

            data['initial_states'].append(initial_state)

        return data

    def generate_excitation_signals(self):
        """Generate input signals for system identification"""
        # Generate various excitation signals
        signals = []

        # Step inputs
        for amp in [0.1, 0.2, 0.5]:
            step_signal = [amp if i > 10 else 0 for i in range(50)]
            signals.append(step_signal)

        # Sinusoidal inputs
        for freq in [0.5, 1.0, 2.0]:
            sine_signal = [0.3 * np.sin(2 * np.pi * freq * i * 0.01) for i in range(50)]
            signals.append(sine_signal)

        # Random inputs
        for _ in range(5):
            random_signal = np.random.uniform(-0.3, 0.3, 50).tolist()
            signals.append(random_signal)

        return signals
```

## Transfer Learning Techniques

### Fine-Tuning Approaches

```python
class TransferLearningAgent:
    def __init__(self, pretrained_policy, real_robot_env):
        self.pretrained_policy = pretrained_policy
        self.real_env = real_robot_env
        self.transfer_network = self.create_transfer_network()
        self.adaptation_phase = True

    def create_transfer_network(self):
        """Create network for adapting simulation policy to real robot"""
        # This could be a small network that adapts the policy
        # or a full network that learns the transfer mapping
        return nn.Sequential(
            nn.Linear(self.pretrained_policy.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.pretrained_policy.action_dim)
        )

    def adapt_policy(self, real_experience_buffer, adaptation_steps=1000):
        """
        Adapt pretrained policy to real robot using limited real experience
        """
        optimizer = torch.optim.Adam(
            list(self.pretrained_policy.parameters()) +
            list(self.transfer_network.parameters()),
            lr=1e-4
        )

        for step in range(adaptation_steps):
            # Sample batch from real experience
            batch = real_experience_buffer.sample(batch_size=32)

            # Get actions from pretrained policy
            with torch.no_grad():
                sim_actions = self.pretrained_policy.select_action(batch.states)

            # Adapt actions for real robot
            adapted_actions = self.transfer_network(batch.states)

            # Calculate loss based on real robot feedback
            loss = self.calculate_adaptation_loss(
                sim_actions, adapted_actions, batch.rewards
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Adaptation step {step}, loss: {loss.item():.4f}")

    def calculate_adaptation_loss(self, sim_actions, adapted_actions, rewards):
        """Calculate loss for adaptation"""
        # Use KL divergence between old and new action distributions
        # or direct action difference weighted by rewards
        action_diff = F.mse_loss(adapted_actions, sim_actions.detach())

        # Weight by rewards (prefer actions that lead to higher rewards)
        reward_weighted_loss = action_diff * torch.mean(rewards)

        return reward_weighted_loss

class MetaLearningTransfer:
    def __init__(self, base_learner, meta_lr=0.001):
        self.base_learner = base_learner
        self.meta_learner = copy.deepcopy(base_learner)
        self.meta_optimizer = torch.optim.Adam(meta_learner.parameters(), lr=meta_lr)

    def meta_train(self, tasks, num_inner_updates=5):
        """
        Meta-training phase - learn to adapt quickly to new tasks
        """
        meta_loss = 0

        for task in tasks:
            # Sample support (training) and query (validation) sets
            support_set, query_set = self.split_task_data(task)

            # Adapt base learner to specific task (inner loop)
            adapted_learner = self.adapt_to_task(
                self.meta_learner, support_set, num_inner_updates
            )

            # Evaluate adapted learner on query set
            query_loss = self.evaluate_on_query(adapted_learner, query_set)
            meta_loss += query_loss

        # Update meta-learner (outer loop)
        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_task(self, learner, support_set, num_updates):
        """Adapt learner to specific task using support set"""
        adapted_learner = copy.deepcopy(learner)

        for _ in range(num_updates):
            loss = self.compute_task_loss(adapted_learner, support_set)

            # Compute gradients
            gradients = torch.autograd.grad(loss, adapted_learner.parameters())

            # Update adapted learner parameters
            for param, grad in zip(adapted_learner.parameters(), gradients):
                param.data = param.data - 0.01 * grad  # Inner loop learning rate

        return adapted_learner

    def evaluate_on_query(self, learner, query_set):
        """Evaluate learner on query set"""
        with torch.no_grad():
            loss = self.compute_task_loss(learner, query_set)
        return loss

    def compute_task_loss(self, learner, dataset):
        """Compute loss for task on dataset"""
        states, actions, rewards = dataset
        pred_actions = learner(states)
        loss = F.mse_loss(pred_actions, actions)
        return loss

    def transfer_to_new_task(self, new_task_data):
        """Transfer to new task using meta-knowledge"""
        # Adapt quickly to new task using few examples
        support_set = new_task_data[:10]  # Use first 10 examples
        adapted_policy = self.adapt_to_task(self.meta_learner, support_set, num_updates=5)
        return adapted_policy
```

## Practical Implementation Strategies

### Sim-to-Real Pipeline

```python
class SimToRealPipeline:
    def __init__(self):
        self.domain_randomizer = AdvancedDomainRandomizer()
        self.system_identifier = SystemIdentifier(None)
        self.transfer_learner = TransferLearningAgent(None, None)
        self.validation_system = TransferValidator()

    def execute_transfer_pipeline(self, sim_policy, real_robot):
        """
        Complete sim-to-real transfer pipeline
        """
        print("Starting sim-to-real transfer pipeline...")

        # Phase 1: Simulation training with domain randomization
        print("Phase 1: Training with domain randomization...")
        robust_policy = self.train_with_domain_randomization(sim_policy)

        # Phase 2: System identification
        print("Phase 2: System identification...")
        identified_params = self.system_identifier.identify_parameters(
            self.collect_system_id_data(real_robot)
        )

        # Phase 3: Simulation update with identified parameters
        print("Phase 3: Updating simulation with identified parameters...")
        updated_sim_env = self.update_simulation_with_real_params(identified_params)

        # Phase 4: Fine-tuning with real data
        print("Phase 4: Fine-tuning with real robot data...")
        real_experience_buffer = self.collect_real_experience(real_robot, robust_policy)
        adapted_policy = self.transfer_learner.adapt_policy(real_experience_buffer)

        # Phase 5: Validation and testing
        print("Phase 5: Validation and testing...")
        validation_results = self.validation_system.validate_transfer(
            adapted_policy, real_robot
        )

        print(f"Transfer completed. Success rate: {validation_results['success_rate']:.3f}")
        print(f"Performance gap: {validation_results['performance_gap']:.3f}")

        return adapted_policy, validation_results

    def train_with_domain_randomization(self, base_policy):
        """Train policy with domain randomization"""
        # This would involve training the policy in randomized simulation
        # environments over many episodes
        policy = copy.deepcopy(base_policy)

        for episode in range(10000):  # Training episodes
            # Randomize environment for this episode
            randomized_env = self.domain_randomizer.randomize_episode(self.sim_env)

            # Train policy in randomized environment
            self.train_policy_step(policy, randomized_env)

            # Update domain randomization based on performance
            if episode % 100 == 0:
                self.domain_randomizer.adaptive_randomization(
                    self.get_recent_performance()
                )

        return policy

    def collect_system_id_data(self, real_robot):
        """Collect data for system identification"""
        # Apply various excitation signals to real robot
        # and record input-output behavior
        pass

    def update_simulation_with_real_params(self, identified_params):
        """Update simulation with real-world parameters"""
        # Create new simulation environment with identified parameters
        updated_env = self.sim_env.copy()
        for param_name, param_value in identified_params.items():
            updated_env.set_parameter(param_name, param_value)
        return updated_env

    def collect_real_experience(self, real_robot, policy, num_episodes=100):
        """Collect experience from real robot using policy"""
        experience_buffer = []

        for episode in range(num_episodes):
            state = real_robot.reset()
            done = False

            while not done:
                action = policy.select_action(state)
                next_state, reward, done, info = real_robot.step(action)

                experience_buffer.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })

                state = next_state

        return experience_buffer

class TransferValidator:
    def __init__(self):
        self.metrics = {
            'success_rate': 0.0,
            'task_completion_time': float('inf'),
            'energy_efficiency': 0.0,
            'safety_metrics': {},
            'robustness_score': 0.0
        }

    def validate_transfer(self, policy, real_robot, num_trials=50):
        """Validate transferred policy on real robot"""
        results = {
            'success_rate': 0,
            'avg_completion_time': 0,
            'performance_gap': 0,
            'safety_violations': 0,
            'robustness_score': 0
        }

        successes = 0
        total_time = 0
        safety_violations = 0

        for trial in range(num_trials):
            state = real_robot.reset()
            done = False
            steps = 0
            episode_time = 0

            while not done and steps < 1000:  # Max steps per trial
                start_time = time.time()
                action = policy.select_action(state)
                state, reward, done, info = real_robot.step(action)
                step_time = time.time() - start_time
                episode_time += step_time

                # Check for safety violations
                if self.check_safety_violation(real_robot):
                    safety_violations += 1

                steps += 1

            if done and self.check_task_success(real_robot):
                successes += 1
                total_time += episode_time

        results['success_rate'] = successes / num_trials
        results['avg_completion_time'] = total_time / max(successes, 1)
        results['safety_violations'] = safety_violations / num_trials

        # Calculate robustness by testing with disturbances
        results['robustness_score'] = self.test_robustness(policy, real_robot)

        return results

    def check_task_success(self, robot):
        """Check if task was completed successfully"""
        # This would depend on the specific task
        # Example: check if object was successfully grasped
        return robot.has_completed_task()

    def check_safety_violation(self, robot):
        """Check if safety constraint was violated"""
        # Check for joint limits, collisions, etc.
        return (robot.is_in_collision() or
                robot.are_joints_at_limits() or
                robot.exceeds_force_limits())

    def test_robustness(self, policy, robot, num_tests=20):
        """Test policy robustness with external disturbances"""
        robust_successes = 0

        for test in range(num_tests):
            state = robot.reset()

            # Apply random disturbance during execution
            disturb_step = random.randint(10, 50)

            for step in range(100):
                if step == disturb_step:
                    # Apply disturbance
                    robot.apply_external_disturbance()

                action = policy.select_action(state)
                state, reward, done, info = robot.step(action)

                if done:
                    break

            if done and self.check_task_success(robot):
                robust_successes += 1

        return robust_successes / num_tests
```

## Best Practices for Sim-to-Real Transfer

### 1. Progressive Complexity

Start with simple tasks and gradually increase complexity:

```python
class ProgressiveTransferStrategy:
    def __init__(self):
        self.difficulty_levels = [
            'position_control',
            'trajectory_tracking',
            'simple_manipulation',
            'complex_manipulation',
            'dynamic_tasks'
        ]
        self.current_level = 0

    def transfer_at_level(self, level_idx, sim_policy, real_robot):
        """Transfer policy at specific difficulty level"""
        level = self.difficulty_levels[level_idx]

        if level == 'position_control':
            # Simple position control tasks
            task_config = self.get_position_control_config()
        elif level == 'trajectory_tracking':
            # Following predefined trajectories
            task_config = self.get_trajectory_tracking_config()
        # ... other levels

        # Train with appropriate domain randomization for level
        level_policy = self.train_with_appropriate_randomization(
            sim_policy, task_config
        )

        # Validate on real robot
        success_rate = self.validate_on_real(level_policy, real_robot, task_config)

        return success_rate > 0.8  # Threshold for advancing

    def advance_to_next_level(self, sim_policy, real_robot):
        """Advance to next difficulty level if current level is mastered"""
        if self.transfer_at_level(self.current_level, sim_policy, real_robot):
            self.current_level += 1
            if self.current_level < len(self.difficulty_levels):
                print(f"Advancing to level: {self.difficulty_levels[self.current_level]}")
                return True
        return False
```

### 2. Validation and Safety

Always validate transfers safely:

```python
class SafeTransferValidator:
    def __init__(self):
        self.safety_checks = [
            self.check_joint_limits,
            self.check_collision_risk,
            self.check_force_limits,
            self.check_stability
        ]

    def safe_transfer_test(self, policy, real_robot):
        """Safely test policy transfer with safety checks"""
        for safety_check in self.safety_checks:
            if not safety_check(policy, real_robot):
                print(f"Safety check {safety_check.__name__} failed!")
                return False

        # If all safety checks pass, proceed with cautious testing
        return self.cautious_real_world_test(policy, real_robot)

    def cautious_real_world_test(self, policy, real_robot):
        """Cautious real-world testing with human oversight"""
        # Start with very limited range of motion
        # Have emergency stop ready
        # Test with safety cage if possible
        pass
```

## Learning Objectives

By the end of this lesson, you should be able to:
- Understand the challenges of transferring RL policies from simulation to real robots
- Implement domain randomization techniques to improve policy robustness
- Apply domain adaptation methods to bridge sim-to-real gaps
- Use system identification to match simulation to real robot dynamics
- Design validation strategies for safe sim-to-real transfer
- Implement progressive transfer strategies for complex tasks
- Evaluate the success and safety of sim-to-real transfers
- Understand the trade-offs between simulation fidelity and transfer success