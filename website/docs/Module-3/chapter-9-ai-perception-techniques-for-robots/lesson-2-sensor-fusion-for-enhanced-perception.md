---
title: 'Sensor Fusion for Enhanced Perception'
description: 'Implement sensor fusion for improved perception, explaining importance of sensor fusion, combining data from multiple sensors, algorithms like Kalman filters'
chapter: 9
lesson: 2
module: 3
sidebar_label: 'Sensor Fusion for Enhanced Perception'
sidebar_position: 2
tags: ['Sensor Fusion', 'Kalman Filters', 'Multi-Sensor Integration', 'Data Fusion', 'Robot Perception']
keywords: ['sensor fusion', 'Kalman filters', 'multi-sensor integration', 'data fusion', 'robot perception', 'information integration', 'uncertainty management']
---

# Sensor Fusion for Enhanced Perception

## Overview

Sensor fusion is a critical technique in robotics that combines data from multiple sensors to achieve more accurate, reliable, and robust perception than what any single sensor could provide. By leveraging the complementary strengths of different sensors while mitigating their individual weaknesses, sensor fusion enables robots to better understand their environment and make informed decisions. This lesson explores the principles, techniques, and applications of sensor fusion in robotic perception.

## Fundamentals of Sensor Fusion

### Why Sensor Fusion?

Sensor fusion addresses several key challenges in robotic perception:

1. **Complementary Information**: Different sensors provide different types of information
2. **Redundancy**: Multiple sensors can provide the same information, increasing reliability
3. **Improved Accuracy**: Combined data often yields better estimates than individual sensors
4. **Robustness**: If one sensor fails, others can maintain perception capabilities
5. **Extended Coverage**: Different sensors may operate in different domains (e.g., visual, thermal, electromagnetic)

### Types of Sensor Fusion

#### 1. Data-Level Fusion
- Combines raw sensor measurements directly
- Highest level of detail but computationally intensive
- Requires synchronized, calibrated sensors

#### 2. Feature-Level Fusion
- Extracts features from each sensor and combines them
- Balances detail with computational efficiency
- Requires feature extraction algorithms

#### 3. Decision-Level Fusion
- Combines decisions or classifications from individual sensors
- Most computationally efficient
- May lose some information during early processing

### Mathematical Foundations

The fundamental principle behind sensor fusion is that multiple noisy measurements can be combined to produce an estimate with lower uncertainty than any individual measurement.

For two independent measurements of the same quantity with uncertainties σ₁ and σ₂, the optimal weighted estimate is:

```
x_combined = (σ₂² * x₁ + σ₁² * x₂) / (σ₁² + σ₂²)
σ_combined² = (σ₁² * σ₂²) / (σ₁² + σ₂²)
```

This shows that the combined estimate has lower uncertainty than either individual measurement.

## Kalman Filter Fundamentals

The Kalman filter is one of the most important algorithms in sensor fusion, providing an optimal way to combine predictions and measurements.

### Standard Kalman Filter

```python
import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector: [x, y, vx, vy] (position and velocity)
        self.x = np.zeros(state_dim)

        # State covariance matrix
        self.P = np.eye(state_dim) * 1000

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0

        # Identity matrix
        self.I = np.eye(state_dim)

    def predict(self, dt):
        """Prediction step"""
        # State transition model (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update step"""
        # Measurement matrix (only position is measured)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Innovation (measurement residual)
        y = measurement - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.P = (self.I - K @ H) @ self.P

    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1.0
        self.I = np.eye(state_dim)

    def predict(self, dt, control=None):
        """Nonlinear prediction step"""
        # Nonlinear motion model
        self.x = self.motion_model(self.x, dt, control)

        # Linearize motion model to get Jacobian
        F = self.compute_jacobian_F(self.x, dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Nonlinear update step"""
        # Nonlinear measurement model
        h_x = self.measurement_model(self.x)

        # Linearize measurement model to get Jacobian
        H = self.compute_jacobian_H(self.x)

        # Innovation
        y = measurement - h_x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.P = (self.I - K @ H) @ self.P

    def motion_model(self, state, dt, control):
        """Nonlinear motion model"""
        x, y, vx, vy, theta, omega = state

        # Bicycle model for robot motion
        new_x = x + vx * np.cos(theta) * dt
        new_y = y + vx * np.sin(theta) * dt
        new_vx = vx + control[0] * dt if control is not None else vx  # acceleration
        new_vy = vy + control[1] * dt if control is not None else vy
        new_theta = theta + omega * dt
        new_omega = omega + control[2] * dt if control is not None else omega  # angular acceleration

        return np.array([new_x, new_y, new_vx, new_vy, new_theta, new_omega])

    def measurement_model(self, state):
        """Nonlinear measurement model"""
        x, y, vx, vy, theta, omega = state
        # Measurement only includes position (x, y)
        return np.array([x, y])

    def compute_jacobian_F(self, state, dt):
        """Compute Jacobian of motion model"""
        x, y, vx, vy, theta, omega = state

        F = np.eye(self.state_dim)
        # Partial derivatives of motion model
        F[0, 2] = np.cos(theta) * dt  # dx/dvx
        F[0, 4] = -vx * np.sin(theta) * dt  # dx/dtheta
        F[1, 3] = np.sin(theta) * dt  # dy/dvy
        F[1, 4] = vx * np.cos(theta) * dt  # dy/dtheta

        return F

    def compute_jacobian_H(self, state):
        """Compute Jacobian of measurement model"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = 1.0  # dx/dx
        H[1, 1] = 1.0  # dy/dy

        return H

class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1.0

        # UKF parameters
        self.alpha = 1e-3
        self.kappa = 0
        self.beta = 2

        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.c = self.state_dim + self.lambda_

        # Weights
        self.Wm = np.full(2 * self.state_dim + 1, 1.0 / (2 * self.c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / self.c
        self.Wc[0] = self.lambda_ / self.c + (1 - self.alpha**2 + self.beta)

    def get_sigma_points(self):
        """Generate sigma points"""
        # Square root of covariance matrix
        U = np.linalg.cholesky(self.c * self.P)

        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = self.x

        for i in range(self.state_dim):
            sigma_points[i + 1] = self.x + U[:, i]
            sigma_points[i + 1 + self.state_dim] = self.x - U[:, i]

        return sigma_points

    def predict(self, dt, control=None):
        """UKF prediction step"""
        # Generate sigma points
        sigma_points = self.get_sigma_points()

        # Propagate sigma points through nonlinear function
        propagated_points = np.zeros_like(sigma_points)
        for i, point in enumerate(sigma_points):
            propagated_points[i] = self.motion_model(point, dt, control)

        # Calculate predicted state and covariance
        x_pred = np.sum(self.Wm[:, np.newaxis] * propagated_points, axis=0)

        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(2 * self.state_dim + 1):
            diff = (propagated_points[i] - x_pred)
            P_pred += self.Wc[i] * np.outer(diff, diff)

        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, measurement):
        """UKF update step"""
        # Generate sigma points
        sigma_points = self.get_sigma_points()

        # Transform sigma points through measurement function
        measurement_points = np.zeros((2 * self.state_dim + 1, self.measurement_dim))
        for i, point in enumerate(sigma_points):
            measurement_points[i] = self.measurement_model(point)

        # Calculate predicted measurement and covariance
        z_pred = np.sum(self.Wm[:, np.newaxis] * measurement_points, axis=0)

        P_zz = np.zeros((self.measurement_dim, self.measurement_dim))
        for i in range(2 * self.state_dim + 1):
            diff = (measurement_points[i] - z_pred)
            P_zz += self.Wc[i] * np.outer(diff, diff)

        P_zz += self.R

        # Calculate cross-covariance
        P_xz = np.zeros((self.state_dim, self.measurement_dim))
        for i in range(2 * self.state_dim + 1):
            x_diff = (sigma_points[i] - self.x)
            z_diff = (measurement_points[i] - z_pred)
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)

        # Calculate Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)

        # Update state and covariance
        self.x = self.x + K @ (measurement - z_pred)
        self.P = self.P - K @ P_zz @ K.T

    def motion_model(self, state, dt, control):
        """Nonlinear motion model (same as EKF)"""
        return self.extended_kf.motion_model(state, dt, control)

    def measurement_model(self, state):
        """Nonlinear measurement model (same as EKF)"""
        return self.extended_kf.measurement_model(state)
```

## Multi-Sensor Fusion Architecture

### Sensor Registration and Calibration

Before fusion, sensors must be properly calibrated and registered:

```python
class SensorCalibrator:
    def __init__(self):
        self.extrinsics = {}  # Sensor-to-robot transformations
        self.intrinsics = {}  # Internal sensor parameters
        self.time_offsets = {}  # Temporal synchronization offsets

    def calibrate_extrinsics(self, sensor_pairs):
        """
        Calibrate extrinsic parameters between sensors
        """
        transformations = {}

        for sensor1, sensor2 in sensor_pairs:
            # Find correspondences between sensors
            correspondences = self.find_sensor_correspondences(sensor1, sensor2)

            if len(correspondences) >= 3:
                # Compute transformation using point cloud registration
                transform = self.compute_rigid_transform(correspondences)
                transformations[(sensor1, sensor2)] = transform
                transformations[(sensor2, sensor1)] = np.linalg.inv(transform)

        return transformations

    def find_sensor_correspondences(self, sensor1, sensor2):
        """
        Find correspondences between two sensors
        """
        # This would use calibration targets, natural features, etc.
        # Implementation depends on sensor types
        pass

    def compute_rigid_transform(self, correspondences):
        """
        Compute rigid body transformation between coordinate systems
        """
        # Use SVD-based method for rigid transformation
        src_points = np.array([c[0] for c in correspondences])
        dst_points = np.array([c[1] for c in correspondences])

        # Center the points
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)

        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center

        # Compute covariance matrix
        H = src_centered.T @ dst_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation
        R = Vt.T @ U.T

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = dst_center - R @ src_center

        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

class MultiModalFusion:
    def __init__(self):
        self.kalman_filter = ExtendedKalmanFilter(state_dim=6, measurement_dim=2)
        self.sensor_weights = {
            'camera': 0.3,
            'lidar': 0.4,
            'imu': 0.5,
            'gps': 0.2
        }

        # Initialize sensor-specific processing
        self.camera_processor = CameraProcessor()
        self.lidar_processor = LidarProcessor()
        self.imu_processor = IMUProcessor()
        self.gps_processor = GPSProcessor()

    def process_multi_sensor_input(self, sensor_data, dt):
        """
        Process data from multiple sensors and fuse into single estimate
        """
        # Process each sensor independently
        camera_result = self.camera_processor.process(sensor_data['camera'])
        lidar_result = self.lidar_processor.process(sensor_data['lidar'])
        imu_result = self.imu_processor.process(sensor_data['imu'])
        gps_result = self.gps_processor.process(sensor_data['gps'])

        # Extract measurements for fusion
        measurements = self.extract_measurements(
            camera_result, lidar_result, imu_result, gps_result
        )

        # Apply sensor fusion
        fused_state = self.fuse_measurements(measurements, dt)

        return fused_state

    def extract_measurements(self, camera_data, lidar_data, imu_data, gps_data):
        """
        Extract relevant measurements from sensor data
        """
        measurements = {}

        # Position measurements
        if camera_data.get('valid') and camera_data.get('position'):
            measurements['camera_pos'] = {
                'value': camera_data['position'],
                'covariance': camera_data['position_covariance'],
                'timestamp': camera_data['timestamp']
            }

        if lidar_data.get('valid') and lidar_data.get('position'):
            measurements['lidar_pos'] = {
                'value': lidar_data['position'],
                'covariance': lidar_data['position_covariance'],
                'timestamp': lidar_data['timestamp']
            }

        if gps_data.get('valid') and gps_data.get('position'):
            measurements['gps_pos'] = {
                'value': gps_data['position'],
                'covariance': gps_data['position_covariance'],
                'timestamp': gps_data['timestamp']
            }

        # Velocity measurements
        if imu_data.get('valid') and imu_data.get('velocity'):
            measurements['imu_vel'] = {
                'value': imu_data['velocity'],
                'covariance': imu_data['velocity_covariance'],
                'timestamp': imu_data['timestamp']
            }

        return measurements

    def fuse_measurements(self, measurements, dt):
        """
        Fuse measurements using weighted averaging or Kalman filtering
        """
        # Prediction step
        self.kalman_filter.predict(dt)

        # Process each measurement
        for sensor_type, measurement_data in measurements.items():
            # Update Kalman filter with each measurement
            self.kalman_filter.update(
                measurement_data['value'],
                measurement_data['covariance']
            )

        return self.kalman_filter.get_state()

    def handle_sensor_failure(self, failed_sensor):
        """
        Handle failure of a particular sensor
        """
        # Reduce weight of failed sensor
        if failed_sensor in self.sensor_weights:
            self.sensor_weights[failed_sensor] *= 0.1  # Significantly reduce weight

        # Increase weights of other sensors proportionally
        remaining_weight = sum(w for s, w in self.sensor_weights.items()
                             if s != failed_sensor)
        if remaining_weight > 0:
            scale_factor = (1.0 - 0.1 * self.sensor_weights[failed_sensor]) / remaining_weight
            for sensor in self.sensor_weights:
                if sensor != failed_sensor:
                    self.sensor_weights[sensor] *= scale_factor

class ParticleFilterFusion:
    def __init__(self, state_dim=6, num_particles=1000):
        self.state_dim = state_dim
        self.num_particles = num_particles

        # Initialize particles
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

        # Sensor noise models
        self.sensor_noise_models = {
            'camera': {'position': 0.05, 'orientation': 0.01},  # meters, radians
            'lidar': {'position': 0.02, 'orientation': 0.005},
            'imu': {'velocity': 0.01, 'orientation_rate': 0.001},
            'gps': {'position': 2.0, 'velocity': 0.1}  # GPS has larger uncertainty
        }

    def predict(self, control_input, dt):
        """Predict particle states based on control input"""
        for i in range(self.num_particles):
            self.particles[i] = self.motion_model(self.particles[i], control_input, dt)

        # Add process noise
        process_noise = np.random.normal(0, 0.1, (self.num_particles, self.state_dim))
        self.particles += process_noise

    def motion_model(self, state, control, dt):
        """Motion model for particle prediction"""
        x, y, z, vx, vy, vz = state
        ax, ay, az = control

        # Simple constant acceleration model
        new_x = x + vx * dt + 0.5 * ax * dt**2
        new_y = y + vy * dt + 0.5 * ay * dt**2
        new_z = z + vz * dt + 0.5 * az * dt**2
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_vz = vz + az * dt

        return np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])

    def update(self, sensor_observations):
        """Update particle weights based on sensor observations"""
        for i in range(self.num_particles):
            total_likelihood = 1.0

            for sensor_type, obs in sensor_observations.items():
                if obs is not None:
                    # Calculate likelihood of this particle given observation
                    particle_obs = self.predict_sensor_observation(
                        self.particles[i], sensor_type
                    )

                    # Calculate likelihood using sensor noise model
                    likelihood = self.calculate_observation_likelihood(
                        obs, particle_obs, sensor_type
                    )

                    total_likelihood *= likelihood

            # Update particle weight
            self.weights[i] *= total_likelihood

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

        # Resample if effective sample size is too low
        effective_samples = 1.0 / np.sum(self.weights**2)
        if effective_samples < self.num_particles / 2:
            self.resample()

    def predict_sensor_observation(self, particle_state, sensor_type):
        """Predict what a sensor would observe for a given particle state"""
        x, y, z, vx, vy, vz = particle_state

        if sensor_type in ['camera', 'lidar', 'gps']:
            # These sensors observe position
            return np.array([x, y, z])
        elif sensor_type == 'imu':
            # IMU observes velocity and acceleration
            return np.array([vx, vy, vz])
        else:
            return particle_state

    def calculate_observation_likelihood(self, observed, predicted, sensor_type):
        """Calculate likelihood of observation given prediction"""
        # Calculate difference
        diff = observed - predicted

        # Get sensor-specific noise
        if sensor_type in self.sensor_noise_models:
            noise_std = self.sensor_noise_models[sensor_type]
            # Simplified: assume position noise for all sensors
            if isinstance(noise_std, dict):
                noise_std = noise_std.get('position', 0.1)
        else:
            noise_std = 0.1

        # Calculate Gaussian likelihood
        likelihood = np.exp(-0.5 * np.sum((diff / noise_std)**2))

        return max(likelihood, 1e-10)  # Avoid zero probabilities

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = self.systematic_resample()

        # Resample particles and reset weights
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def systematic_resample(self):
        """Systematic resampling algorithm"""
        cumulative_sum = np.cumsum(self.weights)
        n = self.num_particles
        u_offset = np.random.uniform(0, 1/n)
        u_values = (np.arange(n) + u_offset) / n

        indices = np.searchsorted(cumulative_sum, u_values, side='right')
        return indices

    def estimate_state(self):
        """Get state estimate from particles"""
        # Weighted average of particles
        estimate = np.average(self.particles, axis=0, weights=self.weights)
        return estimate

    def get_uncertainty(self):
        """Get uncertainty estimate from particle distribution"""
        # Calculate weighted covariance
        estimate = self.estimate_state()
        diff = self.particles - estimate
        uncertainty = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.num_particles):
            uncertainty += self.weights[i] * np.outer(diff[i], diff[i])

        return uncertainty
```

## Advanced Fusion Techniques

### Deep Learning-Based Fusion

Modern sensor fusion increasingly uses deep learning approaches:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSensorFusion(nn.Module):
    def __init__(self, input_dims, fusion_dim=512, output_dim=6):
        super(DeepSensorFusion, self).__init__()

        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        # Modality-specific encoders
        self.camera_encoder = self._create_encoder(input_dims['camera'], fusion_dim)
        self.lidar_encoder = self._create_encoder(input_dims['lidar'], fusion_dim)
        self.imu_encoder = self._create_encoder(input_dims['imu'], fusion_dim)
        self.gps_encoder = self._create_encoder(input_dims['gps'], fusion_dim)

        # Attention mechanism for dynamic fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),  # 4 modalities
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, output_dim)
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty values
        )

    def _create_encoder(self, input_dim, hidden_dim):
        """Create modality-specific encoder"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )

    def forward(self, sensor_inputs):
        """
        Forward pass through the fusion network
        sensor_inputs: dict with keys 'camera', 'lidar', 'imu', 'gps'
        """
        # Encode each modality
        camera_feat = self.camera_encoder(sensor_inputs['camera'])
        lidar_feat = self.lidar_encoder(sensor_inputs['lidar'])
        imu_feat = self.imu_encoder(sensor_inputs['imu'])
        gps_feat = self.gps_encoder(sensor_inputs['gps'])

        # Stack features for attention mechanism
        modalities = torch.stack([
            camera_feat, lidar_feat, imu_feat, gps_feat
        ], dim=1)  # Shape: (batch, 4, fusion_dim)

        # Apply attention to learn fusion weights
        attended_features, attention_weights = self.attention(
            modalities, modalities, modalities
        )

        # Sum attended features
        fused_features = attended_features.sum(dim=1)  # Sum across modalities

        # Final fusion and output
        output = self.fusion_network(fused_features)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(
            self.fusion_network[:-1](fused_features)  # Use features before final layer
        )

        return output, uncertainty, attention_weights

    def compute_modality_confidence(self, attention_weights):
        """Compute confidence scores for each modality based on attention"""
        # Average attention weights across batch and heads
        modality_importance = attention_weights.mean(dim=[0, 1])  # Average across batch and heads
        return modality_importance

class LearnedFusionLayer(nn.Module):
    def __init__(self, input_dim, num_sensors):
        super(LearnedFusionLayer, self).__init__()

        self.num_sensors = num_sensors

        # Learnable weights for each sensor
        self.sensor_weights = nn.Parameter(torch.ones(num_sensors))

        # Learnable fusion transformation
        self.fusion_transform = nn.Linear(input_dim * num_sensors, input_dim)

        # Gate mechanism for adaptive fusion
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * num_sensors, input_dim),
            nn.Sigmoid()
        )

    def forward(self, sensor_features):
        """
        sensor_features: list of tensors, one for each sensor
        """
        # Concatenate all sensor features
        concat_features = torch.cat(sensor_features, dim=-1)

        # Apply sensor-specific weights
        weighted_features = []
        for i, feat in enumerate(sensor_features):
            weight = torch.softmax(self.sensor_weights, dim=0)[i]
            weighted_features.append(feat * weight)

        # Concatenate weighted features
        weighted_concat = torch.cat(weighted_features, dim=-1)

        # Apply fusion transformation
        fused = self.fusion_transform(weighted_concat)

        # Apply gating mechanism
        gate = self.gate_network(concat_features)
        output = fused * gate

        return output

class TemporalFusionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, sequence_length=10):
        super(TemporalFusionNetwork, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # LSTM for temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Attention over time steps
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Output prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, sensor_sequences):
        """
        sensor_sequences: dict where each value is a sequence of features
        """
        outputs = {}

        for sensor_name, sequence in sensor_sequences.items():
            # sequence shape: (batch, seq_len, features)
            temporal_features, _ = self.temporal_encoder(sequence)

            # Apply temporal attention
            attended_features, _ = self.temporal_attention(
                temporal_features, temporal_features, temporal_features
            )

            # Use last attended feature for prediction
            prediction = self.prediction_head(attended_features[:, -1, :])
            outputs[sensor_name] = prediction

        return outputs
```

## Practical Implementation Example

### Complete Fusion System

```python
class RobustSensorFusionSystem:
    def __init__(self):
        # Initialize different fusion approaches
        self.kalman_fusion = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)
        self.particle_fusion = ParticleFilterFusion(state_dim=6, num_particles=500)
        self.deep_fusion = DeepSensorFusion(
            input_dims={'camera': 10, 'lidar': 50, 'imu': 6, 'gps': 3},
            fusion_dim=256,
            output_dim=6
        )

        # Sensor health monitoring
        self.sensor_health = {
            'camera': 1.0,  # 1.0 = healthy, 0.0 = failed
            'lidar': 1.0,
            'imu': 1.0,
            'gps': 1.0
        }

        # Adaptive fusion weights
        self.adaptive_weights = {
            'kalman': 0.4,
            'particle': 0.3,
            'deep': 0.3
        }

        # Time synchronization
        self.time_buffer = {}  # Buffer for time synchronization
        self.max_time_diff = 0.05  # 50ms max acceptable time difference

    def process_sensor_data(self, raw_sensor_data, timestamp):
        """
        Process raw sensor data and perform fusion
        """
        # 1. Preprocess and validate sensor data
        validated_data = self.validate_sensor_data(raw_sensor_data)

        # 2. Time synchronization
        synced_data = self.synchronize_sensors(validated_data, timestamp)

        # 3. Apply different fusion methods
        kalman_result = self.kalman_fusion_step(synced_data)
        particle_result = self.particle_fusion_step(synced_data)
        deep_result = self.deep_fusion_step(synced_data)

        # 4. Adaptive fusion combining all results
        final_estimate = self.adaptive_fusion(
            kalman_result, particle_result, deep_result
        )

        # 5. Update sensor health based on consistency
        self.update_sensor_health(synced_data, final_estimate)

        return final_estimate

    def validate_sensor_data(self, raw_data):
        """Validate and clean sensor data"""
        validated = {}

        for sensor_name, data in raw_data.items():
            # Check for NaN or infinite values
            if isinstance(data, np.ndarray):
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    self.sensor_health[sensor_name] *= 0.5  # Reduce confidence
                    validated[sensor_name] = None  # Mark as invalid
                else:
                    validated[sensor_name] = data
            else:
                validated[sensor_name] = data

        return validated

    def synchronize_sensors(self, validated_data, current_timestamp):
        """Synchronize sensor data to current timestamp"""
        synced = {}

        for sensor_name, data in validated_data.items():
            if data is not None:
                # Check if data is recent enough
                if abs(data.get('timestamp', current_timestamp) - current_timestamp) <= self.max_time_diff:
                    synced[sensor_name] = data
                else:
                    # Use prediction or interpolation if available
                    synced[sensor_name] = self.interpolate_sensor_data(sensor_name, current_timestamp)
            else:
                synced[sensor_name] = None

        return synced

    def kalman_fusion_step(self, synced_data):
        """Perform Kalman filter fusion step"""
        # Extract position measurements from available sensors
        measurements = []

        if synced_data.get('camera') is not None:
            measurements.append(synced_data['camera']['position'])

        if synced_data.get('lidar') is not None:
            measurements.append(synced_data['lidar']['position'])

        if synced_data.get('gps') is not None:
            measurements.append(synced_data['gps']['position'])

        # Average available measurements
        if measurements:
            avg_measurement = np.mean(measurements, axis=0)
            self.kalman_fusion.update(avg_measurement)

        return self.kalman_fusion.get_state()

    def particle_fusion_step(self, synced_data):
        """Perform particle filter fusion step"""
        # Prepare observations for particle filter
        observations = {}

        for sensor_name, data in synced_data.items():
            if data is not None and 'position' in data:
                observations[sensor_name] = data['position']

        # Update particle filter
        self.particle_fusion.update(observations)

        return self.particle_fusion.estimate_state()

    def deep_fusion_step(self, synced_data):
        """Perform deep learning fusion step"""
        # Prepare input tensors for deep fusion
        input_tensors = {}

        # Convert sensor data to appropriate tensor formats
        if synced_data.get('camera') is not None:
            input_tensors['camera'] = torch.tensor(
                synced_data['camera']['features'], dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension

        if synced_data.get('lidar') is not None:
            input_tensors['lidar'] = torch.tensor(
                synced_data['lidar']['point_features'], dtype=torch.float32
            ).unsqueeze(0)

        if synced_data.get('imu') is not None:
            input_tensors['imu'] = torch.tensor(
                synced_data['imu']['measurements'], dtype=torch.float32
            ).unsqueeze(0)

        if synced_data.get('gps') is not None:
            input_tensors['gps'] = torch.tensor(
                synced_data['gps']['coordinates'], dtype=torch.float32
            ).unsqueeze(0)

        # Perform deep fusion
        with torch.no_grad():
            fused_output, uncertainty, attention_weights = self.deep_fusion(input_tensors)

        return fused_output.numpy().flatten()

    def adaptive_fusion(self, kalman_result, particle_result, deep_result):
        """Adaptively combine results from different fusion methods"""
        # Get current sensor health and adjust weights accordingly
        health_weights = self.get_health_based_weights()

        # Combine results using adaptive weights
        final_state = (
            self.adaptive_weights['kalman'] * health_weights['kalman'] * kalman_result +
            self.adaptive_weights['particle'] * health_weights['particle'] * particle_result +
            self.adaptive_weights['deep'] * health_weights['deep'] * deep_result
        ) / (
            self.adaptive_weights['kalman'] * health_weights['kalman'] +
            self.adaptive_weights['particle'] * health_weights['particle'] +
            self.adaptive_weights['deep'] * health_weights['deep']
        )

        return final_state

    def get_health_based_weights(self):
        """Adjust weights based on sensor health"""
        # Calculate health-based weights
        avg_health = np.mean(list(self.sensor_health.values()))

        health_weights = {}

        # Kalman filter weight based on position sensor health
        pos_sensors = ['camera', 'lidar', 'gps']
        pos_health = np.mean([self.sensor_health[s] for s in pos_sensors if s in self.sensor_health])
        health_weights['kalman'] = pos_health

        # Particle filter weight based on all sensor health
        health_weights['particle'] = avg_health

        # Deep fusion weight based on sensor availability
        available_sensors = sum(1 for h in self.sensor_health.values() if h > 0.5)
        health_weights['deep'] = min(1.0, available_sensors / len(self.sensor_health))

        return health_weights

    def update_sensor_health(self, synced_data, estimate):
        """Update sensor health based on consistency with estimate"""
        for sensor_name, data in synced_data.items():
            if data is not None and 'position' in data:
                # Calculate consistency with current estimate
                est_pos = estimate[:3]  # Position part of state
                meas_pos = data['position']

                consistency = np.linalg.norm(est_pos - meas_pos)

                # Update health based on consistency (lower is better)
                if consistency < 0.1:  # Good consistency
                    self.sensor_health[sensor_name] = min(1.0, self.sensor_health[sensor_name] + 0.01)
                else:  # Poor consistency
                    self.sensor_health[sensor_name] = max(0.1, self.sensor_health[sensor_name] - 0.02)

    def get_fusion_confidence(self):
        """Get overall confidence in the fusion result"""
        # Combine sensor health and method weights
        method_confidence = sum(
            self.adaptive_weights[method] * self.get_health_based_weights()[method]
            for method in ['kalman', 'particle', 'deep']
        )

        avg_sensor_health = np.mean(list(self.sensor_health.values()))

        overall_confidence = method_confidence * avg_sensor_health

        return overall_confidence

# Example usage
def run_sensor_fusion_demo():
    """Run a complete sensor fusion demonstration"""

    # Initialize fusion system
    fusion_system = RobustSensorFusionSystem()

    # Simulate sensor data over time
    results = []

    for t in range(100):  # Simulate 100 time steps
        # Simulate sensor data (in real application, this would come from actual sensors)
        raw_data = {
            'camera': {
                'position': np.array([t*0.1 + np.random.normal(0, 0.05), 0.5 + np.random.normal(0, 0.03), 0.2]),
                'features': np.random.rand(10).astype(np.float32),
                'timestamp': t * 0.1
            },
            'lidar': {
                'position': np.array([t*0.1 + np.random.normal(0, 0.02), 0.5 + np.random.normal(0, 0.01), 0.2]),
                'point_features': np.random.rand(50).astype(np.float32),
                'timestamp': t * 0.1
            },
            'imu': {
                'measurements': np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1]),  # [vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
                'timestamp': t * 0.1
            },
            'gps': {
                'coordinates': np.array([t*0.1 + np.random.normal(0, 0.5), 0.5 + np.random.normal(0, 0.3), 0.2]),
                'timestamp': t * 0.1
            }
        }

        # Process through fusion system
        fused_estimate = fusion_system.process_sensor_data(raw_data, t * 0.1)

        # Store results
        results.append({
            'time': t * 0.1,
            'estimate': fused_estimate,
            'confidence': fusion_system.get_fusion_confidence()
        })

        print(f"Time {t*0.1:.1f}s - Estimated position: [{fused_estimate[0]:.2f}, {fused_estimate[1]:.2f}, {fused_estimate[2]:.2f}] "
              f"- Confidence: {fusion_system.get_fusion_confidence():.3f}")

    return results

# Run the demo
if __name__ == "__main__":
    fusion_results = run_sensor_fusion_demo()
```

## Sensor Fusion Best Practices

### 1. Data Association
- Implement robust data association algorithms
- Handle false positives and negatives
- Consider temporal consistency

### 2. Synchronization
- Properly handle time delays between sensors
- Implement interpolation/extrapolation for time alignment
- Use hardware triggers when possible

### 3. Calibration
- Regularly recalibrate extrinsic parameters
- Monitor for calibration drift
- Implement self-calibration routines

### 4. Fault Detection
- Implement sensor health monitoring
- Detect and handle sensor failures gracefully
- Maintain degraded operation when possible

### 5. Performance Optimization
- Optimize algorithms for real-time performance
- Use appropriate data structures
- Consider parallel processing where possible

## Learning Objectives

By the end of this lesson, you should be able to:
- Understand the principles and benefits of sensor fusion in robotics
- Implement Kalman filter variants for sensor fusion
- Design multi-modal fusion architectures combining different sensor types
- Apply particle filtering techniques for non-linear sensor fusion
- Implement deep learning approaches for learned sensor fusion
- Address practical challenges in sensor fusion including calibration and synchronization
- Evaluate fusion system performance and robustness
- Design fault-tolerant fusion systems that handle sensor failures