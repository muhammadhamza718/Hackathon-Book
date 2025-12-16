---
title: 'Using NVIDIA Isaac for Manipulation'
description: 'Employ NVIDIA Isaac for robot manipulation tasks, discussing Isaac for manipulation (grasp planning, motion control), AI in dexterity'
chapter: 8
lesson: 3
module: 3
sidebar_label: 'Using NVIDIA Isaac for Manipulation'
sidebar_position: 3
tags: ['NVIDIA Isaac', 'Manipulation', 'Grasp Planning', 'Motion Control', 'AI Dexterity']
keywords: ['NVIDIA Isaac', 'manipulation', 'grasp planning', 'motion control', 'AI dexterity', 'robotic arms', 'end-effectors']
---

# Using NVIDIA Isaac for Manipulation

## Overview

Robot manipulation is a complex field that involves planning and executing precise movements to interact with objects in the environment. NVIDIA Isaac provides powerful tools for simulating, developing, and testing manipulation tasks in a safe, controlled environment. This lesson explores how to leverage Isaac's capabilities for grasp planning, motion control, and AI-driven dexterity in robotic manipulation.

## Manipulation in Robotics

### The Manipulation Pipeline

Robot manipulation typically involves several interconnected components:

1. **Perception**: Identifying and localizing objects to manipulate
2. **Planning**: Determining how to approach and grasp objects
3. **Control**: Executing precise movements to achieve manipulation goals
4. **Learning**: Improving manipulation skills through experience
5. **Adaptation**: Adjusting to changing conditions and unexpected situations

### Challenges in Manipulation

Manipulation tasks face several significant challenges:

- **Contact Mechanics**: Understanding and controlling physical interactions
- **Precision Requirements**: Achieving millimeter-level accuracy in some tasks
- **Uncertainty Handling**: Managing uncertainty in object poses and properties
- **Dynamic Environments**: Adapting to moving or changing objects
- **Force Control**: Managing forces during grasping and manipulation
- **Multi-finger Coordination**: Coordinating multiple fingers or end-effectors

## Isaac's Manipulation Capabilities

### Physics Simulation for Manipulation

Isaac provides realistic physics simulation essential for manipulation:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim, ClothPrim
import numpy as np

class ManipulationWorld(World):
    def __init__(self):
        super().__init__(stage_units_in_meters=1.0)

        # Configure physics parameters for manipulation
        self.physics_scene.set_solver_type("TGS")  # Use TGS solver for stable contacts
        self.physics_scene.set_friction_enabled(True)
        self.physics_scene.set_rest_offset(0.001)  # Small rest offset for stability
        self.physics_scene.set_contact_offset(0.002)  # Contact offset for proper contacts

        # Add ground plane
        self.scene.add_ground_plane(prim_path="/World/Ground",
                                   static_friction=0.5,
                                   dynamic_friction=0.5,
                                   restitution=0.0)

        # Add robot
        self.robot = self.scene.add(
            Robot(
                prim_path="/World/Robot",
                usd_path="/Isaac/Robots/Franka/franka_instanceable.usd",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        )

        # Add objects to manipulate
        self.add_manipulation_objects()

    def add_manipulation_objects(self):
        """Add objects for manipulation tasks"""
        # Add a cube to manipulate
        self.cube = self.scene.add(
            RigidPrim(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.5, 0.0, 0.1]),
                scale=np.array([0.05, 0.05, 0.05]),
                mass=0.1,
                collision_mesh_usd_path="cube.usd"
            )
        )

        # Add a cylinder
        self.cylinder = self.scene.add(
            RigidPrim(
                prim_path="/World/Cylinder",
                name="cylinder",
                position=np.array([0.6, 0.1, 0.1]),
                scale=np.array([0.03, 0.03, 0.1]),
                mass=0.05,
                collision_mesh_usd_path="cylinder.usd"
            )
        )

# Initialize the manipulation world
world = ManipulationWorld()
world.reset()
```

### Robot Configuration for Manipulation

Configuring robots for manipulation tasks:

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulation import GraspPlanner

class ManipulationRobot:
    def __init__(self, world):
        self.world = world
        self.robot = world.robot
        self.grasp_planner = None

        # Initialize robot controllers
        self.setup_controllers()

    def setup_controllers(self):
        """Set up robot controllers for manipulation"""
        # Get robot's end-effector
        self.end_effector_frame = self.robot.get_end_effector_frame()

        # Initialize inverse kinematics solver
        self.ik_solver = self.setup_inverse_kinematics()

        # Initialize joint position controller
        self.joint_controller = self.robot.get_articulation_controller()

        # Initialize Cartesian controller
        self.cartesian_controller = self.setup_cartesian_controller()

    def setup_inverse_kinematics(self):
        """Set up inverse kinematics solver"""
        from omni.isaac.core.utils.viewports import create_viewport
        from omni.kit.primitive.mesh import SphereMesh

        # Create IK solver for the robot arm
        ik_solver = self.robot.create_ik_solver(
            name="franka_ik",
            joint_names=[
                "panda_joint1", "panda_joint2", "panda_joint3",
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
            ]
        )
        return ik_solver

    def setup_cartesian_controller(self):
        """Set up Cartesian space controller"""
        from omni.isaac.core.controllers import DifferentialIkController

        cartesian_controller = DifferentialIkController(
            name="cartesian_controller",
            target_end_effector_name=self.end_effector_frame.name,
            robot_articulation=self.robot,
            joint_names=[
                "panda_joint1", "panda_joint2", "panda_joint3",
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
            ]
        )
        return cartesian_controller

# Example of setting up a manipulation robot
manipulation_robot = ManipulationRobot(world)
```

## Grasp Planning in Isaac

### Physics-Based Grasp Simulation

Isaac enables realistic grasp simulation with accurate physics:

```python
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdPhysics, PhysxSchema
import omni.physx

class GraspSimulator:
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
        self.physics_sim_view = world.physics_sim_view

        # Initialize grasp analysis tools
        self.contact_manager = self.setup_contact_analysis()

    def setup_contact_analysis(self):
        """Set up contact analysis for grasp evaluation"""
        # Enable contact reporting for the robot's gripper
        gripper_links = self.get_gripper_links()

        for link_path in gripper_links:
            link_prim = get_prim_at_path(link_path)

            # Enable contact reporting
            UsdPhysics.CollisionAPI.Apply(link_prim)

            # Set up contact filters if needed
            contact_api = PhysxSchema.PhysxCollisionAPI.Apply(link_prim)
            contact_api.GetContactOffsetAttr().Set(0.002)
            contact_api.GetRestOffsetAttr().Set(0.001)

        return self.physics_sim_view

    def get_gripper_links(self):
        """Get all gripper link paths"""
        # This would be specific to your robot's USD structure
        gripper_paths = []

        # Example for Franka Panda robot
        if "franka" in self.robot.prim_path:
            gripper_paths = [
                f"{self.robot.prim_path}/panda_leftfinger",
                f"{self.robot.prim_path}/panda_rightfinger",
                f"{self.robot.prim_path}/panda_hand"
            ]

        return gripper_paths

    def evaluate_grasp_quality(self, object_prim_path, grasp_pose):
        """Evaluate the quality of a grasp using physics simulation"""
        # Move robot to grasp pose
        self.move_to_grasp_pose(grasp_pose)

        # Apply small disturbance to test grasp stability
        self.apply_disturbance(object_prim_path)

        # Measure grasp stability over time
        stability_score = self.measure_grasp_stability(object_prim_path)

        return stability_score

    def move_to_grasp_pose(self, grasp_pose):
        """Move robot to a specific grasp pose"""
        # Use IK controller to move to grasp pose
        joint_positions = self.inverse_kinematics(grasp_pose)
        self.robot.set_joint_positions(joint_positions)

    def inverse_kinematics(self, target_pose):
        """Calculate joint positions for target end-effector pose"""
        # This would use the IK solver set up earlier
        joint_positions = self.robot.ik_solver.compute(
            target_position=target_pose.position,
            target_orientation=target_pose.orientation
        )
        return joint_positions

    def apply_disturbance(self, object_prim_path):
        """Apply small disturbance to test grasp stability"""
        object_handle = self.physics_sim_view.get_rigid_body_view(object_prim_path)

        # Apply small random force
        disturbance_force = np.random.uniform(-0.5, 0.5, size=3)
        object_handle.apply_force(disturbance_force, indices=[0])

    def measure_grasp_stability(self, object_prim_path):
        """Measure how well the grasp holds the object"""
        initial_position = self.get_object_position(object_prim_path)

        # Run simulation for a short period
        for _ in range(50):  # 50 steps at 60Hz = ~0.83 seconds
            self.world.step(render=False)

        final_position = self.get_object_position(object_prim_path)

        # Calculate displacement - less displacement = more stable grasp
        displacement = np.linalg.norm(final_position - initial_position)
        stability_score = 1.0 / (1.0 + displacement)  # Higher score = more stable

        return stability_score

    def get_object_position(self, object_prim_path):
        """Get current position of object"""
        object_handle = self.physics_sim_view.get_rigid_body_view(object_prim_path)
        positions, orientations = object_handle.get_world_poses()
        return positions[0]
```

### AI-Based Grasp Planning

Implementing AI-based grasp planning:

```python
import torch
import torch.nn as nn
import numpy as np

class GraspPlannerNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=6):  # 6-DOF grasp poses
        super(GraspPlannerNet, self).__init__()

        # Convolutional layers for processing depth/RGB images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU()
        )

        # Calculate flattened size after convolutions
        self.flattened_size = 256 * 4 * 4  # Assumes 64x64 input

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc_layers(x)
        return x

class AIManipulationController:
    def __init__(self, robot, camera):
        self.robot = robot
        self.camera = camera

        # Initialize AI grasp planner
        self.grasp_net = self.load_trained_model()

        # Initialize grasp evaluation
        self.grasp_evaluator = GraspSimulator(robot, robot.world)

    def load_trained_model(self):
        """Load a pre-trained grasp planning model"""
        # In practice, you'd load a model trained on Isaac-generated data
        model = GraspPlannerNet()

        # Load trained weights (example)
        try:
            model.load_state_dict(torch.load("grasp_planning_model.pth"))
            model.eval()
            print("Loaded trained grasp planning model")
        except:
            print("Could not load trained model, using random initialization")

        return model

    def plan_grasp_from_camera(self):
        """Plan a grasp based on camera observation"""
        # Get camera image
        camera_data = self.camera.get_current_frame()
        rgb_image = camera_data["rgb"]

        # Preprocess image for the network
        input_tensor = self.preprocess_image(rgb_image)

        # Get grasp predictions
        with torch.no_grad():
            grasp_predictions = self.grasp_net(input_tensor)

        # Convert predictions to grasp poses
        grasp_poses = self.predictions_to_grasps(grasp_predictions)

        # Evaluate and select best grasp
        best_grasp = self.select_best_grasp(grasp_poses)

        return best_grasp

    def preprocess_image(self, image):
        """Preprocess camera image for neural network"""
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Resize to network input size (example: 64x64)
        import torch.nn.functional as F
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(64, 64),
            mode='bilinear'
        ).squeeze(0)

        return image_tensor.unsqueeze(0)  # Add batch dimension

    def predictions_to_grasps(self, predictions):
        """Convert network predictions to grasp poses"""
        # This would convert network outputs to 6-DOF grasp poses
        # In practice, this would be specific to your network architecture

        # Example conversion (simplified)
        grasp_poses = []
        for i in range(predictions.shape[0]):  # batch dimension
            # Extract 6-DOF pose (x, y, z, rx, ry, rz)
            pose_data = predictions[i].cpu().numpy()

            grasp_pose = {
                'position': pose_data[:3],      # x, y, z
                'orientation': pose_data[3:],   # r, p, y or quaternion
                'quality': self.calculate_grasp_quality(pose_data)
            }
            grasp_poses.append(grasp_pose)

        return grasp_poses

    def select_best_grasp(self, grasp_poses):
        """Select the best grasp from candidates"""
        if not grasp_poses:
            return None

        # Evaluate each grasp using physics simulation
        evaluated_grasps = []
        for grasp in grasp_poses:
            quality = self.grasp_evaluator.evaluate_grasp_quality(
                object_prim_path="/World/Object",  # Specify target object
                grasp_pose=grasp
            )
            evaluated_grasps.append((grasp, quality))

        # Sort by quality and return best
        best_grasp = max(evaluated_grasps, key=lambda x: x[1])
        return best_grasp[0]  # Return just the grasp pose, not the quality

    def execute_grasp(self, grasp_pose):
        """Execute a planned grasp"""
        # Move to pre-grasp position
        pre_grasp_pose = self.calculate_pre_grasp_pose(grasp_pose)
        self.move_to_pose(pre_grasp_pose)

        # Open gripper
        self.open_gripper()

        # Move to grasp pose
        self.move_to_pose(grasp_pose)

        # Close gripper
        self.close_gripper()

        # Lift object
        lift_pose = self.calculate_lift_pose(grasp_pose)
        self.move_to_pose(lift_pose)

    def calculate_pre_grasp_pose(self, grasp_pose, distance=0.1):
        """Calculate pre-grasp pose (approach from above)"""
        # Move slightly above the grasp pose
        pre_grasp = grasp_pose.copy()
        pre_grasp['position'][2] += distance  # Lift up
        return pre_grasp

    def calculate_lift_pose(self, grasp_pose, lift_distance=0.1):
        """Calculate post-grasp lift pose"""
        lift_pose = grasp_pose.copy()
        lift_pose['position'][2] += lift_distance
        return lift_pose

    def move_to_pose(self, pose):
        """Move end-effector to specified pose"""
        # Use Cartesian controller to move to pose
        joint_positions = self.inverse_kinematics(pose)
        self.robot.set_joint_positions(joint_positions)

    def open_gripper(self):
        """Open the robot gripper"""
        # Set gripper joint positions for open state
        self.robot.set_gripper_positions([0.04, 0.04])  # Example for Franka

    def close_gripper(self):
        """Close the robot gripper"""
        # Set gripper joint positions for closed state
        self.robot.set_gripper_positions([0.0, 0.0])  # Example for Franka

    def inverse_kinematics(self, target_pose):
        """Calculate joint positions for target pose"""
        # This would use the robot's IK solver
        joint_positions = self.robot.ik_solver.compute(
            target_position=target_pose['position'],
            target_orientation=target_pose['orientation']
        )
        return joint_positions
```

## Motion Control in Isaac

### Trajectory Planning and Execution

Advanced motion control capabilities in Isaac:

```python
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import math

class MotionController:
    def __init__(self, robot):
        self.robot = robot
        self.trajectory_generator = TrajectoryGenerator()

    def plan_cartesian_trajectory(self, waypoints, max_velocity=0.5, max_acceleration=1.0):
        """Plan a Cartesian trajectory through waypoints"""
        # Generate smooth trajectory between waypoints
        trajectory = self.trajectory_generator.generate_cartesian_trajectory(
            waypoints,
            max_velocity,
            max_acceleration
        )

        return trajectory

    def execute_cartesian_trajectory(self, trajectory, controller_type="position"):
        """Execute a Cartesian trajectory"""
        for waypoint in trajectory:
            if controller_type == "position":
                # Use position control
                joint_positions = self.cartesian_to_joint(waypoint)
                self.robot.set_joint_positions(joint_positions)
            elif controller_type == "velocity":
                # Use velocity control
                joint_velocities = self.calculate_joint_velocities(waypoint)
                self.robot.set_joint_velocities(joint_velocities)

            # Step simulation
            self.robot.world.step(render=True)

    def cartesian_to_joint(self, cartesian_pose):
        """Convert Cartesian pose to joint positions using IK"""
        joint_positions = self.robot.ik_solver.compute(
            target_position=cartesian_pose['position'],
            target_orientation=cartesian_pose['orientation']
        )
        return joint_positions

    def calculate_joint_velocities(self, target_pose):
        """Calculate joint velocities to reach target pose"""
        current_pose = self.get_current_end_effector_pose()

        # Calculate Cartesian velocity
        pos_error = target_pose['position'] - current_pose['position']
        ori_error = target_pose['orientation'] - current_pose['orientation']

        # Use Jacobian to convert Cartesian velocity to joint velocity
        jacobian = self.compute_jacobian()
        cartesian_velocity = np.concatenate([pos_error, ori_error])

        joint_velocities = np.linalg.pinv(jacobian) @ cartesian_velocity

        return joint_velocities

    def compute_jacobian(self):
        """Compute geometric Jacobian for the robot"""
        # This would use the robot's kinematic model
        # In Isaac, you can use built-in methods or compute numerically
        return self.robot.compute_jacobian()

    def get_current_end_effector_pose(self):
        """Get current end-effector pose"""
        ee_position, ee_orientation = self.robot.get_end_effector_position_orientation()
        return {
            'position': ee_position,
            'orientation': ee_orientation
        }

class TrajectoryGenerator:
    def __init__(self):
        self.interpolation_method = "cubic_spline"

    def generate_cartesian_trajectory(self, waypoints, max_velocity, max_acceleration):
        """Generate smooth Cartesian trajectory"""
        if len(waypoints) < 2:
            return waypoints

        # Use cubic spline interpolation for smooth trajectories
        if self.interpolation_method == "cubic_spline":
            return self.cubic_spline_interpolation(waypoints, max_velocity, max_acceleration)
        else:
            return self.linear_interpolation(waypoints)

    def cubic_spline_interpolation(self, waypoints, max_velocity, max_acceleration):
        """Generate cubic spline trajectory through waypoints"""
        import scipy.interpolate as spi

        # Extract positions for interpolation
        positions = np.array([wp['position'] for wp in waypoints])

        # Parameterize path
        distances = [0]
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            distances.append(distances[-1] + dist)

        # Create spline interpolators for each dimension
        tck_x = spi.splprep([positions[:, 0]], u=distances, s=0)[0]
        tck_y = spi.splprep([positions[:, 1]], u=distances, s=0)[0]
        tck_z = spi.splprep([positions[:, 2]], u=distances, s=0)[0]

        # Generate intermediate points
        total_distance = distances[-1]
        num_points = max(int(total_distance * 10), 10)  # 10 points per unit distance
        u_new = np.linspace(0, total_distance, num_points)

        # Interpolate positions
        x_new = spi.splev(u_new, tck_x)
        y_new = spi.splev(u_new, tck_y)
        z_new = spi.splev(u_new, tck_z)

        # Create interpolated waypoints
        interpolated_waypoints = []
        for i in range(len(u_new)):
            waypoint = {
                'position': np.array([x_new[i], y_new[i], z_new[i]]),
                'orientation': waypoints[0]['orientation']  # Keep initial orientation
            }
            interpolated_waypoints.append(waypoint)

        return interpolated_waypoints

    def linear_interpolation(self, waypoints):
        """Simple linear interpolation between waypoints"""
        interpolated = []

        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            end_wp = waypoints[i + 1]

            # Calculate distance between waypoints
            distance = np.linalg.norm(end_wp['position'] - start_wp['position'])
            num_interpolations = max(int(distance * 10), 5)  # 10 points per unit

            for j in range(num_interpolations):
                t = j / num_interpolations
                interp_pos = (1 - t) * start_wp['position'] + t * end_wp['position']

                interpolated.append({
                    'position': interp_pos,
                    'orientation': start_wp['orientation']
                })

        return interpolated + [waypoints[-1]]  # Add final waypoint
```

### Force Control and Impedance Control

Advanced control techniques for manipulation:

```python
class ForceImpedanceController:
    def __init__(self, robot, stiffness=1000, damping=20):
        self.robot = robot
        self.stiffness = stiffness
        self.damping = damping

        # Initialize force/torque sensors
        self.setup_force_sensors()

    def setup_force_sensors(self):
        """Set up force/torque sensors for the robot"""
        # In Isaac Sim, force sensors can be added to robot links
        from omni.isaac.core.sensors import ContactSensor

        # Add contact sensors to gripper fingers
        self.left_finger_sensor = ContactSensor(
            prim_path=f"{self.robot.prim_path}/panda_leftfinger/ContactSensor",
            translation=np.array([0.0, 0.0, 0.05]),
            radius=0.01
        )

        self.right_finger_sensor = ContactSensor(
            prim_path=f"{self.robot.prim_path}/panda_rightfinger/ContactSensor",
            translation=np.array([0.0, 0.0, 0.05]),
            radius=0.01
        )

    def impedance_control(self, desired_pose, desired_stiffness=None):
        """Implement impedance control for compliant manipulation"""
        if desired_stiffness is not None:
            stiffness = desired_stiffness
        else:
            stiffness = self.stiffness

        current_pose = self.get_current_end_effector_pose()

        # Calculate position and orientation errors
        pos_error = desired_pose['position'] - current_pose['position']
        ori_error = self.orientation_error(desired_pose['orientation'],
                                         current_pose['orientation'])

        # Calculate desired forces based on stiffness
        force = stiffness * pos_error
        torque = stiffness * ori_error

        # Apply impedance control law
        current_vel = self.get_end_effector_velocity()
        damping_force = self.damping * current_vel[:3]  # Linear damping
        damping_torque = self.damping * current_vel[3:]  # Angular damping

        # Total control effort
        total_force = force - damping_force
        total_torque = torque - damping_torque

        # Convert forces to joint torques using transpose Jacobian
        jacobian = self.compute_jacobian()
        jacobian_transpose = jacobian.T

        joint_torques = np.concatenate([total_force, total_torque]) @ jacobian_transpose

        # Apply torques to robot joints
        self.robot.set_joint_efforts(joint_torques)

    def orientation_error(self, desired_quat, current_quat):
        """Calculate orientation error as a 3D vector"""
        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation as R

        desired_rot = R.from_quat(desired_quat)
        current_rot = R.from_quat(current_quat)

        # Calculate relative rotation
        relative_rot = desired_rot * current_rot.inv()

        # Convert to axis-angle representation
        axis_angle = relative_rot.as_rotvec()

        return axis_angle

    def get_end_effector_velocity(self):
        """Get current end-effector velocity"""
        # This would use the robot's kinematic model and joint velocities
        joint_velocities = self.robot.get_joint_velocities()

        # Convert joint velocities to Cartesian velocities using Jacobian
        jacobian = self.compute_jacobian()
        cartesian_vel = jacobian @ joint_velocities

        return cartesian_vel

    def hybrid_force_position_control(self, position_targets, force_targets, selection_mask):
        """Implement hybrid force-position control"""
        # selection_mask indicates which DOFs are position-controlled (1) vs force-controlled (0)
        # 6 DOFs: [x, y, z, rx, ry, rz]

        current_pose = self.get_current_end_effector_pose()
        current_wrench = self.get_end_effector_wrench()

        # Calculate control commands for each DOF
        control_commands = np.zeros(6)

        for i in range(6):
            if selection_mask[i] == 1:  # Position control
                if i < 3:  # Linear DOF
                    pos_error = position_targets[i] - current_pose['position'][i]
                    control_commands[i] = self.stiffness * pos_error
                else:  # Angular DOF
                    # Orientation control would be implemented here
                    control_commands[i] = 0
            else:  # Force control
                force_error = force_targets[i] - current_wrench[i]
                control_commands[i] = force_error  # Pure force control

        # Apply control commands
        jacobian_transpose = self.compute_jacobian().T
        joint_torques = control_commands @ jacobian_transpose

        self.robot.set_joint_efforts(joint_torques)

    def get_end_effector_wrench(self):
        """Get current end-effector wrench (force and torque)"""
        # Get contact forces from sensors
        left_force = self.left_finger_sensor.get_contact_force()
        right_force = self.right_finger_sensor.get_contact_force()

        # Combine forces (simplified)
        total_force = left_force + right_force

        # In practice, you'd also calculate torques
        total_wrench = np.concatenate([total_force, np.zeros(3)])  # [fx, fy, fz, tx, ty, tz]

        return total_wrench
```

## AI in Dexterity

### Learning-Based Manipulation

Using reinforcement learning for dexterous manipulation:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ManipulationPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ManipulationPolicyNetwork, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Value output
        )

        # Action scaling (to map [-1, 1] to actual action space)
        self.register_buffer('action_scale', torch.ones(action_dim))
        self.register_buffer('action_bias', torch.zeros(action_dim))

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        return action * self.action_scale + self.action_bias, value

    def get_action(self, state):
        """Get action from policy (without gradients)"""
        with torch.no_grad():
            action, _ = self.forward(state)
        return action.numpy()

    def get_value(self, state):
        """Get value estimation (without gradients)"""
        with torch.no_grad():
            _, value = self.forward(state)
        return value.item()

class ManipulationRLAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = ManipulationPolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_net = ManipulationPolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_capacity = 100000
        self.batch_size = 128

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter
        self.update_every = 4  # Update frequency

    def select_action(self, state, add_noise=True):
        """Select action using the policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.policy_net(state_tensor)

        if add_noise:
            # Add exploration noise
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)

        return action.cpu().numpy()[0]

    def train(self, experiences):
        """Train the policy on a batch of experiences"""
        if len(experiences) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = self.sample_batch(experiences)

        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).unsqueeze(1).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_actions, _ = self.target_net(next_states)
            next_q_values = self.target_net.critic(next_states)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute current Q-values
        current_q_values = self.policy_net.critic(states)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        # Compute actor loss (maximize Q-values)
        predicted_actions, _ = self.policy_net(states)
        actor_loss = -self.policy_net.critic(
            torch.cat([states[:, :-action_dim], predicted_actions], dim=1)
        ).mean()

        # Update networks
        self.optimizer.zero_grad()
        (critic_loss + actor_loss).backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update()

        return actor_loss.item(), critic_loss.item()

    def sample_batch(self, experiences):
        """Sample a batch of experiences from replay buffer"""
        indices = np.random.choice(len(experiences), self.batch_size, replace=False)

        batch = {
            'states': [experiences[i][0] for i in indices],
            'actions': [experiences[i][1] for i in indices],
            'rewards': [experiences[i][2] for i in indices],
            'next_states': [experiences[i][3] for i in indices],
            'dones': [experiences[i][4] for i in indices]
        }

        return batch

    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class IsaacManipulationEnvironment:
    def __init__(self, robot, camera, objects):
        self.robot = robot
        self.camera = camera
        self.objects = objects

        # Define action and observation spaces
        self.action_dim = 7  # Joint position commands for 7-DOF arm
        self.state_dim = 30  # Example: joint positions, velocities, object poses, etc.

        # Initialize RL agent
        self.agent = ManipulationRLAgent(self.state_dim, self.action_dim)

    def get_state(self):
        """Get current state for RL agent"""
        # Combine various sensor readings into state vector
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()

        # Get object poses relative to robot
        object_poses = []
        for obj in self.objects:
            obj_pose = self.get_relative_pose(obj)
            object_poses.extend([obj_pose[0], obj_pose[1], obj_pose[2]])  # Just position

        # Get end-effector pose
        ee_pose = self.robot.get_end_effector_position_orientation()

        # Combine into state vector
        state = np.concatenate([
            joint_positions,
            joint_velocities,
            object_poses,
            ee_pose[0],  # Position
            ee_pose[1]   # Orientation
        ])

        return state

    def get_relative_pose(self, object_prim):
        """Get object pose relative to robot base"""
        obj_pos, obj_ori = object_prim.get_world_poses()
        base_pos, base_ori = self.robot.get_world_poses()

        # Calculate relative position
        relative_pos = obj_pos - base_pos

        return relative_pos

    def compute_reward(self, action):
        """Compute reward for manipulation task"""
        # Example reward function for reaching and grasping
        ee_pos = self.robot.get_end_effector_position()
        obj_pos = self.objects[0].get_world_poses()[0]  # Target object

        # Distance to object
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)

        # Reward shaping
        reward = 0

        # Reach reward - encourage getting closer to object
        reward -= dist_to_obj * 0.1

        # Bonus for getting very close
        if dist_to_obj < 0.05:
            reward += 1.0

        # Check if grasping is successful
        if self.is_grasp_successful():
            reward += 10.0  # Large bonus for successful grasp

        # Penalty for large actions (smooth movement)
        reward -= np.sum(np.square(action)) * 0.001

        return reward

    def is_grasp_successful(self):
        """Check if grasp is successful"""
        # This would check contact sensors, object motion, etc.
        # Simplified example:
        left_contact = self.check_contact("left_finger")
        right_contact = self.check_contact("right_finger")
        object_moving = self.is_object_moving()

        return left_contact and right_contact and not object_moving

    def check_contact(self, finger_name):
        """Check if finger is in contact"""
        # This would use contact sensor data
        return True  # Simplified

    def is_object_moving(self):
        """Check if object is moving after grasp"""
        # This would check object velocity
        return False  # Simplified

    def reset(self):
        """Reset environment to initial state"""
        # Reset robot to initial configuration
        initial_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Example
        self.robot.set_joint_positions(initial_joints)

        # Reset object positions
        for i, obj in enumerate(self.objects):
            obj.set_world_poses(
                positions=np.array([[0.5 + i*0.1, 0.0, 0.1]]),
                orientations=np.array([[0.0, 0.0, 0.0, 1.0]])
            )

        # Return initial state
        return self.get_state()

    def step(self, action):
        """Execute action and return transition"""
        # Apply action to robot
        self.robot.set_joint_positions(action)

        # Step simulation
        self.robot.world.step(render=True)

        # Get next state
        next_state = self.get_state()

        # Compute reward
        reward = self.compute_reward(action)

        # Check if episode is done
        done = self.is_episode_done()

        # Additional info
        info = {
            'success': self.is_grasp_successful(),
            'distance': np.linalg.norm(
                self.robot.get_end_effector_position() -
                self.objects[0].get_world_poses()[0]
            )
        }

        return next_state, reward, done, info

    def is_episode_done(self):
        """Check if episode is done"""
        # Episode could end if:
        # - Successful task completion
        # - Max steps reached
        # - Robot violates constraints
        return False  # Simplified
```

## Practical Implementation Example

### Complete Manipulation Task

Here's a complete example of implementing a pick-and-place task:

```python
def run_manipulation_demo():
    """Run a complete manipulation demonstration"""
    # Initialize Isaac environment
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims import RigidPrim

    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_ground_plane("/World/Ground", static_friction=0.5, dynamic_friction=0.5)

    # Add robot
    robot = world.scene.add(
        Robot(
            prim_path="/World/Robot",
            usd_path="/Isaac/Robots/Franka/franka_instanceable.usd",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    )

    # Add objects to manipulate
    cube = world.scene.add(
        RigidPrim(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0.5, 0.0, 0.1]),
            scale=np.array([0.05, 0.05, 0.05]),
            mass=0.1
        )
    )

    # Add target location
    target = world.scene.add(
        RigidPrim(
            prim_path="/World/Target",
            name="target",
            position=np.array([0.7, 0.3, 0.1]),
            scale=np.array([0.1, 0.1, 0.01]),
            mass=0.0
        )
    )

    # Reset world
    world.reset()

    # Initialize manipulation controller
    manipulator = AIManipulationController(robot)
    motion_controller = MotionController(robot)

    # Perform pick and place task
    print("Starting pick and place task...")

    # 1. Plan grasp for the cube
    cube_position = cube.get_world_poses()[0]
    grasp_pose = {
        'position': cube_position + np.array([0, 0, 0.1]),  # Above cube
        'orientation': np.array([0, 0, 0, 1])
    }

    print("Executing grasp...")
    manipulator.execute_grasp(grasp_pose)

    # 2. Move to target location
    target_position = target.get_world_poses()[0]
    place_pose = {
        'position': target_position + np.array([0, 0, 0.2]),  # Above target
        'orientation': np.array([0, 0, 0, 1])
    }

    print("Moving to target...")
    motion_controller.move_to_pose(place_pose)

    # 3. Release object
    print("Releasing object...")
    manipulator.open_gripper()

    print("Pick and place task completed!")

    # Wait for a moment to observe result
    for _ in range(100):
        world.step(render=True)

# Run the demonstration
if __name__ == "__main__":
    run_manipulation_demo()
```

## Best Practices for Isaac Manipulation

### 1. Physics Parameter Tuning
- Use appropriate friction and restitution values for objects
- Tune solver parameters for stable contact simulation
- Set proper collision margins for accurate contact detection

### 2. Control Strategy Selection
- Use position control for precision tasks
- Use force control for compliant tasks
- Use hybrid force-position control for contact-rich tasks

### 3. Sensor Integration
- Combine multiple sensor modalities for robust perception
- Use realistic sensor noise models
- Implement sensor fusion techniques

### 4. Learning Approaches
- Use simulation-to-real transfer techniques
- Implement curriculum learning for complex tasks
- Use domain randomization for robust policies

## Learning Objectives

By the end of this lesson, you should be able to:
- Configure robots in Isaac for manipulation tasks with appropriate physics parameters
- Implement grasp planning using both physics simulation and AI techniques
- Design and execute motion control strategies for manipulation
- Apply force and impedance control for compliant manipulation
- Implement learning-based approaches for dexterous manipulation
- Evaluate manipulation performance and optimize control strategies
- Integrate perception and control for complete manipulation pipelines
- Understand the trade-offs between different manipulation control approaches