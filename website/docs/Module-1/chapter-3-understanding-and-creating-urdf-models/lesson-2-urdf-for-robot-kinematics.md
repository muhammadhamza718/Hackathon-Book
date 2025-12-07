---
title: 'URDF for Robot Kinematics'
description: 'Understand how URDF defines robot kinematics, explaining URDF joint types, degrees of freedom, forward/inverse kinematics'
chapter: 3
lesson: 2
module: 1
sidebar_label: 'URDF for Robot Kinematics'
sidebar_position: 2
tags: ['URDF', 'Kinematics', 'Joints', 'Degrees of Freedom', 'Forward Kinematics', 'Inverse Kinematics']
keywords: ['URDF', 'kinematics', 'joint types', 'degrees of freedom', 'forward kinematics', 'inverse kinematics']
---

# URDF for Robot Kinematics

## Overview

Robot kinematics is the study of motion in robotic systems, describing the relationship between joint positions and the position and orientation of the robot's end-effector. URDF plays a crucial role in defining the kinematic structure of robots by specifying the links, joints, and their physical relationships. This lesson explores how URDF defines robot kinematics and how different joint types contribute to the robot's degrees of freedom and motion capabilities.

## Understanding Robot Kinematics

### Forward Kinematics
Forward kinematics is the process of calculating the position and orientation of the robot's end-effector based on the known joint angles. In URDF, this is determined by:

1. The kinematic chain structure defined by parent-child relationships
2. Joint types and their allowed motions
3. Joint positions and orientations
4. Link dimensions and transformations

### Inverse Kinematics
Inverse kinematics is the reverse process: calculating the required joint angles to achieve a desired end-effector position and orientation. URDF provides the geometric foundation needed for inverse kinematics solvers to compute these solutions.

## Joint Types and Their Kinematic Properties

### Fixed Joints
Fixed joints create rigid connections between links with no relative motion:

```xml
<joint name="fixed_connection" type="fixed">
  <parent link="link_a"/>
  <child link="link_b"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>
```

**Kinematic Impact:**
- No degrees of freedom
- Maintains constant spatial relationship between links
- Forms the structural backbone of the robot

### Revolute Joints
Revolute joints allow rotation around a single axis with limited range:

```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
</joint>
```

**Kinematic Impact:**
- 1 degree of freedom
- Rotation constrained to a specific axis
- Limited by joint limits
- Common in robotic arms and legs

### Continuous Joints
Continuous joints allow unlimited rotation around an axis:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base"/>
  <child link="rotating_part"/>
  <axis xyz="0 0 1"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

**Kinematic Impact:**
- 1 degree of freedom
- Unlimited rotation around the axis
- No joint limits (theoretically)
- Common for rotating bases and wheels

### Prismatic Joints
Prismatic joints allow linear translation along a single axis:

```xml
<joint name="slider_joint" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="0.5"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

**Kinematic Impact:**
- 1 degree of freedom
- Linear motion constrained to a specific axis
- Limited by linear joint limits
- Common in linear actuators and telescoping mechanisms

### Other Joint Types
- **Planar**: 2 degrees of freedom (translation in a plane)
- **Floating**: 6 degrees of freedom (3 translation, 3 rotation)

## Degrees of Freedom (DOF)

### Calculating DOF
The total degrees of freedom of a robot is the sum of the DOF of all its joints:

```
Total DOF = Σ (DOF of each joint)
```

For example:
- A robot with 3 revolute joints: 3 DOF
- A robot with 2 revolute and 1 prismatic joint: 3 DOF
- A robot with 6 continuous joints: 6 DOF

### Types of DOF
- **Actuated DOF**: Joints that can be actively controlled
- **Passive DOF**: Joints that move due to external forces or constraints
- **Redundant DOF**: More DOF than necessary to perform a task

### DOF and Workspace
- More DOF generally means a larger workspace
- 6 DOF is typically required for full position and orientation control
- Redundant DOF can provide multiple solutions to the same pose

## URDF Structure for Kinematic Chains

### Kinematic Tree Structure
URDF defines a kinematic tree where:
- Each link (except the base) has exactly one parent
- The base link is the root of the tree
- Joints connect parent and child links
- The tree structure determines the kinematic relationships

### Example: Simple 3-DOF Robotic Arm
```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Shoulder link -->
  <link name="shoulder_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- Elbow link -->
  <link name="elbow_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- End-effector link -->
  <link name="end_effector">
    <visual>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
    </visual>
  </link>

  <!-- Shoulder joint -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Elbow joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child link="elbow_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Wrist joint -->
  <joint name="wrist_joint" type="revolute">
    <parent link="elbow_link"/>
    <child link="end_effector"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>
</robot>
```

## Forward Kinematics in URDF

### Transformation Matrices
Each joint defines a transformation from the parent link frame to the child link frame. These transformations are calculated using:

- Joint position (from joint limits and current angle)
- Joint origin (fixed offset)
- Joint axis (direction of motion)

### Chain Calculations
For forward kinematics, transformations are multiplied along the kinematic chain:

```
T_end_effector = T_base_to_joint1 * T_joint1_to_link2 * T_link2_to_joint2 * ...
```

### URDF's Role in Forward Kinematics
URDF provides the geometric and structural information needed for forward kinematics:
- Link lengths and dimensions
- Joint positions and orientations
- Joint axis directions
- Kinematic chain structure

## Inverse Kinematics in URDF

### Kinematic Solvers
URDF is used by various inverse kinematics solvers including:
- KDL (Kinematics and Dynamics Library)
- TRAC-IK
- MoveIt! IK solvers

### Information Required for IK
- Joint limits and types
- Kinematic chain structure
- Link dimensions and offsets
- End-effector target pose

### Example: IK Configuration
```xml
<joint name="joint1" type="revolute">
  <parent link="base_link"/>
  <child link="link1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="100" velocity="2"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <!-- Kinematic properties for IK solvers -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Joint Dynamics and Physical Properties

### Joint Dynamics
The dynamics element specifies physical properties of joints:

```xml
<dynamics damping="0.1" friction="0.0"/>
```

- **Damping**: Resistance to motion (affects simulation)
- **Friction**: Static friction at the joint (affects simulation)

### Joint Limits
Joint limits are crucial for both kinematics and safety:

```xml
<limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
```

- **Lower/Upper**: Position limits (radians for revolute, meters for prismatic)
- **Effort**: Maximum force/torque (N for prismatic, Nm for revolute)
- **Velocity**: Maximum velocity (m/s for prismatic, rad/s for revolute)

## Kinematic Chains and End-Effector Control

### Chain Definition
In URDF, kinematic chains are defined implicitly through the parent-child relationships:

```xml
<!-- Base -> Joint1 -> Link1 -> Joint2 -> Link2 -> ... -> End Effector -->
```

### Multiple Chains
Robots can have multiple kinematic chains:
- Parallel manipulators
- Mobile manipulators
- Humanoid robots with multiple arms/legs

### Kinematic Base and Tip
For kinematic calculations, it's important to identify:
- **Base**: The starting link of the kinematic chain
- **Tip**: The end link of the kinematic chain (often the end-effector)

## URDF and Kinematic Libraries

### Integration with MoveIt!
MoveIt! uses URDF to:
- Generate kinematic models
- Perform collision checking
- Plan motion trajectories
- Control robot motion

### Robot State Publisher
The robot_state_publisher node uses URDF to:
- Publish joint transforms
- Calculate forward kinematics
- Update robot visualization

## Practical Considerations

### Joint Configuration
When designing kinematic chains:
- Consider the workspace requirements
- Account for joint limits
- Plan for singularities
- Balance DOF with control complexity

### Simulation vs. Real Robot
Remember that URDF defines the idealized model:
- Real robots have tolerances and flexibilities
- Actuator limitations may affect performance
- Calibration may be needed for accuracy

### Common Kinematic Issues
- **Singularities**: Configurations where the robot loses DOF
- **Joint limits**: Positions that exceed physical constraints
- **Collision**: Self-collision or environment collision
- **Dexterity**: Ability to reach desired orientations

## Learning Objectives

By the end of this lesson, you should be able to:
- Explain how URDF defines robot kinematics through joint types and relationships
- Understand the kinematic properties of different joint types
- Calculate degrees of freedom for robot mechanisms
- Describe the relationship between URDF structure and forward/inverse kinematics
- Identify key kinematic elements in URDF files
- Recognize the role of URDF in kinematic solvers and robot control