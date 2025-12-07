---
title: 'Understanding and Creating URDF Models'
description: 'Learn to create URDF models for robots, explaining URDF structure (links, joints, properties), providing examples, and visualization with RViz'
chapter: 3
lesson: 1
module: 1
sidebar_label: 'Understanding and Creating URDF Models'
sidebar_position: 1
tags: ['URDF', 'Robot Modeling', 'Links', 'Joints', 'RViz']
keywords: ['URDF', 'robot modeling', 'links', 'joints', 'visualization', 'RViz']
---

# Understanding and Creating URDF Models

## Overview

Unified Robot Description Format (URDF) is the standard XML-based format used in ROS for describing robot models. URDF allows you to define the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and their relationships in a kinematic chain. This lesson will guide you through the fundamentals of creating URDF models for robots.

## What is URDF?

URDF (Unified Robot Description Format) is an XML-based format that describes robot models in ROS. It defines the physical and visual properties of a robot, including:

- **Links**: Rigid parts of the robot (e.g., base, arms, wheels)
- **Joints**: Connections between links that allow relative motion
- **Visual properties**: How the robot appears in simulation and visualization
- **Collision properties**: How the robot interacts with the environment in simulation
- **Inertial properties**: Mass, center of mass, and moments of inertia for physics simulation

URDF is essential for:
- Robot simulation in Gazebo
- Robot visualization in RViz
- Kinematic calculations
- Motion planning

## Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links definition -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints definition -->
  <joint name="joint_name" type="fixed">
    <parent link="base_link"/>
    <child link="another_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
</robot>
```

### Robot Element
The root element of every URDF file is the `<robot>` tag, which must have a `name` attribute that uniquely identifies the robot.

### Link Elements
A link represents a rigid part of the robot. Each link can contain:

- **Visual**: How the link appears in visualization tools like RViz
- **Collision**: How the link interacts with the environment in simulation
- **Inertial**: Physical properties for physics simulation
- **Material**: Color and appearance properties

### Joint Elements
A joint connects two links and defines their relative motion. Key attributes include:

- **name**: Unique identifier for the joint
- **type**: Type of motion (fixed, revolute, continuous, prismatic, etc.)
- **parent**: The link that serves as the parent in the kinematic tree
- **child**: The link that serves as the child in the kinematic tree
- **origin**: Position and orientation of the joint relative to the parent link

## Link Properties

### Visual Elements
The visual element defines how a link appears in visualization and simulation:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <cylinder length="0.6" radius="0.2"/>
  </geometry>
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
</visual>
```

#### Geometry Types
URDF supports several geometry types:
- **Box**: Defined by `size="x y z"`
- **Cylinder**: Defined by `radius` and `length`
- **Sphere**: Defined by `radius`
- **Mesh**: Defined by a file path using `package://` notation

#### Materials
Materials define the appearance of links:
- **Color**: RGBA values (Red, Green, Blue, Alpha)
- **Texture**: Optional texture mapping
- **Name**: Reference to reusable material definitions

### Collision Elements
The collision element defines the physical shape of the link for collision detection:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <cylinder length="0.6" radius="0.2"/>
  </geometry>
</collision>
```

Collision geometry is often simpler than visual geometry for performance reasons.

### Inertial Elements
The inertial element defines physical properties for simulation:

```xml
<inertial>
  <mass value="10"/>
  <inertia ixx="1e-3" ixy="0.0" ixz="0.0" iyy="1e-3" iyz="0.0" izz="1e-3"/>
</inertial>
```

## Joint Types

URDF supports several joint types:

### Fixed Joint
A fixed joint connects two links with no relative motion:
```xml
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
</joint>
```

### Revolute Joint
A revolute joint allows rotation around a single axis with limited range:
```xml
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

### Continuous Joint
A continuous joint allows unlimited rotation around an axis:
```xml
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

### Prismatic Joint
A prismatic joint allows linear translation along an axis:
```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

## URDF Examples

### Simple Single-Link Robot
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1e-3" ixy="0.0" ixz="0.0" iyy="1e-3" iyz="0.0" izz="1e-3"/>
    </inertial>
  </link>
</robot>
```

### Multi-Link Robot with Joints
```xml
<?xml version="1.0"?>
<robot name="multi_link_robot">
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Leg link -->
  <link name="leg_link">
    <visual>
      <geometry>
        <box size="0.6 0.1 0.2"/>
      </geometry>
      <origin rpy="0 1.57075 0" xyz="0 0 -0.3"/>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.6 0.1 0.2"/>
      </geometry>
      <origin rpy="0 1.57075 0" xyz="0 0 -0.3"/>
    </collision>
  </link>

  <!-- Joint connecting base and leg -->
  <joint name="base_to_leg" type="fixed">
    <parent link="base_link"/>
    <child link="leg_link"/>
    <origin xyz="0 -0.22 0.25"/>
  </joint>
</robot>
```

## Visualizing URDF Models with RViz

### Launching URDF in RViz
To visualize a URDF model in RViz, you can use the URDF tutorial package:

```bash
ros2 launch urdf_tutorial display.launch.py model:=urdf/your_model.urdf
```

### Robot State Publisher
The `robot_state_publisher` node is responsible for publishing the robot's joint states and transforms. It reads the URDF and publishes the static transforms between links based on the joint positions.

## Best Practices for URDF Creation

### 1. Start Simple
Begin with a basic model and gradually add complexity. Start with visual elements before adding collision and inertial properties.

### 2. Use Meaningful Names
Use descriptive names for links and joints that reflect their function in the robot.

### 3. Organize the Kinematic Tree
Plan your robot's kinematic structure carefully. Each link (except the base) should have exactly one parent.

### 4. Consider Performance
Use simple collision geometries (boxes, cylinders, spheres) when possible to improve simulation performance.

### 5. Validate Your URDF
Check your URDF for errors using tools like `check_urdf`:
```bash
check_urdf your_robot.urdf
```

### 6. Use Standard Units
- Length: meters
- Mass: kilograms
- Angles: radians
- Time: seconds

## Common URDF Issues and Solutions

### Issue: Robot appears as a single shape
**Solution**: Check that joints are properly defined with correct parent-child relationships.

### Issue: Robot parts are not visible
**Solution**: Ensure visual elements are properly defined and materials are specified.

### Issue: Robot falls through the ground in simulation
**Solution**: Check that collision elements are properly defined for the base link.

### Issue: Joint limits not working
**Solution**: Verify that joint limits are correctly specified with proper units.

## Learning Objectives

By the end of this lesson, you should be able to:
- Understand the structure and components of URDF files
- Create basic robot models with links and fixed joints
- Define visual, collision, and inertial properties for links
- Understand how to visualize URDF models in RViz
- Apply best practices for URDF creation
- Troubleshoot common URDF issues