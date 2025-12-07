---
title: 'ROS 2 Launch Files'
description: 'Learn about ROS 2 launch files, their structure, and how to create and use them to manage multiple nodes and configurations'
chapter: 4
lesson: 2
module: 1
sidebar_label: 'ROS 2 Launch Files'
sidebar_position: 2
tags: ['ROS 2', 'Launch Files', 'Node Management', 'Configuration', 'Python Launch']
keywords: ['ROS 2', 'launch files', 'node management', 'configuration', 'Python launch', 'system orchestration']
---

# ROS 2 Launch Files

## Overview

ROS 2 launch files provide a powerful way to start multiple nodes with specific configurations simultaneously. They allow you to define complex robot systems with a single command, manage parameters, handle dependencies, and orchestrate the startup sequence of your robot's software stack. This lesson covers the structure, syntax, and best practices for creating and using ROS 2 launch files.

## What are Launch Files?

Launch files in ROS 2 are scripts that define and start multiple nodes with specific configurations. They serve several important purposes:

- **Convenience**: Start multiple nodes with a single command
- **Configuration**: Set parameters and configurations for nodes
- **Orchestration**: Control the startup order and dependencies
- **Reusability**: Define common system configurations that can be reused
- **Modularity**: Break down complex systems into manageable components

## Launch File Formats

ROS 2 supports multiple launch file formats, but Python is the most commonly used and recommended approach.

### Python Launch Files (Recommended)

Python launch files offer the most flexibility and are the recommended approach:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop'
        )
    ])
```

### XML Launch Files

XML launch files provide a declarative approach:

```xml
<launch>
  <node pkg="turtlesim" exec="turtlesim_node" name="sim"/>
  <node pkg="turtlesim" exec="turtle_teleop_key" name="teleop"/>
</launch>
```

### YAML Launch Files

YAML launch files offer another declarative option:

```yaml
launch:
  - node:
      pkg: "turtlesim"
      exec: "turtlesim_node"
      name: "sim"
  - node:
      pkg: "turtlesim"
      exec: "turtle_teleop_key"
      name: "teleop"
```

## Basic Python Launch File Structure

### Required Imports

Every Python launch file needs these basic imports:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
```

### The generate_launch_description Function

This is the entry point for the launch file:

```python
def generate_launch_description():
    return LaunchDescription([
        # List of launch actions here
    ])
```

### Basic Node Launch Action

The most common launch action is starting a node:

```python
Node(
    package='package_name',
    executable='executable_name',
    name='node_name',  # Optional, defaults to executable name
    namespace='namespace',  # Optional
    parameters=[{'param1': 'value1'}],  # Optional
    remappings=[('original_topic', 'new_topic')],  # Optional
    arguments=['arg1', 'arg2'],  # Optional
    output='screen'  # Optional: 'log', 'screen', or 'both'
)
```

## Advanced Launch Features

### Parameters

You can set parameters for nodes in several ways:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navigation2',
            executable='nav2_amcl',
            name='amcl',
            parameters=[
                # Direct parameter specification
                {'use_sim_time': True},
                # Load from YAML file
                PathJoinSubstitution([
                    FindPackageShare('my_robot_bringup'),
                    'config',
                    'amcl_config.yaml'
                ])
            ]
        )
    ])
```

### Remappings

Remap topics between nodes:

```python
Node(
    package='image_proc',
    executable='rectify',
    name='image_rect',
    remappings=[
        ('image', '/camera/image_raw'),
        ('camera_info', '/camera/camera_info'),
        ('image_rect', '/camera/image_rect')
    ]
)
```

### Arguments and Substitutions

Launch files can accept arguments and use substitutions:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        Node(
            package='my_robot_control',
            executable='controller',
            name='controller',
            parameters=[{'use_sim_time': use_sim_time}]
        )
    ])
```

### Conditional Launch Actions

Execute actions based on conditions:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare argument
    use_rviz = LaunchConfiguration('use_rviz')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    return LaunchDescription([
        declare_use_rviz_cmd,
        # Start RViz only if use_rviz is true
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            condition=IfCondition(use_rviz)
        )
    ])
```

## Common Launch Actions

### Including Other Launch Files

Include other launch files to build modular systems:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include another launch file
    turtlebot3_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_bringup'),
                'launch',
                'robot.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        turtlebot3_bringup_launch,
        # Additional nodes can be added here
    ])
```

### Setting Environment Variables

Set environment variables for launched nodes:

```python
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    return LaunchDescription([
        # Set environment variable
        SetEnvironmentVariable(name='RCUTILS_LOGGING_SEVERITY_THRESHOLD', value='INFO'),
        Node(
            package='my_package',
            executable='my_node',
            name='my_node'
        )
    ])
```

### Executing Shell Commands

Execute shell commands as part of the launch process:

```python
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Execute a shell command
        ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/my_node', 'param_name', 'param_value'],
            output='screen'
        ),
        Node(
            package='my_package',
            executable='my_node',
            name='my_node'
        )
    ])
```

## Complex Launch File Example

Here's a comprehensive example showing multiple advanced features:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    slam = LaunchConfiguration('slam')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    declare_slam_cmd = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Whether to run SLAM'
    )

    # Nodes
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/joint_states', 'demo_joint_states'),
        ]
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/joint_states', 'demo_joint_states'),
        ]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_slam_cmd)

    # Add nodes
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)

    # Add conditional nodes
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('my_robot_description'),
            'rviz',
            'view_robot.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )
    ld.add_action(rviz_node)

    # Add SLAM node if enabled
    slam_node = Node(
        condition=IfCondition(slam),
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_bringup'),
                'config',
                'slam.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ]
    )
    ld.add_action(slam_node)

    return ld
```

## Launch File Organization

### Package Structure

A well-organized package structure for launch files:

```
my_robot_bringup/
├── launch/
│   ├── robot.launch.py
│   ├── navigation.launch.py
│   └── simulation.launch.py
├── config/
│   ├── robot.yaml
│   └── navigation.yaml
├── rviz/
│   └── view_robot.rviz
└── package.xml
```

### Installation

To make launch files available in your package, add to `setup.py`:

```python
import os
from glob import glob
from setuptools import setup

package_name = 'my_robot_bringup'

setup(
    # ... other setup parameters ...
    data_files=[
        # ... other data files ...
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
    ],
)
```

Or in `CMakeLists.txt`:

```cmake
install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}/
)
```

## Running Launch Files

### From Command Line

Launch files can be run from the command line:

```bash
# Run a launch file from a package
ros2 launch my_package my_launch_file.py

# With arguments
ros2 launch my_package my_launch_file.py use_sim_time:=true

# With multiple arguments
ros2 launch my_package my_launch_file.py use_sim_time:=true use_rviz:=false
```

### Launch File Arguments

Arguments can be passed to launch files:

```bash
# Boolean argument
ros2 launch my_package launch_file.py enable_viz:=true

# Numeric argument
ros2 launch my_package launch_file.py rate:=10.0

# String argument
ros2 launch my_package launch_file.py robot_name:=turtlebot3
```

## Launch System Best Practices

### 1. Modular Design
- Break complex systems into smaller, reusable launch files
- Use `IncludeLaunchDescription` to compose systems
- Keep launch files focused on specific functionality

### 2. Parameter Management
- Use YAML files for complex parameter configurations
- Group related parameters logically
- Use launch arguments for configurable options

### 3. Naming Conventions
- Use descriptive names for nodes and parameters
- Follow consistent naming patterns
- Use snake_case for launch file names

### 4. Error Handling
- Provide clear error messages
- Use conditions to handle optional components
- Validate input arguments when possible

### 5. Documentation
- Comment complex launch files
- Document launch arguments and their purposes
- Include usage examples

## Launch File Debugging

### Verbose Output
Get detailed information about launch execution:

```bash
ros2 launch -v my_package my_launch_file.py
```

### Check Launch File Syntax
Verify launch file syntax without executing:

```bash
python3 path/to/launch_file.py
```

### Common Issues and Solutions

#### Issue: Launch file not found
**Solution**: Check package installation and launch file path.

#### Issue: Node fails to start
**Solution**: Check package dependencies and node executable permissions.

#### Issue: Parameters not loaded
**Solution**: Verify parameter file paths and YAML syntax.

#### Issue: Nodes can't communicate
**Solution**: Check namespaces, remappings, and network configuration.

## Advanced Launch Concepts

### Launch Substitutions

Substitutions allow dynamic value insertion:

```python
from launch.substitutions import TextSubstitution, EnvironmentVariable
from launch.actions import LogInfo

def generate_launch_description():
    return LaunchDescription([
        LogInfo(
            msg=["Launch time: ", TextSubstitution(text=str(time.time()))]
        ),
        LogInfo(
            msg=["Home directory: ", EnvironmentVariable(name='HOME')]
        )
    ])
```

### Event Handling

Handle events during launch execution:

```python
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    node1 = Node(
        package='my_package',
        executable='node1',
        name='node1'
    )

    node2 = Node(
        package='my_package',
        executable='node2',
        name='node2'
    )

    # Start node2 only after node1 starts
    delayed_node2 = RegisterEventHandler(
        OnProcessStart(
            target_action=node1,
            on_start=[node2],
        )
    )

    return LaunchDescription([
        node1,
        delayed_node2
    ])
```

## Learning Objectives

By the end of this lesson, you should be able to:
- Create and structure ROS 2 launch files using Python
- Configure nodes with parameters, remappings, and arguments
- Use advanced launch features like conditions and substitutions
- Organize launch files in a modular, reusable way
- Debug common launch file issues
- Apply best practices for launch file development
- Understand the launch system architecture and event handling