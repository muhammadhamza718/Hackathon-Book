---
title: 'Using Gazebo with ROS 2'
description: 'Learn to use Gazebo with ROS 2, detailing launching Gazebo with ROS 2, spawning robots (URDF), interfacing with simulated sensors/controllers'
chapter: 5
lesson: 2
module: 2
sidebar_label: 'Using Gazebo with ROS 2'
sidebar_position: 2
tags: ['Gazebo', 'ROS 2', 'Robot Spawning', 'Sensors', 'Controllers']
keywords: ['Gazebo', 'ROS 2', 'robot spawning', 'URDF', 'sensors', 'controllers', 'simulation']
---

# Using Gazebo with ROS 2

## Overview

Integrating Gazebo with ROS 2 enables seamless simulation of robotic systems, allowing developers to test algorithms, validate controllers, and train systems in a realistic virtual environment. This lesson covers the practical aspects of launching Gazebo with ROS 2, spawning robots defined in URDF, and interfacing with simulated sensors and controllers.

## Launching Gazebo with ROS 2

### Basic Launch Structure

The integration between Gazebo and ROS 2 typically involves launch files that coordinate:

1. Starting the Gazebo simulation environment
2. Loading robot models into the simulation
3. Launching ROS 2 nodes for control and communication
4. Setting up bridges between Gazebo and ROS 2

### ROS 2 Launch Files for Gazebo

A typical launch file for Gazebo integration:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')

    # Declare launch arguments
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/my_robot_gazebo/worlds`'
    )

    # Start Gazebo with specified world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world,
            'paused': 'false',
            'use_sim_time': use_sim_time,
        }.items()
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        declare_world_arg,
        gazebo,
    ])
```

### Launching Gazebo with Custom World

To launch Gazebo with a custom world file:

```bash
# Launch with default empty world
ros2 launch my_robot_gazebo empty_world.launch.py

# Launch with custom world
ros2 launch my_robot_gazebo custom_world.launch.py world:=my_world.sdf

# Launch with simulation time enabled
ros2 launch my_robot_gazebo empty_world.launch.py use_sim_time:=true
```

## Spawning Robots in Gazebo

### Robot Description Parameter

Before spawning a robot, the robot description must be available as a ROS parameter:

```python
def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("my_robot_description"), "urdf", "my_robot.urdf.xacro"]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        # Set robot description parameter
        DeclareParameter(
            'robot_description',
            ParameterValue(robot_description_content, value_type=str)
        ),
        spawn_entity,
    ])
```

### Complete Robot Spawning Example

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    # Get URDF via xacro
    robot_description_content = Command(
        [
            'xacro ',
            os.path.join(get_package_share_directory('my_robot_description'), 'urdf', 'my_robot.urdf.xacro')
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        )
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )

    # Joint state publisher (for non-controlled joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to start RViz'
        ),
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity,
    ])
```

## Interfacing with Simulated Sensors

### Camera Sensors

Gazebo can simulate various camera types that publish to ROS 2 topics:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
      <update_rate>30</update_rate>
      <topic_name>image_raw</topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Sensors

Simulating LiDAR sensors in Gazebo:

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensors

Configuring IMU sensors in Gazebo:

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

## Controlling Robots in Gazebo

### Differential Drive Controller

Setting up a differential drive controller:

```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="left_wheel_joint">
    <command_interface name="velocity">
      <param name="min">-1.0</param>
      <param name="max">1.0</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
  <joint name="right_wheel_joint">
    <command_interface name="velocity">
      <param name="min">-1.0</param>
      <param name="max">1.0</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
</ros2_control>

<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_control)/config/diff_drive.yaml</parameters>
  </plugin>
</gazebo>
```

### Joint Position Controllers

Configuring joint position controllers:

```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="arm_joint">
    <command_interface name="position"/>
    <command_interface name="velocity"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>
</ros2_control>
```

### Controller Configuration File

Example controller configuration (YAML):

```yaml
controller_manager:
  ros__parameters:
    use_sim_time: true
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    diff_drive_base_controller:
      type: diff_drive_controller/DiffDriveController

diff_drive_base_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]

    wheel_separation: 0.3
    wheel_radius: 0.075

    use_stamped_vel: false
    publish_rate: 50.0
    odom_frame_id: odom
    base_frame_id: base_link
    pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
```

## Gazebo-ROS Communication Bridge

### Topic Bridge

The `ros_gz_bridge` facilitates communication between Gazebo and ROS 2:

```bash
# Bridge laser scan data
ros2 run ros_gz_bridge parameter_bridge /scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan

# Bridge camera data
ros2 run ros_gz_bridge parameter_bridge /camera/image@sensor_msgs/msg/Image[ignition.msgs.Image

# Bridge velocity commands
ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist
```

### Example Bridge Configuration

In a launch file:

```python
# Bridge for sensor data
bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
        '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
        '/tf@tf2_msgs/msg/TFMessage[tf2_msgs/msg/TFMessage',
        '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        '/odom@nav_msgs/msg/Odometry]ignition.msgs.Odometry'
    ],
    remappings=[
        ('/scan', '/my_robot/scan'),
        ('/camera/image', '/my_robot/camera/image_raw'),
        ('/cmd_vel', '/my_robot/cmd_vel'),
    ],
    parameters=[{'use_sim_time': use_sim_time}]
)
```

## Debugging and Troubleshooting

### Common Issues

#### Issue: Robot not spawning in Gazebo
**Solution**: Check that the robot_description parameter is properly set and the URDF is valid.

#### Issue: Controllers not responding
**Solution**: Verify controller configuration files and ensure the controller manager is running.

#### Issue: Sensor data not publishing
**Solution**: Check Gazebo plugin configurations and topic remappings.

### Debugging Commands

```bash
# Check available topics
ros2 topic list

# Check robot state
ros2 run joint_state_publisher_gui joint_state_publisher_gui

# Check TF tree
ros2 run tf2_tools view_frames

# Echo sensor data
ros2 topic echo /scan sensor_msgs/msg/LaserScan
```

## Best Practices

### 1. Parameter Management
- Use launch arguments for configurable parameters
- Set `use_sim_time: true` for simulation
- Organize parameters in YAML files

### 2. Resource Management
- Limit simulation update rates appropriately
- Use efficient collision geometries
- Configure sensors with appropriate rates

### 3. Modular Design
- Separate world files from robot models
- Use modular launch files
- Organize controllers in separate configuration files

## Learning Objectives

By the end of this lesson, you should be able to:
- Launch Gazebo with ROS 2 integration
- Spawn robots defined in URDF into Gazebo
- Configure and interface with simulated sensors
- Set up controllers for robot simulation
- Establish communication bridges between Gazebo and ROS 2
- Troubleshoot common issues in Gazebo-ROS integration