---
title: 'Gazebo Plugins and World Files'
description: 'Understanding and creating Gazebo world files and plugins for sensors/controllers'
chapter: 5
lesson: 3
module: 2
sidebar_label: 'Gazebo Plugins and World Files'
sidebar_position: 3
tags: ['Gazebo', 'Plugins', 'World Files', 'Sensors', 'Controllers']
keywords: ['Gazebo', 'plugins', 'world files', 'sensors', 'controllers', 'simulation']
---

# Gazebo Plugins and World Files

## Overview

Gazebo's extensibility is achieved through its plugin architecture and customizable world files. Plugins allow you to extend Gazebo's functionality, while world files define the simulation environment. This lesson covers creating custom world files and developing plugins for sensors and controllers that integrate seamlessly with ROS 2.

## Gazebo World Files

### World File Structure

World files in Gazebo are written in SDF (Simulation Description Format) and define the complete simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- World properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>

    <!-- Environment lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Models in the world -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <!-- Static objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <static>true</static>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Physics Engine Configuration

World files allow configuration of the physics engine:

```xml
<physics type="ode">
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>

  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Environment and Atmosphere

Configure environmental properties:

```xml
<!-- Atmosphere -->
<atmosphere type="adiabatic">
  <temperature>288.15</temperature>
  <pressure>101325</pressure>
</atmosphere>

<!-- Magnetic field -->
<magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>

<!-- Wind -->
<wind>
  <linear_velocity>0.5 0 0</linear_velocity>
</wind>
```

### Models and Objects

World files can include models in several ways:

```xml
<!-- Include from model database -->
<include>
  <uri>model://ground_plane</uri>
  <pose>0 0 0 0 0 0</pose>
</include>

<!-- Include from local model -->
<include>
  <uri>model://my_custom_model</uri>
  <pose>1 2 0.5 0 0 1.57</pose>
</include>

<!-- Define model inline -->
<model name="obstacle_1">
  <pose>5 5 0.5 0 0 0</pose>
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.2</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.2</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>1 0 0 1</ambient>
        <diffuse>1 0 0 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

## Gazebo Plugins

### Plugin Types

Gazebo supports several types of plugins:

- **World plugins**: Modify world behavior and properties
- **Model plugins**: Control specific models and robots
- **Sensor plugins**: Process sensor data and publish messages
- **System plugins**: Extend core Gazebo functionality
- **Visual plugins**: Add visualization elements

### Creating a Custom World Plugin

Here's an example of a custom world plugin that adds dynamic elements to the simulation:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>

namespace gazebo
{
  class DynamicWorldPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      this->world = _world;

      // Parse parameters from SDF
      if (_sdf->HasElement("update_rate"))
        this->updateRate = _sdf->Get<double>("update_rate");
      else
        this->updateRate = 100.0;

      // Connect to pre-update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&DynamicWorldPlugin::OnUpdate, this));

      gzmsg << "DynamicWorldPlugin loaded with update rate: "
            << this->updateRate << " Hz\n";
    }

    public: void OnUpdate()
    {
      // Add dynamic behavior to the world
      // For example, move objects, change lighting, etc.

      // Get all models in the world
      physics::Model_V models = this->world->Models();

      for (auto& model : models)
      {
        // Example: Apply a small random force to each model
        if (!model->IsStatic())
        {
          ignition::math::Vector3d force(
            (rand() % 100 - 50) * 0.001,
            (rand() % 100 - 50) * 0.001,
            0
          );
          model->GetLink()->AddForce(force);
        }
      }
    }

    private: physics::WorldPtr world;
    private: double updateRate;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_WORLD_PLUGIN(DynamicWorldPlugin)
}
```

### Creating a Sensor Plugin

Here's an example of a custom sensor plugin that processes camera data:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/rendering/rendering.hh>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace gazebo
{
  class CustomCameraPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the camera sensor
      this->cameraSensor = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);
      if (!this->cameraSensor)
      {
        gzerr << "CustomCameraPlugin requires a camera sensor\n";
        return;
      }

      // Connect to camera update event
      this->newImageConnection = this->cameraSensor->Camera()->ConnectNewImageFrame(
          std::bind(&CustomCameraPlugin::OnNewFrame, this,
                   std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3, std::placeholders::_4,
                   std::placeholders::_5));

      gzmsg << "CustomCameraPlugin loaded\n";
    }

    private: void OnNewFrame(const unsigned char *_image,
                            unsigned int _width, unsigned int _height,
                            unsigned int _depth, const std::string &_format)
    {
      // Process the image data
      cv::Mat image = cv::Mat(_height, _width, CV_8UC3, (void*)_image);

      // Apply custom processing (e.g., edge detection)
      cv::Mat processed_image;
      cv::Canny(image, processed_image, 50, 150);

      // Convert back to ROS message format and publish
      // (Implementation details for ROS 2 publishing would go here)
    }

    private: sensors::CameraSensorPtr cameraSensor;
    private: rendering::CameraPtr camera;
    private: event::ConnectionPtr newImageConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomCameraPlugin)
}
```

### ROS 2 Integration with Plugins

For ROS 2 integration, plugins typically use the `gazebo_ros` package:

```cpp
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace gazebo
{
  class ROSLidarPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Initialize ROS node
      this->ros_node = gazebo_ros::Node::Get(_sdf);

      // Get sensor
      this->lidarSensor = std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);

      // Create publisher
      this->scan_pub = this->ros_node->create_publisher<sensor_msgs::msg::LaserScan>(
          "scan", rclcpp::QoS(10));

      // Connect to sensor update
      this->update_connection_ = this->lidarSensor->ConnectUpdated(
          std::bind(&ROSLidarPlugin::OnScan, this));
    }

    private: void OnScan()
    {
      // Get laser scan data from Gazebo
      auto ranges = this->lidarSensor->Ranges();

      // Create ROS message
      sensor_msgs::msg::LaserScan scan_msg;
      scan_msg.header.frame_id = "laser_frame";
      scan_msg.header.stamp = this->ros_node->now();

      // Fill in scan data
      scan_msg.angle_min = this->lidarSensor->AngleMin().Radian();
      scan_msg.angle_max = this->lidarSensor->AngleMax().Radian();
      scan_msg.angle_increment = this->lidarSensor->AngleResolution();
      scan_msg.time_increment = 0;
      scan_msg.scan_time = 0.1;
      scan_msg.range_min = this->lidarSensor->RangeMin();
      scan_msg.range_max = this->lidarSensor->RangeMax();

      scan_msg.ranges.resize(ranges.size());
      for (size_t i = 0; i < ranges.size(); ++i)
      {
        scan_msg.ranges[i] = ranges[i];
      }

      // Publish scan
      this->scan_pub->publish(scan_msg);
    }

    private: gazebo_ros::Node::SharedPtr ros_node;
    private: sensors::RaySensorPtr lidarSensor;
    private: rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub;
    private: event::ConnectionPtr update_connection_;
  };

  GZ_REGISTER_SENSOR_PLUGIN(ROSLidarPlugin)
}
```

## ROS 2 Control Integration

### Using ros2_control with Gazebo

The `gazebo_ros2_control` plugin enables integration with ROS 2 control:

```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="joint1">
    <command_interface name="position">
      <param name="min">-1.57</param>
      <param name="max">1.57</param>
    </command_interface>
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
    <parameters>$(find my_robot_control)/config/robot_control.yaml</parameters>
  </plugin>
</gazebo>
```

### Controller Configuration

The controller configuration file (YAML) for ros2_control:

```yaml
controller_manager:
  ros__parameters:
    use_sim_time: true
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    forward_position_controller:
      type: position_controllers/JointGroupPositionController

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

forward_position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
    interface_name: position

velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
    interface_name: velocity
```

## Creating Custom World Files

### World File Organization

Organize your world files in a package structure:

```
my_robot_gazebo/
├── worlds/
│   ├── empty_world.sdf
│   ├── maze_world.sdf
│   └── office_world.sdf
├── models/
│   ├── custom_obstacle/
│   └── custom_room/
├── launch/
│   └── launch_world.py
└── CMakeLists.txt
```

### Example: Maze World

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="maze_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 20</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 20</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Maze walls -->
    <model name="wall_1">
      <static>true</static>
      <pose>0 5 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <static>true</static>
      <pose>5 0 0.5 0 0 1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Goal marker -->
    <model name="goal">
      <static>true</static>
      <pose>8 8 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 0.5</ambient>
            <diffuse>0 1 0 0.5</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## World File Best Practices

### 1. Performance Optimization
- Use simple geometries for collision objects
- Limit the number of dynamic objects
- Configure appropriate physics parameters
- Use static models when possible

### 2. Organization
- Group related objects into logical sections
- Use descriptive names for models
- Comment complex world configurations
- Separate complex environments into multiple files

### 3. Reusability
- Create modular world components
- Use includes to reuse common elements
- Parameterize values that might change
- Document world file parameters

## Plugin Development Best Practices

### 1. Error Handling
- Validate SDF parameters during Load()
- Handle null pointers gracefully
- Provide informative error messages
- Implement proper cleanup in destructors

### 2. Performance
- Minimize work in update callbacks
- Cache frequently accessed values
- Use appropriate update rates
- Consider multithreading for complex operations

### 3. ROS Integration
- Use appropriate QoS settings
- Handle simulation time correctly
- Implement proper parameter handling
- Provide diagnostics and health monitoring

## Debugging World Files and Plugins

### World File Debugging
```bash
# Validate SDF file
ign sdf -k worlds/my_world.sdf

# View SDF file structure
ign sdf -p worlds/my_world.sdf

# Launch with verbose output
ign gazebo -v 4 worlds/my_world.sdf
```

### Plugin Debugging
- Check console output for error messages
- Verify plugin library paths
- Confirm SDF syntax and plugin names
- Use Gazebo's logging system for debugging

## Learning Objectives

By the end of this lesson, you should be able to:
- Create and configure custom Gazebo world files
- Develop plugins for sensors and controllers
- Integrate custom plugins with ROS 2
- Configure ros2_control for simulation
- Apply best practices for world file and plugin development
- Debug common issues with world files and plugins