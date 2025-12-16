---
title: 'Creating a ROS 2 Package'
description: 'Learn how to create and manage ROS 2 packages, including package structure, ros2 pkg create, package.xml, and CMakeLists.txt/setup.py'
chapter: 2
lesson: 2
module: 1
sidebar_label: 'Creating a ROS 2 Package'
sidebar_position: 2
tags: ['ROS 2', 'Packages', 'Development', 'Build System']
keywords: ['ROS 2', 'packages', 'package.xml', 'CMakeLists.txt', 'setup.py', 'ros2 pkg create']
---

# Creating a ROS 2 Package

## Overview

A ROS 2 package is the fundamental unit of organization in ROS 2. It contains source code, launch files, configuration files, and documentation. This lesson will guide you through creating ROS 2 packages, understanding their structure, and configuring them properly for development.

## Package Structure

A typical ROS 2 package follows a standardized structure:

```
my_package/
├── CMakeLists.txt          # Build configuration for C++ packages
├── package.xml             # Package metadata and dependencies
├── src/                    # Source code files (for C++ packages)
├── include/my_package/     # Header files (for C++ packages)
├── scripts/                # Standalone executable scripts
├── launch/                 # Launch files
├── config/                 # Configuration files
├── test/                   # Test files
└── setup.py                # Build configuration for Python packages (if applicable)
```

## Creating a Package with ros2 pkg create

The easiest way to create a new ROS 2 package is using the `ros2 pkg create` command:

```bash
cd ~/ros2_ws/src
ros2 pkg create --help
```

### Basic Package Creation

To create a simple package:
```bash
ros2 pkg create my_robot_package
```

### Creating a Package with Build Type

For C++ packages:
```bash
ros2 pkg create --build-type ament_cmake my_cpp_package
```

For Python packages:
```bash
ros2 pkg create --build-type ament_python my_py_package
```

### Creating a Package with Dependencies

To create a package with dependencies:
```bash
ros2 pkg create --build-type ament_cmake --dependencies rclcpp std_msgs geometry_msgs my_cpp_package
```

### Creating a Package with Node Templates

To create a package with template files:
```bash
ros2 pkg create --build-type ament_cmake --node-name my_node my_cpp_package
```

## Package.xml Configuration

The `package.xml` file contains metadata about the package and its dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Key Elements in package.xml

- **name**: The package name (must be unique within the workspace)
- **version**: Package version in semantic versioning format (major.minor.patch)
- **description**: Brief description of the package
- **maintainer**: Contact information for the package maintainer
- **license**: Software license for the package
- **buildtool_depend**: Build system dependencies (e.g., ament_cmake)
- **depend**: Runtime dependencies for the package
- **test_depend**: Dependencies required for testing
- **export/build_type**: Specifies the build system type

## CMakeLists.txt for C++ Packages

For C++ packages, the `CMakeLists.txt` file configures the build process:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_cpp_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Create executable
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node
  rclcpp
  std_msgs
  geometry_msgs
)

# Install targets
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install other files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

### Key CMake Configuration Elements

- **cmake_minimum_required**: Specifies minimum CMake version
- **project**: Defines the project name
- **find_package**: Locates required packages and their dependencies
- **add_executable**: Creates an executable target from source files
- **ament_target_dependencies**: Links dependencies to the target
- **install**: Specifies files to install when the package is built
- **ament_package**: Finalizes the package configuration

## Setup.py for Python Packages

For Python packages, the `setup.py` file handles the build configuration:

```python
from setuptools import setup

package_name = 'my_py_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_py_package.my_node:main',
        ],
    },
)
```

### Key Setup.py Elements

- **name**: The package name
- **version**: Version of the package
- **packages**: Python packages to include
- **data_files**: Additional files to install
- **install_requires**: Python dependencies
- **entry_points**: Console scripts that can be executed
- **tests_require**: Dependencies needed for testing

## Package Development Best Practices

### Naming Conventions

- Use lowercase letters with underscores separating words (snake_case)
- Avoid spaces and special characters
- Choose descriptive names that indicate the package's purpose
- Keep names relatively short but meaningful

### Directory Structure

- Organize source files in appropriate subdirectories
- Keep launch files in a `launch/` directory
- Store configuration files in a `config/` directory
- Place test files in a `test/` directory
- Document the package structure in README files

### Dependency Management

- Only list actual dependencies in package.xml
- Use specific versions when necessary to ensure compatibility
- Regularly review and update dependencies
- Document any special installation requirements

## Creating Nodes within Packages

### C++ Node Example

Create a simple C++ node in `src/my_node.cpp`:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MyNode : public rclcpp::Node
{
public:
  MyNode() : Node("my_node")
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&MyNode::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world!";
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyNode>());
  rclcpp::shutdown();
  return 0;
}
```

### Python Node Example

Create a simple Python node in `my_py_package/my_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, world!'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Package Verification

After creating your package, verify it's properly set up:

1. **Check package structure**:
   ```bash
   ros2 pkg executables my_robot_package
   ```

2. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_package
   ```

3. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

4. **List available nodes**:
   ```bash
   ros2 run my_robot_package my_node
   ```

## Common Issues and Troubleshooting

### Missing Dependencies
- **Problem**: Build fails due to missing dependencies
- **Solution**: Add missing dependencies to package.xml and run `rosdep install`

### Build Errors
- **Problem**: Compilation errors during build
- **Solution**: Check CMakeLists.txt configuration and source file locations

### Package Not Found
- **Problem**: ROS 2 cannot find the package after building
- **Solution**: Ensure you've sourced the workspace setup file

## Learning Objectives

By the end of this lesson, you should be able to:
- Create ROS 2 packages using the ros2 pkg create command
- Understand and configure the package.xml file properly
- Set up CMakeLists.txt for C++ packages or setup.py for Python packages
- Organize package structure according to ROS 2 conventions
- Create and configure nodes within packages
- Verify package functionality and troubleshoot common issues