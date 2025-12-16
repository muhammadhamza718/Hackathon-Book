---
title: 'Building and Running ROS 2 Code'
description: 'Understanding how to build and run ROS 2 packages using colcon build, sourcing setup files, and running nodes with ros2 run. Includes publisher/subscriber examples.'
chapter: 2
lesson: 3
module: 1
sidebar_label: 'Building and Running ROS 2 Code'
sidebar_position: 3
tags: ['ROS 2', 'Building', 'Running', 'colcon', 'Publisher', 'Subscriber']
keywords: ['ROS 2', 'colcon build', 'ros2 run', 'publisher', 'subscriber', 'launch']
---

# Building and Running ROS 2 Code

## Overview

Once you have created your ROS 2 packages, the next step is to build and run them. This lesson covers the essential tools and techniques for building ROS 2 packages using colcon, sourcing setup files, and running nodes with the ros2 run command. We'll also explore publisher and subscriber examples to demonstrate the complete development cycle.

## Building ROS 2 Packages with Colcon

### Basic Build Process

After creating your ROS 2 packages, you need to build them before they can be executed. The build process compiles your source code and creates executables:

```bash
cd ~/ros2_ws
colcon build
```

This command will build all packages in your workspace. The build process includes:
- Compiling source code
- Linking libraries
- Generating message/service/action interfaces
- Installing executables and libraries to the install directory

### Selective Building

For faster development cycles, you can build specific packages:

```bash
colcon build --packages-select my_package_name
```

You can also build multiple specific packages:

```bash
colcon build --packages-select package1 package2 package3
```

### Build with Symlinks

For faster rebuilds during development, use the symlink option:

```bash
colcon build --symlink-install
```

This creates symbolic links instead of copying files, which speeds up rebuilds when you make changes to your source code.

### Parallel Building

To speed up the build process, you can use multiple parallel jobs:

```bash
colcon build --parallel-workers 4
```

## Sourcing Setup Files

After building your packages, you need to source the setup files to make the executables available in your environment:

```bash
source install/setup.bash
```

This command adds the built executables and libraries to your PATH and LD_LIBRARY_PATH, allowing you to run your ROS 2 nodes.

For convenience, you can add this to your shell's startup script:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Running ROS 2 Nodes

### Using ros2 run

To run a specific node from a package, use the `ros2 run` command:

```bash
ros2 run <package_name> <executable_name>
```

For example:
```bash
ros2 run demo_nodes_cpp talker
```

### Publisher Example

Let's create and run a simple publisher node:

1. **Create a publisher node** in `src/my_publisher.cpp`:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

2. **Build the package**:
```bash
colcon build --packages-select my_robot_package
```

3. **Source the workspace**:
```bash
source install/setup.bash
```

4. **Run the publisher**:
```bash
ros2 run my_robot_package my_publisher
```

### Subscriber Example

Now let's create a subscriber to receive messages from the publisher:

1. **Create a subscriber node** in `src/my_subscriber.cpp`:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
      });
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
```

2. **Build the package** (if not already built):
```bash
colcon build --packages-select my_robot_package
```

3. **Source the workspace**:
```bash
source install/setup.bash
```

4. **Run the subscriber** in a new terminal:
```bash
ros2 run my_robot_package my_subscriber
```

5. **Run the publisher** in another terminal:
```bash
ros2 run my_robot_package my_publisher
```

You should now see the publisher sending messages and the subscriber receiving them.

### Python Publisher/Subscriber Example

For Python nodes, the process is similar:

**Python Publisher** (`my_robot_package/my_robot_package/my_publisher.py`):
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Python Subscriber** (`my_robot_package/my_robot_package/my_subscriber.py`):
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
}
```

## Advanced Build Options

### Build with Specific Build Type

For packages that use different build systems:
```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Build and Test

To run tests after building:
```bash
colcon build
colcon test
```

To run tests for specific packages:
```bash
colcon test --packages-select my_package_name
```

### Build with Verbose Output

For debugging build issues:
```bash
colcon build --event-handlers console_direct+
```

## Launch Files

For running multiple nodes together, you can use launch files:

**Python Launch File** (`my_robot_package/launch/my_launch.py`):
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_publisher',
            name='publisher_node'
        ),
        Node(
            package='my_robot_package',
            executable='my_subscriber',
            name='subscriber_node'
        )
    ])
```

To run the launch file:
```bash
ros2 launch my_robot_package my_launch.py
```

## Troubleshooting Common Issues

### Build Errors
- **Problem**: Build fails with compilation errors
- **Solution**: Check error messages for missing dependencies or syntax errors
- **Check**: Ensure all dependencies are listed in package.xml

### Node Not Found
- **Problem**: `ros2 run` cannot find the node
- **Solution**: Ensure you've built the package and sourced the setup file
- **Check**: Verify the executable name matches what's defined in CMakeLists.txt

### Nodes Not Communicating
- **Problem**: Publisher and subscriber don't communicate
- **Solution**: Verify they're on the same topic and have matching message types
- **Check**: Use `ros2 topic list` and `ros2 topic echo` to verify topics

### Environment Issues
- **Problem**: Commands fail due to environment not being set up
- **Solution**: Source the ROS 2 installation and workspace setup files
- **Check**: Verify the environment with `printenv | grep ROS`

## Best Practices

1. **Incremental Building**: Use `--packages-select` to build only changed packages
2. **Regular Cleaning**: Clean build artifacts periodically with `rm -rf build/ install/ log/`
3. **Testing**: Run tests regularly with `colcon test`
4. **Documentation**: Document your build and run procedures
5. **Version Control**: Keep only source files in version control, not build artifacts

## Learning Objectives

By the end of this lesson, you should be able to:
- Build ROS 2 packages using colcon with various options
- Source workspace setup files to make executables available
- Run ROS 2 nodes using the ros2 run command
- Create and run publisher/subscriber examples
- Use launch files to run multiple nodes together
- Troubleshoot common building and running issues