---
title: 'ROS 2 Architecture'
description: "Understanding ROS 2's architecture, client libraries (rclcpp, rclpy), build system (colcon), and communication mechanisms (DDS)"
chapter: 1
lesson: 2
module: 1
sidebar_label: 'ROS 2 Architecture'
sidebar_position: 2
tags: ['ROS 2', 'Architecture', 'Client Libraries', 'DDS']
keywords: ['ROS 2', 'architecture', 'rclcpp', 'rclpy', 'DDS', 'colcon', 'middleware']
---

# ROS 2 Architecture

## Overview

The architecture of ROS 2 represents a fundamental shift from the centralized design of ROS 1 to a distributed, standards-based approach. This new architecture addresses critical limitations of ROS 1, particularly in areas of scalability, real-time performance, and security. Understanding the ROS 2 architecture is essential for developing robust and production-ready robotic systems.

## DDS-Based Communication Layer

ROS 2's communication layer is built on Data Distribution Service (DDS), an industry-standard middleware for real-time systems. DDS provides:

- **Quality of Service (QoS) policies** that allow fine-tuning of communication behavior
- **Language and platform independence** for multi-vendor interoperability
- **Built-in discovery mechanisms** for automatic node detection
- **Real-time capabilities** with deterministic timing characteristics

DDS implements the publish-subscribe pattern that underlies ROS 2's communication model, providing reliable message delivery with configurable policies for durability, reliability, and liveliness.

## Client Libraries (rclcpp and rclpy)

ROS 2 provides client libraries that wrap the underlying ROS concepts in different programming languages:

### rclcpp (C++)
The C++ client library provides:

- **Node creation**: `#include "rclcpp/rclcpp.hpp"` for creating ROS 2 nodes
- **Initialization**: `rclcpp::init(argc, argv);` to initialize the client library
- **Node interface**: Access to publishers, subscribers, services, and actions
- **Execution management**: Single-threaded and multi-threaded executors

Example C++ node initialization:
```cpp
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("my_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

### rclpy (Python)
The Python client library provides:

- **Node creation**: Import `rclpy` and use the Node class
- **Initialization**: `rclpy.init()` to initialize the client library
- **Node interface**: Access to publishers, subscribers, services, and actions
- **Dependency management**: Requires `rclpy` as a dependency in package.xml

Example Python node initialization:
```python
import rclpy
from rclpy.node import Node

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
```

## Build System (Colcon)

Colcon is the build tool for ROS 2 workspaces, replacing the previous catkin tools:

### Key Features
- **Multi-package builds**: Builds multiple packages in dependency order
- **Parallel execution**: Builds packages in parallel to reduce build time
- **Symlink installation**: Creates symlinks to avoid copying files
- **Flexible configuration**: Supports various build systems (CMake, ament_cmake, etc.)

### Common Colcon Commands
- `colcon build`: Build all packages in the workspace
- `colcon build --packages-select <package_name>`: Build specific packages
- `colcon build --symlink-install`: Build with symlinks
- `colcon test`: Run tests for packages

## Communication Mechanisms

### Topics (Publish-Subscribe)
Topics enable asynchronous, decoupled communication:
- Publishers send messages to topics without knowledge of subscribers
- Subscribers receive messages from topics without knowledge of publishers
- Multiple publishers and subscribers can exist for the same topic
- Provides loose coupling between components

### Services (Request-Response)
Services enable synchronous, request-response communication:
- Clients send requests to services and wait for responses
- Services process requests and return responses
- Useful for operations that require immediate results
- Blocking communication model

Example Python service implementation:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\\na: %d b: %d' % (request.a, request.b))
        return response
```

### Actions (Long-Running Tasks)
Actions extend services for long-running operations:
- Support goal, feedback, and result messages
- Allow for preemption and cancellation
- Suitable for tasks like navigation and manipulation
- Non-blocking with progress monitoring

## Quality of Service (QoS) Profiles

QoS profiles allow fine-tuning of communication behavior:

### Reliability Policy
- **Reliable**: Guarantees message delivery (default for most cases)
- **Best Effort**: Does not guarantee message delivery (for real-time systems)

### Durability Policy
- **Transient Local**: Provides late-joining subscribers access to recent messages
- **Volatile**: Does not store messages for late joiners (default)

### History Policy
- **Keep Last**: Maintains a specified number of most recent messages
- **Keep All**: Maintains all messages (use with caution for memory usage)

## Parameter System

ROS 2 provides a hierarchical parameter system:
- Parameters can be declared with types, descriptions, and constraints
- Support for parameter validation and callbacks
- Dynamic parameter updates during runtime
- Parameter services for external tools

## Security Architecture

ROS 2 includes comprehensive security features:
- **Authentication**: Verifies identity of nodes and users
- **Access Control**: Controls what resources nodes can access
- **Encryption**: Protects data in transit and at rest
- **Secure Discovery**: Prevents unauthorized nodes from joining the system

## Lifecycle Management

ROS 2 provides a state machine for managing node lifecycles:
- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Running and operational
- **Finalized**: Node is shutting down

This allows for coordinated startup, shutdown, and reconfiguration of complex systems.

## Learning Objectives

By the end of this lesson, you should be able to:
- Explain the DDS-based architecture of ROS 2
- Understand the role of client libraries (rclcpp and rclpy)
- Describe the colcon build system and its features
- Identify the components of the ROS 2 communication mechanisms (DDS)
- Recognize the benefits of Quality of Service (QoS) policies