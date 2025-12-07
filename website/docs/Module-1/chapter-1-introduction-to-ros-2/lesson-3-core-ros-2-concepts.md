---
title: 'Core ROS 2 Concepts'
description: 'Understanding fundamental concepts of ROS 2: nodes, topics, messages, services, and actions with examples'
chapter: 1
lesson: 3
module: 1
sidebar_label: 'Core ROS 2 Concepts'
sidebar_position: 3
tags: ['ROS 2', 'Core Concepts', 'Nodes', 'Topics', 'Services', 'Actions']
keywords: ['ROS 2', 'nodes', 'topics', 'messages', 'services', 'actions', 'publishers', 'subscribers']
---

# Core ROS 2 Concepts

## Overview

Understanding the core concepts of ROS 2 is fundamental to developing effective robotic applications. This lesson covers the essential building blocks of ROS 2 systems: nodes, topics, messages, services, and actions. These concepts form the foundation of communication and organization in ROS 2-based robotics applications.

## Nodes

### Definition
A node is a process that performs computation in ROS 2. Nodes are the fundamental building blocks of a ROS 2 system, and each node typically performs a specific task or set of related tasks. Nodes encapsulate functionality and communicate with other nodes through various communication mechanisms.

### Node Structure
In ROS 2, a node:
- Contains publishers, subscribers, services, and other communication interfaces
- Has its own namespace for organizing topics, services, and parameters
- Can be grouped with other nodes in the same process for efficiency
- Manages its own lifecycle and resources

### Creating Nodes
Nodes are typically created by inheriting from the Node class in your chosen client library (rclcpp for C++ or rclpy for Python). Each node must have a unique name within its namespace to avoid conflicts.

Example C++ node creation:
```cpp
#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node
{
public:
  MyNode() : Node("my_node")
  {
    // Node initialization code here
  }
};
```

Example Python node creation:
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        # Node initialization code here
```

## Topics and Messages

### Publish-Subscribe Pattern
Topics implement a publish-subscribe communication pattern where:
- Publishers send data to topics without knowledge of subscribers
- Subscribers receive data from topics without knowledge of publishers
- Multiple publishers and subscribers can exist for the same topic
- Communication is asynchronous and decoupled

### Messages
Messages are the data structures sent over topics. They are defined in `.msg` files and contain fields with specific data types. Messages enable structured data exchange between nodes and can include:
- Primitive types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, string, bool)
- Arrays of primitive types
- Other message types (nested messages)
- Constants

Example message definition (geometry_msgs/msg/Twist.msg):
```
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
```

### Creating Publishers and Subscribers

C++ Publisher Example:
```cpp
auto publisher = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
geometry_msgs::msg::Twist msg;
msg.linear.x = 1.0;
publisher->publish(msg);
```

Python Publisher Example:
```python
publisher = self.create_publisher(geometry_msgs.msg.Twist, 'cmd_vel', 10)
msg = geometry_msgs.msg.Twist()
msg.linear.x = 1.0
publisher.publish(msg)
```

C++ Subscriber Example:
```cpp
auto subscription = this->create_subscription<geometry_msgs::msg::Twist>(
  "cmd_vel", 10,
  [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received: %f", msg->linear.x);
  });
```

Python Subscriber Example:
```python
def callback(self, msg):
    self.get_logger().info('Received: %f' % msg.linear.x)

subscription = self.create_subscription(
    geometry_msgs.msg.Twist,
    'cmd_vel',
    callback,
    10)
```

### Quality of Service (QoS)
QoS settings allow fine-tuning of topic communication:
- **Reliability**: Reliable (guaranteed delivery) or Best Effort (no guarantee)
- **Durability**: Volatile (no storage for late joiners) or Transient Local (storage for late joiners)
- **History**: Keep All (store all messages) or Keep Last (store only recent messages)
- **Depth**: Number of messages to store when using Keep Last policy

## Services

### Request-Response Pattern
Services implement a request-response communication pattern where:
- A client sends a request to a service
- The service processes the request and sends back a response
- Communication is synchronous and blocking
- Only one service server can exist for each service name

### Service Definitions
Services are defined in `.srv` files containing two parts:
- **Request**: The data sent from the client to the service
- **Response**: The data sent from the service back to the client

Example service definition (example_interfaces/srv/AddTwoInts.srv):
```
int64 a
int64 b
---
int64 sum
```

### Creating Services and Clients

C++ Service Server Example:
```cpp
auto service = this->create_service<example_interfaces::srv::AddTwoInts>(
  "add_two_ints",
  [this](
    const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
    example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
  {
    response->sum = request->a + request->b;
    RCLCPP_INFO(this->get_logger(), "Incoming request\na: %ld b: %ld", request->a, request->b);
  });
```

Python Service Server Example:
```python
def add_two_ints_callback(self, request, response):
    response.sum = request.a + request.b
    self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
    return response

service = self.create_service(
    example_interfaces.srv.AddTwoInts,
    'add_two_ints',
    add_two_ints_callback)
```

C++ Service Client Example:
```cpp
auto client = this->create_client<example_interfaces::srv::AddTwoInts>("add_two_ints");
while (!client->wait_for_service(std::chrono::seconds(1))) {
  if (!rclcpp::ok()) {
    RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service.");
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
}
auto request = std::make_shared<example_interfaces::srv::AddTwoInts::Request>();
request->a = 2;
request->b = 3;
auto future = client->async_send_request(request);
```

## Actions

### Asynchronous Long-Running Tasks
Actions provide a communication pattern for long-running tasks that require:
- Goal requests from clients
- Continuous feedback during execution
- Result responses upon completion
- Ability to cancel or preempt ongoing tasks

### Action Structure
Actions are defined in `.action` files with three parts:
- **Goal**: Request sent by the client to start an action
- **Result**: Response sent by the server upon completion
- **Feedback**: Continuous updates sent during execution

Example action definition (turtlesim/action/RotateAbsolute.action):
```
float32 theta
---
float32 theta
---
float32 remaining
```

### Action States
Actions have a state machine that includes:
- **Pending**: Goal received but not yet started
- **Active**: Goal is being processed
- **Succeeded**: Goal completed successfully
- **Aborted**: Goal failed to complete
- **Canceled**: Goal was canceled by the client

## Parameters

### Dynamic Configuration
Parameters in ROS 2 allow for:
- Dynamic configuration of nodes during runtime
- Hierarchical organization of configuration data
- Type safety with validation
- Parameter callbacks for reactive configuration changes

### Parameter Features
- Declarative parameter definition with types and constraints
- Automatic parameter validation
- Parameter services for external tools
- Parameter events for monitoring changes

Example parameter usage in C++:
```cpp
this->declare_parameter("param_name", "default_value");
std::string param_value = this->get_parameter("param_name").as_string();
```

Example parameter usage in Python:
```python
self.declare_parameter('param_name', 'default_value')
param_value = self.get_parameter('param_name').value
```

## Command Line Tools

ROS 2 provides several command-line tools for inspecting and managing the system:

- `ros2 node list`: List all active nodes
- `ros2 topic list`: List all active topics
- `ros2 service list`: List all available services
- `ros2 action list`: List all available actions
- `ros2 node info <node_name>`: Get detailed information about a specific node
- `ros2 topic echo <topic_name>`: Print messages from a topic
- `ros2 service call <service_name> <service_type> <args>`: Call a service

Example of node inspection:
```
$ ros2 node info /my_node
/my_node
  Subscribers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
  Service Servers:
    /my_node/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /my_node/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
  Service Clients:
  Action Servers:
  Action Clients:
```

## Learning Objectives

By the end of this lesson, you should be able to:
- Define and create ROS 2 nodes with proper communication interfaces
- Implement topic-based communication with publishers and subscribers
- Design and use services for synchronous operations
- Create and manage actions for long-running tasks
- Configure parameters for dynamic node behavior
- Use command-line tools to inspect and manage ROS 2 systems
- Understand the relationships between nodes, topics, messages, services, and actions