---
title: 'ROS 2 Development Environment Setup'
description: 'Guide to setting up the ROS 2 development environment, including installation, workspace creation, and configuration.'
chapter: 2
lesson: 1
module: 1
sidebar_label: 'ROS 2 Development Environment Setup'
sidebar_position: 1
tags: ['ROS 2', 'Environment Setup', 'Installation', 'Workspace']
keywords: ['ROS 2', 'installation', 'development environment', 'workspace', 'colcon', 'setup']
---

# ROS 2 Development Environment Setup

## Overview

Setting up a proper development environment is crucial for working with ROS 2 effectively. This lesson will guide you through the process of installing ROS 2, creating a workspace, and configuring your environment to start developing robotic applications. A well-configured environment will help you avoid common issues and ensure smooth development workflows.

## Prerequisites

Before installing ROS 2, ensure your system meets the following requirements:

### Supported Operating Systems
- Ubuntu 22.04 (Jammy Jellyfish) - Recommended for development
- Ubuntu 20.04 (Focal Fossa)
- Windows 10/11 (with WSL2 recommended)
- macOS (with Docker or virtual machines)

### System Requirements
- At least 4GB RAM (8GB recommended)
- At least 10GB free disk space
- Internet connection for package installation
- Administrative privileges for installation

### Recommended Development Setup
- A modern computer with multiple cores for efficient builds
- Reliable internet connection for package updates
- Terminal/shell proficiency (bash, zsh, or PowerShell)

## Installing ROS 2

### Ubuntu Installation

For Ubuntu users, the recommended approach is to install ROS 2 via Debian packages:

1. **Set up your sources.list**
   ```bash
   sudo apt update && sudo apt install -y curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

2. **Update apt and install ROS 2 packages**
   ```bash
   sudo apt update
   sudo apt install -y ros-humble-desktop
   ```

3. **Install additional development tools**
   ```bash
   sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
   ```

4. **Initialize rosdep**
   ```bash
   sudo rosdep init
   rosdep update
   ```

### Windows Installation (with WSL2)

For Windows users, using Windows Subsystem for Linux (WSL2) is recommended:

1. **Install WSL2 with Ubuntu 22.04**
   ```cmd
   wsl --install -d Ubuntu-22.04
   ```

2. **Follow the Ubuntu installation steps above**

### macOS Installation

For macOS users, Docker or virtual machines are recommended:

1. **Install Docker Desktop for Mac**
   - Download from docker.com
   - Install and start Docker Desktop

2. **Use a ROS 2 Docker container for development**
   ```bash
   docker run -it --rm ros:humble
   ```

## Sourcing ROS 2 Environment

After installation, you need to source the ROS 2 environment to use ROS 2 commands:

```bash
source /opt/ros/humble/setup.bash
```

To make this permanent, add the following line to your shell's startup script:

For bash:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

For zsh:
```bash
echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc
source ~/.zshrc
```

## Creating a ROS 2 Workspace

A ROS 2 workspace is a directory where you'll develop and build your ROS 2 packages. Here's how to create one:

1. **Create the workspace directory structure**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

2. **Source your main ROS 2 installation** (the underlay for your development workspace)
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. **Build the workspace for the first time**
   ```bash
   colcon build
   ```

4. **Source the workspace**
   ```bash
   source install/setup.bash
   ```

## Using Colcon for Building

Colcon is the build tool for ROS 2 workspaces. It's an improvement over the previous catkin tools:

### Basic Colcon Commands

- **Build the entire workspace:**
  ```bash
  colcon build
  ```

- **Build only specific packages:**
  ```bash
  colcon build --packages-select <package_name>
  ```

- **Build with parallel jobs (faster):**
  ```bash
  colcon build --parallel-workers 4
  ```

- **Build with symlinks (faster rebuilds):**
  ```bash
  colcon build --symlink-install
  ```

- **Clean build artifacts:**
  ```bash
  rm -rf build/ install/ log/
  ```

## Environment Configuration

### Setting Up Your Development Environment

1. **Create a bash alias for sourcing your workspace**
   ```bash
   echo "alias cw='cd ~/ros2_ws && source install/setup.bash'" >> ~/.bashrc
   echo "alias cb='cd ~/ros2_ws && colcon build --symlink-install'" >> ~/.bashrc
   ```

2. **Set up ROS 2 environment variables**
   ```bash
   echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc  # Default domain
   echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc  # Use CycloneDDS
   ```

### ROS 2 Environment Variables

- **ROS_DOMAIN_ID**: Sets the ROS domain (0-232), allowing multiple ROS graphs on the same network
- **RMW_IMPLEMENTATION**: Sets the middleware implementation (e.g., rmw_fastrtps_cpp, rmw_cyclonedx_cpp)
- **ROS_LOG_DIR**: Sets the directory for log files
- **RCUTILS_LOGGING_SEVERITY_THRESHOLD**: Sets the minimum severity for logging (DEBUG, INFO, WARN, ERROR)

## Verification and Testing

After setting up your environment, verify that everything works correctly:

1. **Check ROS 2 installation:**
   ```bash
   ros2 --version
   ```

2. **Run a simple example:**
   ```bash
   ros2 run demo_nodes_cpp talker
   ```

3. **In a new terminal, run the listener:**
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/ros2_ws/install/setup.bash
   ros2 run demo_nodes_py listener
   ```

4. **Use the ROS 2 CLI tools:**
   ```bash
   ros2 topic list
   ros2 node list
   ros2 service list
   ```

## Troubleshooting Common Issues

### Package Installation Issues
- **Problem**: Unable to locate package ros-humble-desktop
- **Solution**: Ensure your sources.list is correctly configured and update apt:
  ```bash
  sudo apt update
  ```

### Permission Issues
- **Problem**: Permission denied when running ROS 2 commands
- **Solution**: Ensure you're running in the correct environment and have proper permissions

### Network Configuration
- **Problem**: Nodes can't communicate across machines
- **Solution**: Check firewall settings and ensure ROS_DOMAIN_ID matches across machines

## Best Practices

1. **Use Virtual Environments**: Consider using virtual environments for Python-based ROS packages
2. **Version Control**: Always use git for your workspace and packages
3. **Documentation**: Document your workspace setup for reproducibility
4. **Regular Updates**: Keep your ROS 2 installation updated with security patches
5. **Clean Builds**: Regularly clean build artifacts to avoid build issues

## Learning Objectives

By the end of this lesson, you should be able to:
- Install ROS 2 on your preferred operating system
- Create and configure a ROS 2 development workspace
- Understand and use colcon for building packages
- Configure environment variables for ROS 2 development
- Verify your setup with basic ROS 2 examples
- Troubleshoot common environment setup issues