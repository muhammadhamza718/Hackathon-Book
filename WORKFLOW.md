# Master Workflow: Physical AI & Humanoid Robotics Book

**Structure:** 4 Modules → 13 Chapters → 39 Lessons  
**Quota Strategy:** Planning (gemini-2.5-flash-lite) | Implementation (gemini-2.0-flash-lite)

---

## ✅ PHASE 1: PLANNING (COMPLETE)

### Task 1.1: Create Specification ✅

**Model:** gemini-2.0-flash  
**Command:** `/sp.specify`

**Prompt:**

```
Based on the constitution and the Physical AI & Humanoid Robotics Textbook document, create a detailed Specification for the Physical AI & Humanoid Robotics book. The specification must include:

1. **Book Structure**
   - Define the chapter structure (specify number of chapters, each with 3 lessons)
   - For each chapter: provide title and brief description
   - For each lesson within chapters: provide title and learning objectives
   - Ensure progression aligns with the hands-on learning focus from the constitution

2. **Content Guidelines**
   - Define the lesson format template (structure, sections, components)
   - Specify code example standards and formatting requirements
   - Outline hands-on project requirements per lesson/chapter

3. **Docusaurus-Specific Requirements**
   - Specify file organization structure (directory layout, naming conventions)
   - Define markdown formatting standards and frontmatter requirements
   - Specify how to organize code examples and assets
   - Outline navigation structure and sidebar configuration needs

Ensure the specification is actionable and directly implementable for building the book in Docusaurus.
```

**Output:** `specs/physical-ai-humanoid-robotics/spec.md`

---

### Task 1.2: Create Development Plan ✅

**Model:** gemini-2.5-flash  
**Command:** `/sp.plan`

**Prompt:**

```
Based on the Physical AI & Humanoid Robotics Textbook document, the Constitution, and the Specification, create a comprehensive Development Plan for building the Physical AI & Humanoid Robotics book in Docusaurus. The plan must include:

1. **Docusaurus Setup and Configuration**
   - Step-by-step Docusaurus installation and initialization
   - Use the typescript ( --typescript )
   - Required configuration files (docusaurus.config.ts, sidebars.ts)
   - Theme customization aligned with the book's requirements
   - Plugin setup and dependencies

2. **Content Development Phases**
   - Phase breakdown aligned with the 13 chapters (each with 3 lessons) from the specification
   - Prioritization and sequencing of chapters based on dependencies
   - Milestones and deliverables for each phase
   - Timeline estimates for content creation

3. **File Structure for Chapters and Lessons**
   - Complete directory structure matching the specification's file organization
   - Naming conventions for chapters and lessons
   - Frontmatter template for lesson files
   - Code example and asset organization structure

Ensure the development plan is actionable, follows the specification's requirements, and provides clear implementation guidance for building the book in Docusaurus.
Use the mcp tool for Context7, Read The all Step and Configuration content in Context7 documentation about Docusaurus.
```

**Output:** `specs/001-docusaurus-plan/plan.md`, `research.md`, `data-model.md`, `quickstart.md`, `contracts/`

---

## 🔄 PHASE 2: TASK BREAKDOWN (NEXT) ✅

### Task 1.3: Create Detailed Task List

**Model:** gemini-2.5-flash-lite  
**Command:** `/sp.tasks`

**Prompt:**

```
/sp.tasks

Create detailed tasks for filling all 13 chapters (39 lessons) with content based on the Physical AI & Humanoid Robotics Textbook and specification.

**CRITICAL REQUIREMENTS:**

1. **Research Phase (MANDATORY):**
   - Use MCP tool with context7 to research Physical AI & Humanoid Robotics topics
   - Gather comprehensive context for all 13 chapters
   - Research each module's focus areas:
     * Module 1 (Chapters 1-4): ROS 2, robotic middleware, URDF, sensor systems, launch files
     * Module 2 (Chapters 5-7): Gazebo, Unity, physics simulation, sensors, Isaac Sim
     * Module 3 (Chapters 8-10): NVIDIA Isaac, AI perception, reinforcement learning, sim-to-real transfer
     * Module 4 (Chapters 11-13): Humanoid development, kinematics, dynamics, locomotion, VLA, conversational robotics

2. **Structure Understanding:**
   - 4 MODULES (high-level from textbook) → 13 CHAPTERS (from spec) → 39 LESSONS (3 per chapter)
   - Module 1 (ROS 2) → Chapters 1-4
   - Module 2 (Digital Twin) → Chapters 5-7
   - Module 3 (AI-Robot Brain) → Chapters 8-10
   - Module 4 (VLA) → Chapters 11-13

3. **Task Generation:**
   - Create one task per lesson (39 tasks total)
   - Each task must specify:
     * Exact file path: `website/docs/Module-N/chapter-N-title/lesson-N-title.md`
     * Content requirements from Physical AI & Humanoid Robotics Textbook + spec.md
     * Learning objectives from spec.md section 1
     * Frontmatter format from spec.md section 3.3 (must include module number)
   - Organize by module → chapter → lesson
   - Include dependencies and parallel markers [P]

4. **Content Requirements:**
   - Align with Physical AI & Humanoid Robotics Textbook content
   - Follow lesson format from specs/physical-ai-humanoid-robotics/spec.md section 2
   - Include frontmatter from spec.md section 3.3
   - Reference code examples from textbook where applicable
```

**Output:** `specs/001-docusaurus-plan/tasks.md` with 39 detailed tasks

---

## 📝 PHASE 3: CONTENT IMPLEMENTATION

### Task 2.1: Fill Chapter 1 Content ✅

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapter 1 (Introduction to ROS 2):

1. **Lesson 1.1: Introduction to ROS 2**
   - File: `website/docs/Module-1/chapter-1-introduction-to-ros-2/lesson-1-introduction-to-ros-2.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives from spec.md: Understand what ROS 2 is, its role in robotics, and key components (nodes, topics, services, actions)
   - Follow lesson format template from spec.md section 2
   - Add frontmatter from spec.md section 3.3 (include module: 1, chapter: 1, lesson: 1)
   - Content: Explain ROS 2's role, key components (nodes, topics, services, actions), and architecture
   - Reference Physical AI & Humanoid Robotics Textbook.md and tasks.md T006

2. **Lesson 1.2: ROS 2 Architecture**
   - File: `website/docs/Module-1/chapter-1-introduction-to-ros-2/lesson-2-ros-2-architecture.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives: Understand ROS 2's architecture, client libraries (rclcpp, rclpy), build system (colcon), and communication mechanisms (DDS)
   - Content: Detail client libraries (rclcpp, rclpy), build system (colcon), communication mechanisms (DDS), and key concepts
   - Reference tasks.md T007

3. **Lesson 1.3: Core ROS 2 Concepts**
   - File: `website/docs/Module-1/chapter-1-introduction-to-ros-2/lesson-3-core-ros-2-concepts.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives: Grasp fundamental concepts of ROS 2: nodes, topics, messages, services, and actions with examples
   - Content: Explain nodes, topics, messages, services, and actions with examples, emphasizing publish-subscribe and client-server patterns
   - Reference tasks.md T008

4. **Create _category_.json** for chapter-1-introduction-to-ros-2 folder

Use the write tool (NOT Update) for all file operations. Reference tasks.md T006-T008 for detailed requirements.
```

---

### Task 2.2: Fill Chapter 2 Content ✅

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapter 2 (ROS 2 Development Environment Setup):

1. **Lesson 2.1: ROS 2 Development Environment Setup**
   - File: `website/docs/Module-1/chapter-2-ros-2-development-environment-setup/lesson-1-ros-2-development-environment-setup.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives from spec.md (Setting up the ROS 2 development environment and workspace)
   - Follow lesson format template from spec.md section 2
   - Add frontmatter from spec.md section 3.3 (include module: 1)
   - Content: Guide on installing ROS 2, setting up workspace with `colcon`, configuring environment
   - Reference Physical AI & Humanoid Robotics Textbook.md weeks 3-5 content

2. **Lesson 2.2: Creating a ROS 2 Package**
   - File: `website/docs/Module-1/chapter-2-ros-2-development-environment-setup/lesson-2-creating-a-ros-2-package.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives: Learn to create and manage ROS 2 packages
   - Content: Explain package structure, `ros2 pkg create`, `package.xml`, `CMakeLists.txt`/`setup.py`

3. **Lesson 2.3: Building and Running ROS 2 Code**
   - File: `website/docs/Module-1/chapter-2-ros-2-development-environment-setup/lesson-3-building-and-running-ros-2-code.md`
   - Use context7 MCP tool to gather information
   - Include learning objectives: Understand how to build and run ROS 2 packages
   - Content: Detail `colcon build`, sourcing setup files, running nodes with `ros2 run`. Include publisher/subscriber examples

4. **Create _category_.json** for chapter-2-ros-2-development-environment-setup folder

Use the write tool (NOT Update) for all file operations. Reference tasks.md T009-T011 for detailed requirements.
```

---

### Task 2.3: Fill Chapters 3-4 Content ✅

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapters 3-4 (6 lessons total):

**Chapter 3: Understanding and Creating URDF Models**
1. **Lesson 3.1: Understanding and Creating URDF Models**
   - File: `website/docs/Module-1/chapter-3-understanding-and-creating-urdf-models/lesson-1-understanding-and-creating-urdf-models.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Learn to create URDF models for robots
   - Content: Explain URDF structure (links, joints, properties), provide examples, visualization with RViz
   - Reference tasks.md T012

2. **Lesson 3.2: URDF for Robot Kinematics**
   - File: `website/docs/Module-1/chapter-3-understanding-and-creating-urdf-models/lesson-2-urdf-for-robot-kinematics.md`
   - Learning objectives: Understand how URDF defines robot kinematics
   - Content: Explain URDF joint types, degrees of freedom, forward/inverse kinematics
   - Reference tasks.md T013

3. **Lesson 3.3: Advanced URDF Features**
   - File: `website/docs/Module-1/chapter-3-understanding-and-creating-urdf-models/lesson-3-advanced-urdf-features.md`
   - Learning objectives: Explore advanced URDF features for robot modeling
   - Content: Cover transmissions, visual/collision elements, materials, and xacro
   - Reference tasks.md T014

**Chapter 4: Working with ROS 2 Sensor Interfaces**
1. **Lesson 4.1: Working with ROS 2 Sensor Interfaces**
   - File: `website/docs/Module-1/chapter-4-working-with-ros-2-sensor-interfaces/lesson-1-working-with-ros-2-sensor-interfaces.md`
   - Learning objectives: Integrate sensor data into ROS 2
   - Content: Explain interfacing with cameras, LiDAR, IMUs; message types; publishing sensor data
   - Reference tasks.md T015

2. **Lesson 4.2: ROS 2 Launch Files**
   - File: `website/docs/Module-1/chapter-4-working-with-ros-2-sensor-interfaces/lesson-2-ros-2-launch-files.md`
   - Learning objectives: Understand and create ROS 2 launch files
   - Content: Explain launch files (launch.py) for starting nodes, parameters, graph configuration
   - Reference tasks.md T016

3. **Lesson 4.3: Sensor Data Processing in ROS 2**
   - File: `website/docs/Module-1/chapter-4-working-with-ros-2-sensor-interfaces/lesson-3-sensor-data-processing-in-ros-2.md`
   - Learning objectives: Process sensor data within ROS 2 nodes
   - Content: Demonstrate subscribing to sensor topics, processing data, publishing results
   - Reference tasks.md T017

Create _category_.json files for both chapters. Use write tool (NOT Update) for all operations. Reference tasks.md T012-T017 for detailed requirements.
```

---

### Task 2.4: Fill Chapters 5-7 Content ✅

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapters 5-7 (9 lessons total):

**Chapter 5: Introduction to Gazebo**
1. **Lesson 5.1: Introduction to Gazebo**
   - File: `website/docs/Module-2/chapter-5-introduction-to-gazebo/lesson-1-introduction-to-gazebo.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Introduction to Gazebo for robot simulation
   - Content: Overview of Gazebo, its features (physics, sensors, rendering), and ROS 2 integration
   - Reference tasks.md T018

2. **Lesson 5.2: Using Gazebo with ROS 2**
   - File: `website/docs/Module-2/chapter-5-introduction-to-gazebo/lesson-2-using-gazebo-with-ros-2.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Learn to use Gazebo with ROS 2
   - Content: Detail launching Gazebo with ROS 2, spawning robots (URDF), interfacing with simulated sensors/controllers
   - Reference tasks.md T019

3. **Lesson 5.3: Gazebo Plugins and World Files**
   - File: `website/docs/Module-2/chapter-5-introduction-to-gazebo/lesson-3-gazebo-plugins-and-world-files.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand and create Gazebo world files and plugins
   - Content: Explain creating custom Gazebo world files and plugins for sensors/controllers
   - Reference tasks.md T020

**Chapter 6: Advanced Simulation with Unity**
1. **Lesson 6.1: Advanced Simulation with Unity**
   - File: `website/docs/Module-2/chapter-6-advanced-simulation-with-unity/lesson-1-advanced-simulation-with-unity.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Explore advanced simulation techniques using Unity
   - Content: Discuss Unity for simulation, graphical capabilities, and ROS 2 integration methods
   - Reference tasks.md T021

2. **Lesson 6.2: Unity for Realistic Physics Simulation**
   - File: `website/docs/Module-2/chapter-6-advanced-simulation-with-unity/lesson-2-unity-for-realistic-physics-simulation.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Utilize Unity's physics engine for realistic robot simulation
   - Content: Explain Unity's physics engine for simulating interactions, forces, materials, rigid bodies, colliders, and joints
   - Reference tasks.md T022

3. **Lesson 6.3: Integrating ROS 2 with Unity**
   - File: `website/docs/Module-2/chapter-6-advanced-simulation-with-unity/lesson-3-integrating-ros-2-with-unity.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Integrate ROS 2 with Unity simulations
   - Content: Describe methods for connecting ROS 2 with Unity, data exchange, controlling simulated robots, receiving sensor data
   - Reference tasks.md T023

**Chapter 7: Integrating Real-World Sensors into Simulation**
1. **Lesson 7.1: Integrating Real-World Sensors into Simulation**
   - File: `website/docs/Module-2/chapter-7-integrating-real-world-sensors-into-simulation/lesson-1-integrating-real-world-sensors-into-simulation.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Integrate real-world sensor data into simulation environments
   - Content: Explain bridging real-world sensor data and simulation, using recorded data or live feeds
   - Reference tasks.md T024

2. **Lesson 7.2: Sensor Simulation Challenges and Solutions**
   - File: `website/docs/Module-2/chapter-7-integrating-real-world-sensors-into-simulation/lesson-2-sensor-simulation-challenges-and-solutions.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Address challenges in accurately simulating sensors
   - Content: Discuss issues like noise modeling, calibration, realistic environmental conditions, and explore solutions
   - Reference tasks.md T025

3. **Lesson 7.3: Sim-to-Real Transfer Techniques**
   - File: `website/docs/Module-2/chapter-7-integrating-real-world-sensors-into-simulation/lesson-3-sim-to-real-transfer-techniques.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand techniques for sim-to-real transfer
   - Content: Explain sim-to-real transfer, challenges, and techniques like domain randomization and adaptation
   - Reference tasks.md T026

For each lesson: Follow lesson format template from spec.md section 2, add frontmatter from spec.md section 3.3 (include module: 2, chapter, lesson numbers). Create _category_.json files for each chapter. Use write tool (NOT Update) for all operations.
```

---

### Task 2.5: Fill Chapters 8-10 Content ✅

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapters 8-10 (9 lessons total):

**Chapter 8: Introduction to NVIDIA Isaac**
1. **Lesson 8.1: Introduction to NVIDIA Isaac**
   - File: `website/docs/Module-3/chapter-8-introduction-to-nvidia-isaac/lesson-1-introduction-to-nvidia-isaac.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Introduction to NVIDIA Isaac for AI in robotics
   - Content: Overview of NVIDIA Isaac, its components (Isaac Sim, Isaac SDK), and its role in AI robotics
   - Reference tasks.md T027

2. **Lesson 8.2: Using NVIDIA Isaac for Perception**
   - File: `website/docs/Module-3/chapter-8-introduction-to-nvidia-isaac/lesson-2-using-nvidia-isaac-for-perception.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Utilize NVIDIA Isaac for robot perception tasks
   - Content: Explain using NVIDIA Isaac for perception (object detection, segmentation), AI model integration
   - Reference tasks.md T028

3. **Lesson 8.3: NVIDIA Isaac for Robot Manipulation**
   - File: `website/docs/Module-3/chapter-8-introduction-to-nvidia-isaac/lesson-3-nvidia-isaac-for-robot-manipulation.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Employ NVIDIA Isaac for robot manipulation tasks
   - Content: Discuss using NVIDIA Isaac for manipulation (grasp planning, motion control), AI in dexterity
   - Reference tasks.md T029

**Chapter 9: AI Perception Techniques for Robots**
1. **Lesson 9.1: AI Perception Techniques for Robots**
   - File: `website/docs/Module-3/chapter-9-ai-perception-techniques-for-robots/lesson-1-ai-perception-techniques-for-robots.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand AI perception techniques for robots
   - Content: Detail AI perception techniques (deep learning for recognition, object detection, sensor fusion, 3D understanding)
   - Reference tasks.md T030

2. **Lesson 9.2: Sensor Fusion for Enhanced Perception**
   - File: `website/docs/Module-3/chapter-9-ai-perception-techniques-for-robots/lesson-2-sensor-fusion-for-enhanced-perception.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Implement sensor fusion for improved perception
   - Content: Explain sensor fusion importance, combining data from multiple sensors, algorithms like Kalman filters
   - Reference tasks.md T031

3. **Lesson 9.3: 3D Environment Perception**
   - File: `website/docs/Module-3/chapter-9-ai-perception-techniques-for-robots/lesson-3-3d-environment-perception.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand 3D environment perception for robots
   - Content: Cover 3D perception (point cloud processing, depth image analysis, SLAM basics)
   - Reference tasks.md T032

**Chapter 10: Reinforcement Learning in Robotics**
1. **Lesson 10.1: Reinforcement Learning in Robotics**
   - File: `website/docs/Module-3/chapter-10-reinforcement-learning-in-robotics/lesson-1-reinforcement-learning-in-robotics.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Apply reinforcement learning to robotic tasks
   - Content: Introduce RL fundamentals (states, actions, rewards, policies) and applications in robotics
   - Reference tasks.md T033

2. **Lesson 10.2: Sim-to-Real Transfer in RL**
   - File: `website/docs/Module-3/chapter-10-reinforcement-learning-in-robotics/lesson-2-sim-to-real-transfer-in-rl.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand sim-to-real transfer for reinforcement learning
   - Content: Discuss sim-to-real transfer challenges and techniques (domain randomization, adaptation)
   - Reference tasks.md T034

3. **Lesson 10.3: Deep Reinforcement Learning for Robotics**
   - File: `website/docs/Module-3/chapter-10-reinforcement-learning-in-robotics/lesson-3-deep-reinforcement-learning-for-robotics.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Explore deep reinforcement learning applications in robotics
   - Content: Introduce DRL algorithms (DQN, policy gradients, actor-critic) for robotics
   - Reference tasks.md T035

For each lesson: Follow lesson format template from spec.md section 2, add frontmatter from spec.md section 3.3 (include module: 3, chapter, lesson numbers). Create _category_.json files for each chapter. Use write tool (NOT Update) for all operations.
```

---

### Task 2.6: Fill Chapters 11-13 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapters 11-13 (9 lessons total):

**Chapter 11: Humanoid Robot Development**
1. **Lesson 11.1: Humanoid Robot Development**
   - File: `website/docs/Module-4/chapter-11-humanoid-robot-development/lesson-1-humanoid-robot-development.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand the fundamentals of humanoid robot development
   - Content: Overview of humanoid robot development challenges (structure, actuation, control)
   - Reference tasks.md T036

2. **Lesson 11.2: Humanoid Robot Kinematics and Dynamics**
   - File: `website/docs/Module-4/chapter-11-humanoid-robot-development/lesson-2-humanoid-robot-kinematics-and-dynamics.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand humanoid robot kinematics and dynamics
   - Content: Explain humanoid kinematics (forward/inverse) and dynamics (joint control, balance, stability), mathematical models
   - Reference tasks.md T037

3. **Lesson 11.3: Humanoid Robot Control and Balance**
   - File: `website/docs/Module-4/chapter-11-humanoid-robot-development/lesson-3-humanoid-robot-control-and-balance.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Learn about humanoid robot control and balance
   - Content: Discuss humanoid control, balance (ZMP), stable locomotion, feedback mechanisms
   - Reference tasks.md T038

**Chapter 12: Humanoid Robot Locomotion**
1. **Lesson 12.1: Humanoid Robot Locomotion**
   - File: `website/docs/Module-4/chapter-12-humanoid-robot-locomotion/lesson-1-humanoid-robot-locomotion.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand humanoid robot locomotion
   - Content: Explore legged locomotion principles (walking gaits, footstep planning, dynamic stability)
   - Reference tasks.md T039

2. **Lesson 12.2: Advanced Locomotion Planning**
   - File: `website/docs/Module-4/chapter-12-humanoid-robot-locomotion/lesson-2-advanced-locomotion-planning.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Learn advanced locomotion planning for humanoids
   - Content: Discuss advanced locomotion planning (trajectory optimization, uneven terrain, reactive behaviors)
   - Reference tasks.md T040

3. **Lesson 12.3: Humanoid Robot Manipulation and Interaction**
   - File: `website/docs/Module-4/chapter-12-humanoid-robot-locomotion/lesson-3-humanoid-robot-manipulation-and-interaction.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Explore humanoid robot manipulation and interaction
   - Content: Cover manipulation (grasping, object handling), interaction with environment/humans, integrating perception/control
   - Reference tasks.md T041

**Chapter 13: Visual-Language-Agent (VLA) Paradigm**
1. **Lesson 13.1: Visual-Language-Agent (VLA) Paradigm**
   - File: `website/docs/Module-4/chapter-13-visual-language-agent-vla-paradigm/lesson-1-visual-language-agent-vla-paradigm.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Understand the VLA paradigm for conversational robotics
   - Content: Introduce VLA concept (vision, language, action), how they enable natural language commands for robots
   - Reference tasks.md T042

2. **Lesson 13.2: Building VLA Systems**
   - File: `website/docs/Module-4/chapter-13-visual-language-agent-vla-paradigm/lesson-2-building-vla-systems.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Learn how to build VLA systems for robots
   - Content: Discuss VLA system components and architecture (visual features, language models, action policies)
   - Reference tasks.md T043

3. **Lesson 13.3: Conversational Robotics with VLAs**
   - File: `website/docs/Module-4/chapter-13-visual-language-agent-vla-paradigm/lesson-3-conversational-robotics-with-vlas.md`
   - Use context7 MCP tool to gather information
   - Learning objectives: Enable conversational interaction with robots using VLAs
   - Content: Explain how VLAs enable natural language interaction, commands, questions, feedback, challenges, future directions
   - Reference tasks.md T044

For each lesson: Follow lesson format template from spec.md section 2, add frontmatter from spec.md section 3.3 (include module: 4, chapter, lesson numbers). Create _category_.json files for each chapter. Use write tool (NOT Update) for all operations.
```

---

## ✅ PHASE 4: VERIFICATION

### Task 3.1: Verify All Content

**Model:** gemini-2.0-flash-lite  
**Command:** Manual review or `/sp.analyze`

**Prompt:**

```
Verify that all 13 chapters (39 lessons) have been created with complete content:

1. Check all files exist in website/docs/
2. Verify frontmatter format matches spec.md section 3.3
3. Verify learning objectives are included
4. Check code examples are properly formatted
5. Verify _category_.json files exist for each chapter
6. Check sidebar configuration in sidebars.ts matches chapter structure

Report any missing files or incomplete content.
```

---

## 📋 STATUS TRACKER

- ✅ Task 1.1: Specification created
- ✅ Task 1.2: Development plan created
- ✅ Task 1.3: Task breakdown created
- ⏳ Task 2.1: Chapter 1 content (Introduction to ROS 2)
- ⏳ Task 2.2: Chapter 2 content (ROS 2 Development Environment Setup)
- ⏳ Task 2.3: Chapters 3-4 content (URDF Models, Sensor Interfaces)
- ⏳ Task 2.4: Chapters 5-7 content (Gazebo, Unity, Sensor Integration)
- ⏳ Task 2.5: Chapters 8-10 content (NVIDIA Isaac, AI Perception, RL)
- ⏳ Task 2.6: Chapters 11-13 content (Humanoid Development, Locomotion, VLA)
- ⏳ Task 3.1: Verification

**NEXT ACTION:** Run Task 2.1 prompt with `/sp.implement` command to fill Chapter 1 content
