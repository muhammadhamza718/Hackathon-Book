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
     * Exact file path: `website/docs/chapter-N-title/lesson-N-title.md`
     * Content requirements from Physical AI & Humanoid Robotics Textbook + spec.md
     * Learning objectives from spec.md section 1
     * Frontmatter format from spec.md section 3.3
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

### Task 2.1: Fill Chapter 1 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapter 1 (Introduction to Physical AI):

1. **Lesson 1.1: Foundations of Physical AI**
   - File: `website/docs/chapter-1-introduction/lesson-1-foundations-of-physical-ai.md`
   - Use context7 MCP tool to research Physical AI foundations
   - Include learning objectives from spec.md
   - Follow lesson format template from spec.md section 2
   - Add frontmatter from spec.md section 3.3
   - Reference Physical AI & Humanoid Robotics Textbook.md weeks 1-2 content

2. **Lesson 1.2: Embodied Intelligence**
   - File: `website/docs/chapter-1-introduction/lesson-2-embodied-intelligence.md`
   - Research embodied intelligence principles
   - Include hands-on learning elements

3. **Lesson 1.3: Humanoid Robotics Landscape**
   - File: `website/docs/chapter-1-introduction/lesson-3-humanoid-robotics-landscape.md`
   - Research current humanoid robotics landscape
   - Include real-world applications

4. **Create _category_.json** for chapter-1-introduction folder

Use the write tool (NOT Update) for all file operations.
```

---

### Task 2.2: Fill Chapter 2 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Based on tasks.md, Physical AI & Humanoid Robotics Textbook, and spec.md, fill content for Chapter 2 (Sensor Systems):

1. **Lesson 2.1: LIDAR Sensors** → `website/docs/chapter-2-sensor-systems/lesson-1-lidar-sensors.md`
2. **Lesson 2.2: Camera Sensors** → `website/docs/chapter-2-sensor-systems/lesson-2-camera-sensors.md`
3. **Lesson 2.3: IMU and Force/Torque Sensors** → `website/docs/chapter-2-sensor-systems/lesson-3-imu-and-force-torque-sensors.md`
4. **Create _category_.json** for chapter-2-sensor-systems

Use context7 to research sensor technologies. Reference textbook sensor systems section. Use write tool for all operations.
```

---

### Task 2.3: Fill Chapters 3-4 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Fill content for Chapters 3-4 (6 lessons total):
- Chapter 3: ROS 2 Fundamentals (3 lessons)
- Chapter 4: Launch Files and Parameter Management (3 lessons)

Follow same format as previous chapters. Use context7 to research ROS 2 topics. Reference Physical AI & Humanoid Robotics Textbook.md weeks 3-5 content. Use write tool for all operations.
```

---

### Task 2.4: Fill Chapters 5-7 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Fill content for Chapters 5-7 (9 lessons total):
- Chapter 5: Robot Simulation with Gazebo (3 lessons)
- Chapter 6: Introduction to Unity for Robot Visualization (3 lessons)
- Chapter 7: NVIDIA Isaac SDK and Isaac Sim (3 lessons)

Reference Physical AI & Humanoid Robotics Textbook.md Module 2 and Module 3 content. Use context7 for Gazebo, Unity, and Isaac Sim research. Use write tool for all operations.
```

---

### Task 2.5: Fill Chapters 8-10 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Fill content for Chapters 8-10 (9 lessons total):
- Chapter 8: AI-powered perception and manipulation (3 lessons)
- Chapter 9: Reinforcement learning for robot control (3 lessons)
- Chapter 10: Sim-to-real transfer techniques (3 lessons)

Reference Physical AI & Humanoid Robotics Textbook.md Module 3 content. Use context7 for AI perception, RL, and sim-to-real research. Use write tool for all operations.
```

---

### Task 2.6: Fill Chapters 11-13 Content

**Model:** gemini-2.0-flash-lite  
**Command:** `/sp.implement` or manual

**Prompt:**

```
Fill content for Chapters 11-13 (9 lessons total):
- Chapter 11: Humanoid robot kinematics and dynamics (3 lessons)
- Chapter 12: Bipedal locomotion and balance control (3 lessons)
- Chapter 13: Manipulation and grasping with humanoid hands and Natural human-robot interaction design (3 lessons)

Reference Physical AI & Humanoid Robotics Textbook.md Module 4 and weeks 11-13 content. Use context7 for humanoid robotics, VLA, and conversational AI research. Use write tool for all operations.
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
- ⏳ Task 1.3: Task breakdown (NEXT)
- ⏳ Task 2.1: Chapter 1 content
- ⏳ Task 2.2: Chapter 2 content
- ⏳ Task 2.3: Chapters 3-4 content
- ⏳ Task 2.4: Chapters 5-7 content
- ⏳ Task 2.5: Chapters 8-10 content
- ⏳ Task 2.6: Chapters 11-13 content
- ⏳ Task 3.1: Verification

**NEXT ACTION:** Run Task 1.3 prompt with `/sp.tasks` command
