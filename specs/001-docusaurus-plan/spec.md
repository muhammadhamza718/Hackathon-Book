# Specification: Physical AI & Humanoid Robotics Book in Docusaurus

This document outlines the Docusaurus-specific implementation specification for the "Physical AI & Humanoid Robotics" book, covering setup, configuration, and file structure aligned with the 4-module organization.

## 1. Docusaurus Setup and Configuration

This section details the initial setup and configuration of the Docusaurus project for the "Physical AI & Humanoid Robotics" book.

### 1.1 Docusaurus Installation and Initialization

- Step-by-step guide for installing Node.js, npm, and the Docusaurus CLI.
- Command for initializing a new Docusaurus project with TypeScript support.
- Basic project structure after initialization.

### 1.2 Required Configuration Files

- **`docusaurus.config.ts`**:
  - Site metadata (title, tagline, URL, baseUrl).
  - Theme configuration (navbar, footer, color mode).
  - Plugin configuration (docs, blog, pages).
  - Custom webpack configurations if needed.
- **`sidebars.ts`**:
  - Definition of the sidebar structure organized by modules, chapters, and lessons.
  - Nesting structure: Module → Chapter → Lessons.
  - Custom labels and links for navigation.

### 1.3 Theme Customization

- Override default Docusaurus theme components (e.g., `Navbar`, `Footer`, `DocItem`).
- Custom CSS for styling components, code blocks, and overall layout to match the book's aesthetic.
- Font integration and typography settings.

### 1.4 Plugin Setup and Dependencies

- Installation and configuration of `@docusaurus/plugin-content-docs` for managing book content.
- Integration of search plugins (e.g., Algolia DocSearch or local search).
- Any other necessary plugins for features like sitemaps, PWA, etc.
- Listing of `package.json` dependencies for Docusaurus and its plugins.

## 2. Content Development Phases

This section outlines the phased approach for creating the book's content, structured around 4 modules, 13 chapters, and 39 lessons.

### 2.1 Phase Breakdown (Aligned with Modules)

- **Module 1: The Robotic Nervous System (ROS 2) - Chapters 1-4**: Introduction to ROS 2, development environment setup, URDF models, and sensor interfaces.
- **Module 2: The Digital Twin (Gazebo & Unity) - Chapters 5-7**: Gazebo simulation, Unity visualization, and real-world sensor integration.
- **Module 3: The AI-Robot Brain (NVIDIA Isaac) - Chapters 8-10**: NVIDIA Isaac platform, AI perception techniques, and reinforcement learning.
- **Module 4: Vision-Language-Action (VLA) - Chapters 11-13**: Humanoid robot development, locomotion, and VLA paradigm.

### 2.2 Prioritization and Sequencing

- Modules will be sequenced linearly based on logical dependencies (ROS 2 foundation → Simulation → AI → VLA).
- Chapters within each module will be developed sequentially.
- Each chapter's 3 lessons will be developed sequentially within that chapter.

### 2.3 Milestones and Deliverables

- **Per Chapter**:
  - Outline complete (lesson titles, key topics).
  - First draft of all lesson content.
  - Technical review and feedback incorporation.
  - Final polished content.
- **Per Module**:
  - All chapters within the module completed and reviewed.
  - Integrated Docusaurus build for the module content.
  - Module transition content added.

### 2.4 Timeline Estimates

- **Module 1 (Chapters 1-4)**: 4 weeks
- **Module 2 (Chapters 5-7)**: 3 weeks
- **Module 3 (Chapters 8-10)**: 3 weeks
- **Module 4 (Chapters 11-13)**: 3 weeks
- **Overall Review and Publishing**: 2 weeks

## 3. File Structure for Chapters and Lessons

This section defines the file organization within the Docusaurus project to accommodate the book's Module-based content structure.

### 3.1 Complete Directory Structure

```
website/docs/
├── Module-1/                    # The Robotic Nervous System (ROS 2)
│   ├── chapter-1-introduction-to-ros-2/
│   │   ├── _category_.json
│   │   ├── lesson-1-introduction-to-ros-2.md
│   │   ├── lesson-2-ros-2-architecture.md
│   │   └── lesson-3-core-ros-2-concepts.md
│   ├── chapter-2-ros-2-development-environment-setup/
│   │   ├── _category_.json
│   │   ├── lesson-1-ros-2-development-environment-setup.md
│   │   ├── lesson-2-creating-a-ros-2-package.md
│   │   └── lesson-3-building-and-running-ros-2-code.md
│   ├── chapter-3-understanding-and-creating-urdf-models/
│   │   ├── _category_.json
│   │   ├── lesson-1-understanding-and-creating-urdf-models.md
│   │   ├── lesson-2-urdf-for-robot-kinematics.md
│   │   └── lesson-3-advanced-urdf-features.md
│   └── chapter-4-working-with-ros-2-sensor-interfaces/
│       ├── _category_.json
│       ├── lesson-1-working-with-ros-2-sensor-interfaces.md
│       ├── lesson-2-ros-2-launch-files.md
│       └── lesson-3-sensor-data-processing-in-ros-2.md
├── Module-2/                    # The Digital Twin (Gazebo & Unity)
│   ├── chapter-5-introduction-to-gazebo/
│   │   ├── _category_.json
│   │   ├── lesson-1-introduction-to-gazebo.md
│   │   ├── lesson-2-using-gazebo-with-ros-2.md
│   │   └── lesson-3-gazebo-plugins-and-world-files.md
│   ├── chapter-6-advanced-simulation-with-unity/
│   │   ├── _category_.json
│   │   ├── lesson-1-advanced-simulation-with-unity.md
│   │   ├── lesson-2-unity-for-realistic-physics-simulation.md
│   │   └── lesson-3-integrating-ros-2-with-unity.md
│   └── chapter-7-integrating-real-world-sensors-into-simulation/
│       ├── _category_.json
│       ├── lesson-1-integrating-real-world-sensors-into-simulation.md
│       ├── lesson-2-sensor-simulation-challenges-and-solutions.md
│       └── lesson-3-sim-to-real-transfer-techniques.md
├── Module-3/                    # The AI-Robot Brain (NVIDIA Isaac)
│   ├── chapter-8-introduction-to-nvidia-isaac/
│   │   ├── _category_.json
│   │   ├── lesson-1-introduction-to-nvidia-isaac.md
│   │   ├── lesson-2-using-nvidia-isaac-for-perception.md
│   │   └── lesson-3-nvidia-isaac-for-robot-manipulation.md
│   ├── chapter-9-ai-perception-techniques-for-robots/
│   │   ├── _category_.json
│   │   ├── lesson-1-ai-perception-techniques-for-robots.md
│   │   ├── lesson-2-sensor-fusion-for-enhanced-perception.md
│   │   └── lesson-3-3d-environment-perception.md
│   └── chapter-10-reinforcement-learning-in-robotics/
│       ├── _category_.json
│       ├── lesson-1-reinforcement-learning-in-robotics.md
│       ├── lesson-2-sim-to-real-transfer-in-rl.md
│       └── lesson-3-deep-reinforcement-learning-for-robotics.md
└── Module-4/                    # Vision-Language-Action (VLA)
    ├── chapter-11-humanoid-robot-development/
    │   ├── _category_.json
    │   ├── lesson-1-humanoid-robot-development.md
    │   ├── lesson-2-humanoid-robot-kinematics-and-dynamics.md
    │   └── lesson-3-humanoid-robot-control-and-balance.md
    ├── chapter-12-humanoid-robot-locomotion/
    │   ├── _category_.json
    │   ├── lesson-1-humanoid-robot-locomotion.md
    │   ├── lesson-2-advanced-locomotion-planning.md
    │   └── lesson-3-humanoid-robot-manipulation-and-interaction.md
    └── chapter-13-visual-language-agent-vla-paradigm/
        ├── _category_.json
        ├── lesson-1-visual-language-agent-vla-paradigm.md
        ├── lesson-2-building-vla-systems.md
        └── lesson-3-conversational-robotics-with-vlas.md
static/
├── img/             # For general images (diagrams, figures)
└── code/            # For downloadable code examples
```

### 3.2 Naming Conventions

- **Module Directories**: `Module-1/`, `Module-2/`, `Module-3/`, `Module-4/`
- **Chapter Directories**: `chapter-N-chapter-title` (e.g., `chapter-1-introduction-to-ros-2`).
- **Lesson Files**: `lesson-N-lesson-title.md` (e.g., `lesson-1-introduction-to-ros-2.md`).
- **Category Files**: `_category_.json` for defining chapter titles and positions in the sidebar.

### 3.3 Frontmatter Template for Lesson Files

Each lesson Markdown file will include the following frontmatter:

```yaml
---
title: "Lesson Title"
description: "Brief description of the lesson"
chapter: <chapter_number>
lesson: <lesson_number>
module: <module_number>
sidebar_label: "Short Label"
sidebar_position: <position>
tags: ["tag1", "tag2"]
keywords: ["keyword1", "keyword2"]
---
```

**Required fields:**

- `title`: Full lesson title
- `description`: Brief description
- `chapter`: Chapter number (1-13)
- `lesson`: Lesson number within chapter (1-3)
- `module`: Module number (1-4) - **CRITICAL: Must be included**
- `sidebar_label`: Short label for sidebar navigation
- `sidebar_position`: Position within chapter sidebar
- `tags`: Array of topic tags
- `keywords`: Array of keywords for search

### 3.4 Code Example and Asset Organization Structure

- **Images**: Stored in `static/img/` and referenced relatively in Markdown files.
- **Code Examples**: Stored in `static/code/` and linked as downloadable files or embedded using Docusaurus's code block features.
- **Videos/Multimedia**: If applicable, stored in `static/media/` and embedded.

### 3.5 Module Organization in Sidebar

The sidebar configuration should reflect the Module-based hierarchy:

```typescript
module.exports = {
  tutorialSidebar: [
    {
      type: "category",
      label: "Module 1: The Robotic Nervous System (ROS 2)",
      items: [
        {
          type: "category",
          label: "Chapter 1: Introduction to ROS 2",
          items: [
            "Module-1/chapter-1-introduction-to-ros-2/lesson-1-introduction-to-ros-2",
            "Module-1/chapter-1-introduction-to-ros-2/lesson-2-ros-2-architecture",
            "Module-1/chapter-1-introduction-to-ros-2/lesson-3-core-ros-2-concepts",
          ],
        },
        // ... more chapters
      ],
    },
    // ... more modules
  ],
};
```
