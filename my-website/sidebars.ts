import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 *
 * Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "intro",
    {
      type: "category",
      label: "Module 1: Foundations",
      items: [
        "module-1-foundations/overview",
        "module-1-foundations/calculus",
        "module-1-foundations/calculus-exercises",
        "module-1-foundations/linalg",
        "module-1-foundations/linalg-exercises",
        "module-1-foundations/probability",
        "module-1-foundations/probability-exercises",
      ],
    },
    {
      type: "category",
      label: "Module 2: ROS 2",
      items: [
        "module-2-ros-2/overview",
        "module-2-ros-2/nodes-topics-services",
        "module-2-ros-2/urdf",
        "module-2-ros-2/rclpy-bridging",
      ],
    },
    {
      type: "category",
      label: "Module 3: Digital Twin",
      items: [
        "module-3-simulation/overview",
        "module-3-simulation/physics-sim",
        "module-3-simulation/gazebo-physics",
        "module-3-simulation/rendering-interaction",
        "module-3-simulation/unity-rendering",
        "module-3-simulation/sensors",
        "module-3-simulation/sensor-simulation",
      ],
    },
    {
      type: "category",
      label: "Module 4: VLA",
      items: [
        "module-4-vla/overview",
        "module-4-vla/whisper",
        "module-4-vla/nav2",
        "module-4-vla/isaac-sim",
        "module-4-vla/isaac-ros",
        "module-4-vla/cognitive-planning",
        "module-4-vla/capstone",
      ],
    },
    {
      type: "category",
      label: "Hardware and Assessments",
      items: [
        "hardware-requirements/hardware",
        "assessments",
        "learning-outcomes",
      ],
    },
  ],
};

export default sidebars;
