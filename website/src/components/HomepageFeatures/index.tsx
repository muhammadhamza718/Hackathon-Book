import type { ReactNode, CSSProperties } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
  link: string;
  color: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Module 1: The Robotic Nervous System",
    emoji: "🧠",
    description: (
      <>
        Master ROS 2 fundamentals, development environment setup, URDF models,
        and sensor interfaces. Build the foundation for robotic communication
        and control.
      </>
    ),
    link: "/docs/Module-1/chapter-1-introduction-to-ros-2/lesson-1-introduction-to-ros-2",
    color: "#3578e5",
  },
  {
    title: "Module 2: The Digital Twin",
    emoji: "🌐",
    description: (
      <>
        Learn Gazebo simulation, Unity visualization, and real-world sensor
        integration. Create realistic digital twins of robotic systems.
      </>
    ),
    link: "/docs/Module-2/chapter-5-introduction-to-gazebo/lesson-1-introduction-to-gazebo",
    color: "#25c2a0",
  },
  {
    title: "Module 3: The AI-Robot Brain",
    emoji: "🤖",
    description: (
      <>
        Explore NVIDIA Isaac platform, AI perception techniques, and
        reinforcement learning. Build intelligent robotic systems with advanced
        AI capabilities.
      </>
    ),
    link: "/docs/Module-3/chapter-8-introduction-to-nvidia-isaac/lesson-1-introduction-to-nvidia-isaac",
    color: "#ff6b6b",
  },
  {
    title: "Module 4: Vision-Language-Action",
    emoji: "🚶",
    description: (
      <>
        Develop humanoid robots, master locomotion, and implement the VLA
        paradigm. Create conversational and intelligent humanoid systems.
      </>
    ),
    link: "/docs/Module-4/chapter-11-humanoid-robot-development/lesson-1-humanoid-robot-development",
    color: "#9b59b6",
  },
];

function Feature({ title, emoji, description, link, color }: FeatureItem) {
  return (
    <div className={clsx("col col--3 margin-bottom--lg")}>
      <div
        className={clsx("card", styles.featureCard)}
        style={{ "--feature-color": color } as CSSProperties}
      >
        <div className="card__header">
          <div className={styles.emojiContainer}>{emoji}</div>
          <Heading as="h3" className={styles.title}>
            {title}
          </Heading>
        </div>
        <div className="card__body">
          <div className={styles.description}>{description}</div>
        </div>
        <div className="card__footer">
          <Link
            className={clsx(
              "button",
              "button--primary",
              "button--block",
              styles.button
            )}
            to={link}
          >
            Start Learning →
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className={styles.container}>
        <div className="text--center margin-bottom--lg">
          <Heading as="h2">Explore the Complete Curriculum</Heading>
          <p className={styles.subtitle}>
            Four comprehensive modules covering everything from ROS 2 basics to
            advanced humanoid robotics
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
