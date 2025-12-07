import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "Physical AI & Humanoid Robotics",
  tagline: "Master ROS 2, Simulation, AI, and Humanoid Robot Development",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  // Update these for your GitHub Pages deployment
  url: "https://hackathon-book-mwoqt91dl-muhammad-hamzas-projects-88f218fc.vercel.app/", // Replace with your GitHub username
  baseUrl: "/", // Replace with your repo name if different

  // GitHub pages deployment config
  organizationName: "muhammadhamzas-projects", // Replace with your GitHub username
  projectName: "hackathon-book", // Replace with your repo name

  onBrokenLinks: "throw",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/muhammadhamza718/Hackathon-Book/tree/main/", // Update with your repo
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          editUrl:
            "https://github.com/muhammadhamza718/Hackathon-Book/tree/main/", // Update with your repo
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      defaultMode: "light",
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: "Physical AI & Humanoid Robotics",
      logo: {
        alt: "Physical AI & Humanoid Robotics Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "📖 Book",
        },
        { to: "/blog", label: "Blog", position: "left" },
        {
          type: "search",
          position: "right",
        },
        {
          href: "https://github.com/muhammadhamza718/Hackathon-Book", // Update with your repo
          label: "GitHub",
          position: "right",
          className: "header-github-link",
          "aria-label": "GitHub repository",
        },
      ],
      hideOnScroll: true,
    },

    footer: {
      style: "dark",
      links: [
        {
          title: "Book Modules",
          items: [
            {
              label: "Module 1: ROS 2",
              to: "/docs/Module-1/chapter-1-introduction-to-ros-2/lesson-1-introduction-to-ros-2",
            },
            {
              label: "Module 2: Simulation",
              to: "/docs/Module-2/chapter-5-introduction-to-gazebo/lesson-1-introduction-to-gazebo",
            },
            {
              label: "Module 3: AI & Isaac",
              to: "/docs/Module-3/chapter-8-introduction-to-nvidia-isaac/lesson-1-introduction-to-nvidia-isaac",
            },
            {
              label: "Module 4: Humanoids",
              to: "/docs/Module-4/chapter-11-humanoid-robot-development/lesson-1-humanoid-robot-development",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "Blog",
              to: "/blog",
            },
            {
              label: "GitHub",
              href: "https://github.com/your-username/hackathon-book", // Update
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["python", "cpp", "bash", "yaml", "json"],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
