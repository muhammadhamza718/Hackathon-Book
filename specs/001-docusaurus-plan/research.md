# Research

This document consolidates research findings for setting up a Docusaurus website for the Physical AI & Humanoid Robotics book.

## Technical Stack

- **Decision:** Use TypeScript (Node.js v18.0+ or v20.0+), React v18.0+, MDX v3.0+, TypeScript v5.1+, prism-react-renderer v2.0+, react-live v4.0+, remark-emoji v4.0+, mermaid v10.4+.
- **Rationale:** These technologies are the foundation of Docusaurus and provide the necessary tools for creating a modern, interactive documentation website.
- **Alternatives considered:** None. These are the standard technologies for Docusaurus.

## Configuration Options

- **Decision:** Use standard Docusaurus configuration options for `docusaurus.config.ts` and `sidebars.ts`.
- **Rationale:** The standard options provide the necessary flexibility for configuring the website. Custom options can be added as needed.
- **Alternatives considered:** None. The standard options are sufficient.

## Theme Customization

- **Decision:** Use swizzling to override default Docusaurus theme components and custom CSS for styling.
- **Rationale:** Swizzling allows for fine-grained control over the website's appearance. Custom CSS provides additional styling options.
- **Alternatives considered:** None. These are the standard methods for theme customization.

## Plugin Setup

- **Decision:** Use `@docusaurus/plugin-content-docs` for managing book content and Algolia DocSearch for search.
- **Rationale:** `@docusaurus/plugin-content-docs` is the standard plugin for managing documentation content. Algolia DocSearch provides a powerful search experience.
- **Alternatives considered:** Local search plugin. Algolia DocSearch provides a better search experience, so it is preferred.
