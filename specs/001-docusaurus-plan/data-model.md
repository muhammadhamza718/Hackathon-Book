# Content Structure

The Docusaurus website will be structured as follows:

-   **Chapters:** Each chapter will be a directory under the `docs` directory.
-   **Lessons:** Each lesson will be a Markdown file under the chapter directory.
-   **Images:** Images will be stored in the `static/img` directory.
-   **Code Examples:** Code examples will be stored in the `static/code-examples` directory.

The `_category_.json` file will be used to define the order and labels of chapters and lessons in the sidebar.

## Frontmatter Template

Each lesson Markdown file will include the following frontmatter:

```markdown
---
id: lesson-N-lesson-title
title: Lesson N: Lesson Title
sidebar_label: Lesson Title
sidebar_position: N
---
```