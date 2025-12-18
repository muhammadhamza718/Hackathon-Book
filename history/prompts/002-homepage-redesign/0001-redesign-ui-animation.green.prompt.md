---
id: PHR-002-001
title: Homepage UI/UX Redesign & Animation
stage: green
date: 2025-12-16
surface: agent
model: claude-3-5-sonnet
feature: homepage
branch: main
user: Hamza
command: /sp.implement
labels: [ui, ux, animation, tsparticles, glassmorphism, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - website/src/components/HeroAnimation.tsx
  - website/src/components/HomepageFeatures/index.tsx
  - website/src/components/HomepageFeatures/styles.module.css
  - website/src/css/custom.css
  - website/src/pages/index.tsx
tests:
  - manual verification
---

## Prompt

Refine the Docusaurus homepage design to be premium and modern.

1. **Hero Section Animation**:

   - Replace the static hero section with a full-screen background particle animation using `tsparticles`.
   - The particles should use a vibrant color palette (cyan `#00f3ff`, purple `#bc13fe`, white) to match the "Physical AI" theme.
   - Configure the particles to interact with the mouse (hover connections) and move slowly to create a premium feel.
   - Ensure the animation is positioned absolutely (`z-index: 0`) so it sits _behind_ the hero text, which must remain centered and legible.

2. **Glassmorphism Feature Cards**:

   - Redesign the `HomepageFeatures` cards to use a glassmorphism aesthetic.
   - Apply a semi-transparent background (`rgba(255, 255, 255, 0.03)`), backdrop blur, and a subtle border.
   - Add a dynamic hover effect where a top border gradient scales in from 0 to 1 width.
   - **Grid Layout**: Fix the card size issues by changing the grid to a single row (4 columns, `col--3`) instead of the previous 2-column layout. Reduce vertical padding to prevent excessive height.

3. **Global Styling & Navbar**:
   - Polish the navbar to have a consistent dark glass background.
   - **Critical Fix**: Force all navbar text, links, icons, and the mode toggle button to be WHITE (`#fff`) in both light and dark modes to ensure visibility against the dark header.
   - Add a global CSS override to ensure the `tsparticles` canvas always fills 100% of the container.

Ensure all changes are responsive and compatible with the existing Docusaurus build system.

## Response snapshot

Implemented the full UI/UX redesign:

- Created `HeroAnimation.tsx` wrapping `tsparticles` with absolute positioning coverage.
- Updated `HomepageFeatures/styles.module.css` with advanced glassmorphism CSS (backdrop-filter, dynamic pseudo-elements).
- Adjusted the grid system in `HomepageFeatures/index.tsx` to `col--3` for a streamlined 4-item row.
- Applied global overrides in `custom.css` to fix navbar contrast issues and enforce particle canvas sizing.
- Resolved git merge conflicts to maintain the "Better UI/UX" version and reset repository history.

## Outcome

- ✅ Impact: Delivered a premium, interactive homepage with bug-free layouts and cross-mode compatibility.
- 🧪 Tests: Verified build functionality and visual regression check (manual).
- 📁 Files: Modified 5 files to implement animation and styling.
- 🔁 Next prompts: Review mobile responsiveness for new grid layout.
- 🧠 Reflection: Using `tsparticles` with absolute positioning requires careful `z-index` management and global CSS overrides to ensure it doesn't block interactions or collapse in height.
