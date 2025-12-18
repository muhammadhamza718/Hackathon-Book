# Feature Specification: Better Auth Authentication & Background Profiling

**Feature Branch**: `3-better-auth`
**Created**: 2025-12-16
**Source**: [Hackathon Textbook - Step 5 (Bonus Points)](../../Hackathon%20I_%20Physical%20AI%20&%20Humanoid%20Robotics%20Textbook.md)

## Feature Overview

Implement a robust, production-grade authentication system using **Better Auth** (v1.x) that facilitates a personalized learning experience. The core of this feature is a mandatory **Background Questionnaire** during signup that captures the user's technical proficiency across software, hardware, and AI. This data is critical for satisfying the "Personalized Content" bonus requirements (Steps 6 & 7).

The UI must depart from the initial "cyberpunk/gamer" aesthetic of the project's hero section, adopting a "Normal Professional" design that is clean, accessible (WCAG AA), and fully compatible with both light and dark themes.

## User Scenarios & Testing

### User Story 1 - Technical Onboarding (Priority: P1)

**As a** student or researcher,
**I want to** provide my technical background during the account creation process,
**So that** the platform can tailor its content to my current skill level.

**Acceptance Criteria**:

1. **Signup Accessibility**: One-click access from the main navigation bar.
2. **Personalization Fields**: The signup form MUST include:
   - `softwareExperience`: Enum (Beginner, Intermediate, Advanced, Expert)
   - `hardwareExperience`: Enum (None, Hobbyist, Professional)
   - `aiMlFamiliarity`: Enum (Novice, Familiar, Practitioner, Researcher)
   - `learningGoals`: Enum (Career Pivot, Skill Upgrade, Academic, Interest)
   - `programmingLanguages`: Free-text/Comma-separated input.
3. **Mandatory Completion**: User cannot proceed without completing the background profile.
4. **Data Persistence**: Data must be stored in the `user` table via Better Auth's `additionalFields` plugin, connected to a Neon Serverless Postgres instance.

### User Story 2 - Professional Interface (Priority: P1)

**As a** user who values clarity and accessibility,
**I want to** interact with authentication forms that follow modern professional design patterns,
**So that** I am not distracted by excessive visual effects like glows or high-contrast neon.

**Acceptance Criteria**:

1. **Zero-Glow Mandate**: All `box-shadow` and `text-shadow` glows MUST be removed from `SignupForm` and `SigninForm`.
2. **Typography**: Use standard professional sans-serif fonts with clear hierarchy.
3. **Theme Logic**:
   - **Light Mode**: Black text on white/light-gray surfaces.
   - **Dark Mode**: White text on dark-charcoal surfaces.
4. **Feedback**: Clear, professional error messages for invalid credentials or missing fields.

### User Story 3 - Profile Transparency (Priority: P2)

**As a** registered user,
**I want to** view my background information on a dediated profile page,
**So that** I can verify how the system is personalizing my experience.

**Acceptance Criteria**:

1. **Profile Navigation**: "Profile" link in the user dropdown menu.
2. **Data Presentation**: Technical background fields displayed in a professional table format.
3. **Responsive**: Profile page must be mobile-friendly.

## Functional Requirements

- **FR-01**: Implementation must use `better-auth` v1.x with `pg` driver.
- **FR-02**: Auth API must be decoupled from the Docusaurus frontend build using the root `/api` folder strategy (Vercel Serverless Functions).
- **FR-03**: Password hashing and secure session management (HTTP-Only cookies) must be enabled by default.
- **FR-04**: Use `additionalFields` plugin for the 5 technical background markers.

## Non-Functional Requirements (UI/UX)

- **NFR-01**: Designs must be clean, minimalist, and "Stripe-like".
- **NFR-02**: All interactive elements must have unique IDs for automated testing.
- **NFR-03**: Accessibility compliance with WCAG AA standards.

## Success Metrics

- **Data Integrity**: 100% of newly registered users have values for all 5 background fields.
- **Build Reliability**: Stable Docusaurus build without Node.js polyfill conflicts.
- **Visual Harmony**: Auth forms integrate perfectly with Docusaurus's native theme switcher.
