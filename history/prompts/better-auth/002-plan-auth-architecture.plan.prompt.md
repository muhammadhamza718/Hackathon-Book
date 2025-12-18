---
id: 5002
title: Designing Serverless + Dual-Port Auth Architecture
stage: plan
date_iso: 2024-11-21
surface: agent
model: claude-3-5-sonnet
feature: better-auth
branch: main
user: muhammadhamza718
command: /sp.plan
labels: ["architecture", "serverless", "dual-port", "better-auth", "mermaid"]
links_spec: "specs/003-better-auth/spec.md"
links_plan: "specs/003-better-auth/plan.md"
---

# PROMPT

/sp.plan @specs/003-better-auth/spec.md --agents auth-architect

Design the implementation plan for Step 5. Focus on the architectural challenges of Docusaurus.

**Requirements for the Plan**:

1. **The "Clean Break"**: How do we prevent Docusaurus from trying to bundle server-side libraries like `better-auth` and `pg` into the browser bundle?
2. **Local Development**: We need a reliable way to test auth locally since `vercel dev` can be inconsistent with ports. Let's design a dual-server setup: Docusaurus on 3000 and a dedicated Express Auth Server on 7860.
3. **Subagent Workflow**: Explicitly use the **Auth Architect** agent to design the `additionalFields` database schema for the background questionnaire.

Create a detailed `plan.md` with Mermaid diagrams.

# RESPONSE

I have formulated the "Clean Break" architectural strategy. By moving all auth logic to the root `/api` folder and using a standalone `auth-server.js` on Port 7860 for development, we completely isolate the server logic from the Docusaurus frontend build.

The `plan.md` now includes a detailed ER diagram for the user table extensions and a subagent workflow that leverages the **Auth Architect** for schema design and the **UI Designer** for "De-gamifying" the CSS modules.

# OUTCOME

- Decoupled serverless architecture designed.
- Reliable Port 7860/3000 connectivity strategy established.
- Step 4 subagent integration points defined for the implementation phase.
