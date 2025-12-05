---
id: 0007
title: User Greeting
stage: general
date: 2025-12-04
surface: agent
model: gemini-2.0-flash-live
feature: none
branch: master
user: user
command: hi
labels: ["greeting"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:

tests:

---

## Prompt

hi

## Response snapshot

Hello! How can I help you today?

## Outcome

- ✅ Impact: User greeted, PHR created.
- 🧪 Tests: N/A
- 📁 Files: N/A
- 🔁 Next prompts: User's next question
- 🧠 Reflection: Responded to user's greeting and created a PHR as per instructions.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
