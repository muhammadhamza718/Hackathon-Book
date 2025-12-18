---
id: 5005
title: Resolving Signup 404 Not Found in Serverless Environment
stage: red
date_iso: 2024-11-24
surface: agent
model: claude-3-5-sonnet
feature: better-auth
branch: main
user: muhammadhamza718
command: /sp.fix
labels: ["debugging", "404", "routing", "vercel-serverless", "import-aliases"]
links_spec: "specs/003-better-auth/spec.md"
files_yaml:
  - website/api/auth/[...all].ts
tests_yaml:
  - API endpoint reachability check (POST /api/auth/signup)
---

# PROMPT

/sp.fix "Signup failing with 404 Not Found in browser console"

I just finished the implementation and tried a test signup, but the browser is throwing a 404 error: `POST http://localhost:3000/api/auth/sign-up/email 404 (Not Found)`.

I verified the file is in `website/api/auth/[...all].ts`. Could it be an issue with how Vercel or `vercel dev` handles the directory? Also, I see some `@/lib/auth` imports in that file—does the serverless runtime support those aliases?

# RESPONSE

I have diagnosed the routing failure. The 404 error was caused by two issues:

1. **Alias Resolution**: The serverless runtime in the root `/api` folder cannot resolve the `@/` alias which is configured relative to the `website/src/` folder. I have updated all imports in `[...all].ts` to use relative paths (`../../src/lib/auth`).
2. **Directory Structure**: Confirmed that the Vercel function routing is strictly tied to the project root's `api` folder, and our pathing is now compliant.

# OUTCOME

- Fixed 404 Not Found error by correcting serverless import logic.
- Verified API connectivity on dynamic catch-all route.
- Established relative path standard for serverless functions.
