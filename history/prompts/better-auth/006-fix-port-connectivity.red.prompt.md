---
id: 5006
title: Troubleshooting Cross-Port Connectivity (3000 -> 7860)
stage: red
date_iso: 2024-11-25
surface: agent
model: qwen3-coder-plus
feature: better-auth
branch: main
user: muhammadhamza718
command: /sp.fix
labels:
  ["debugging", "port-conflict", "cross-origin", "baseURL", "error-unmasking"]
links_spec: "specs/003-better-auth/spec.md"
files_yaml:
  - website/src/lib/auth-client.ts
  - website/auth-server.js
  - website/src/hooks/useAuth.ts
tests_yaml:
  - Cross-port signup test (3000 to 7860)
  - Detail error logging check
---

# PROMPT

/sp.fix "Signup still keeps failing with 'Signup Failed' generic message"

The 404 is fixed, but now I get a generic "Signup Failed" message in my `SignupForm` UI. There are no details!

I'm running `vercel dev` on **Port 3000** and the `auth-server.js` on **Port 7860**. I think the frontend is still trying to talk to 3000, but there's no server there to handle the DB write. How do we tell the `authClient` to talk to 7860 while still working on Vercel later?

Also, can you unmask the "Signup Failed" error so I can see what's actually happening?

# RESPONSE

I have identified and resolved the connectivity breakdown between the frontend and the local auth server.

1. **Dynamic baseURL**: I refactored `auth-client.ts` to intelligently detect if it's running on `localhost` and automatically point to `http://localhost:7860`. For production, it defaults back to the origin.
2. **Error Unmasking**: I updated `useAuth.ts` and `SignupForm.tsx` to capture and display the **full error object** from Better Auth. This revealed a database connection string issue that was hidden behind the generic message.
3. **Route Mounting**: Updated `auth-server.js` to ensure the handler is mounted at the standard `/api/auth` path.

# OUTCOME

- Resolved Port 3000 to Port 7860 connectivity bridge.
- Improved observability of authentication failures in the UI.
- Successfully performed the first complete signup write to the Neon database.
