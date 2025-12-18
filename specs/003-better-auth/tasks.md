# Tasks: Better Auth & Personalized Profiling

**Spec Reference**: [spec.md](./spec.md)
**Plan Reference**: [plan.md](./plan.md)

## Phase 1: Architectural Foundation (The Split)

- [x] **T1.1: Directory Sanitization**
  - Delete `website/src/pages/api` to prevent SSG build conflicts.
  - Create root `website/api/` folder for Vercel Serverless Functions.
- [x] **T1.2: Server-Side Auth Configuration**
  - Create `website/src/lib/auth.ts`.
  - Configure `postgres` pool for Neon.
  - Implement `additionalFields` mapping with 5 technical markers.
- [x] **T1.3: Browser-Safe Auth Client**
  - Create `website/src/lib/auth-client.ts`.
  - Ensure zero server-side leaks using `createAuthClient` correctly.
  - Implement dynamic `baseURL` for Port 7860/3000 compatibility.

## Phase 2: Feature Implementation (The Forms)

- [x] **T2.1: Signup Questionnaire Logic**
  - Build `SignupForm.tsx` with all 9 fields (4 standard + 5 technical).
  - Implement background profiling data mapping to `signUp.email()`.
- [x] **T2.2: Professional UI Styling (De-Gamification)**
  - Refactor `SignupForm.module.css`.
  - **Cleanup**: Remove `text-shadow` and `box-shadow` neon glows.
  - **Styling**: Implement clean, Stripe-like borders and soft shadows.
- [x] **T2.3: Verification Feedback**
  - Add explicit "Signup Failed" error unmasking in the UI to facilitate debugging.

## Phase 3: Profile & Personalization

- [x] **T3.1: Profile Page Development**
  - Create `website/src/pages/profile.tsx`.
  - Use `authClient.useSession()` to fetch user background data.
  - Display technical markers in a professional table layout.
- [x] **T3.2: Navigation Integration**
  - Update `NavbarUserMenu` to include a "Profile" link.
  - Ensure Navbar reflects the logged-in state accurately using session hooks.

## Phase 4: Verification & Infrastructure

- [x] **T4.1: Database Migration**
  - Run `npx @better-auth/cli migrate` to sync Neon Postgres with the updated schema.
- [x] **T4.2: Build & Deploy Verification**
  - Run `npm run build` to confirm zero SSG errors.
  - Verify local connectivity between Port 3000 and the Port 7860 auth server.

## Acceptance Tests

| ID    | Test Case            | Expected Result                                          |
| ----- | -------------------- | -------------------------------------------------------- |
| AT-01 | Full Signup Flow     | User created with 5 technical fields in DB.              |
| AT-02 | Build Integrity      | No "Module not found: fs" error during compilation.      |
| AT-03 | Theme Compatibility  | Signup form is perfectly readable in Light Mode.         |
| AT-04 | User Personalization | Profile page shows correct technical proficiency levels. |
