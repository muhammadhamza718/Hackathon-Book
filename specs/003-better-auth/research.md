# Research: Better Auth Integration & Design Decisions

**Feature**: Better Auth Authentication with User Profiling  
**Date**: 2025-12-16  
**Status**: Complete

This document consolidates all research findings and design decisions made during the planning phase.

## Research 1: Better Auth Integration with Docusaurus

### Problem

How to integrate Better Auth authentication library with Docusaurus static site generator that doesn't natively support server-side API routes?

### Decision

Use Docusaurus custom pages with API route handlers

### Rationale

- Docusaurus supports custom pages under `src/pages/` directory
- Can create API routes using file-based routing: `pages/api/auth/[...all].ts`
- Better Auth provides framework-agnostic `handler` method that accepts standard Request/Response
- Session cookies set by API routes persist across static page navigations

### Implementation Strategy

```typescript
// website/src/pages/api/auth/[...all].ts
import { auth } from "@/lib/auth";

export default async function handler(req, res) {
  return auth.handler(req, res);
}
```

### Alternatives Rejected

1. **Separate Express/Fastify server** - Adds deployment complexity, CORS issues, session cookie domain problems
2. **Client-only Firebase/Supabase** - Doesn't meet Better Auth requirement
3. **Migrate to Next.js** - Major rewrite, breaks existing Docusaurus content

### References

- Better Auth docs: https://www.better-auth.com/docs/installation#mount-handler
- Docusaurus custom pages: https://docusaurus.io/docs/creating-pages

---

## Research 2: User Schema Extension Strategy

### Problem

Better Auth provides core user table (id, email, password). How to add custom background questionnaire fields (software experience, hardware familiarity, etc.)?

### Decision

Use Better Auth's `additionalFields` configuration

### Rationale

- Better Auth natively supports schema extension via `additionalFields`
- Type-safe integration with TypeScript
- CLI automatically generates database migrations for additional fields
- Fields accessible in all auth hooks (`useSession`, `signUp`, etc.)

### Schema Design

```typescript
const auth = betterAuth({
  user: {
    additionalFields: {
      softwareExperience: {
        type: "string",
        required: true,
        input: true, // Allow during signup
      },
      aiMlFamiliarity: {
        type: "string",
        required: true,
        input: true,
      },
      hardwareExperience: {
        type: "string",
        required: true,
        input: true,
      },
      learningGoals: {
        type: "string",
        required: true,
        input: true,
      },
      programmingLanguages: {
        type: "string",
        required: false,
        input: true,
      },
    },
  },
});
```

### Alternatives Rejected

1. **Separate `user_profiles` junction table** - Unnecessary complexity, 1:1 relationship better inline
2. **JSON column** - Loses type safety, difficult to query
3. **Post-signup profile wizard** - Spec requires collection during signup

### References

- Better Auth custom fields: https://www.better-auth.com/docs/concepts/database#extending-core-schema

---

## Research 3: Session Management & Security

### Problem

How to securely manage user sessions across static pages without exposing tokens to JavaScript?

### Decision

HTTP-only cookies with Better Auth's built-in session management

### Rationale

- Better Auth defaults to secure HTTP-only cookies
- Prevents XSS attacks (JavaScript cannot access cookies)
- Automatic CSRF protection via built-in token validation
- Works with static hosting (GitHub Pages, Vercel)
- No need for JWT management complexity

### Configuration

```typescript
const auth = betterAuth({
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // Refresh every 24 hours
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5-minute client cache
    },
  },
  advanced: {
    cookiePrefix: "hackathon_",
    crossSubDomainCookies: {
      enabled: false, // Single domain deployment
    },
  },
});
```

### Security Features

- HTTP-only flag prevents JavaScript access
- Secure flag ensures HTTPS-only transmission
- SameSite=Lax prevents CSRF attacks
- Automatic session rotation on privilege changes

### Alternatives Rejected

1. **JWT in localStorage** - Vulnerable to XSS attacks
2. **Server-side sessions in Redis** - Adds infrastructure, Postgres sufficient
3. **OAuth-only (no password)** - Spec requires email/password

### References

- Better Auth session docs: https://www.better-auth.com/docs/concepts/session

---

## Research 4: Database Migration & Schema Management

### Problem

How to create and manage database tables for Better Auth with Neon Serverless Postgres?

### Decision

Use Better Auth CLI for automated schema generation and migrations

### Rationale

- Better Auth provides `npx better-auth migrate` command
- Generates SQL compatible with PostgreSQL 14+
- Handles core tables (user, session, account, verification)
- Automatically includes additional fields from configuration
- Neon supports standard PostgreSQL migrations

### Migration Process

1. Configure auth instance with all settings
2. Run `npx better-auth migrate` to generate migration SQL
3. Review generated SQL for correctness
4. Apply to Neon database via `@neondatabase/serverless`
5. Verify schema with `\d+ user` in Neon SQL editor

### Core Tables Created

```sql
-- User table
CREATE TABLE "user" (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  email_verified BOOLEAN DEFAULT FALSE,
  name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- Additional fields
  software_experience TEXT NOT NULL,
  ai_ml_familiarity TEXT NOT NULL,
  hardware_experience TEXT NOT NULL,
  learning_goals TEXT NOT NULL,
  programming_languages TEXT
);

-- Session table
CREATE TABLE "session" (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  expires_at TIMESTAMP NOT NULL,
  token TEXT UNIQUE NOT NULL,
  ip_address TEXT,
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Account table (for future OAuth)
CREATE TABLE "account" (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  account_id TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  access_token TEXT,
  refresh_token TEXT,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Verification table (email verification)
CREATE TABLE "verification" (
  id TEXT PRIMARY KEY,
  identifier TEXT NOT NULL,
  value TEXT NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Alternatives Rejected

1. **Prisma/Drizzle ORM** - Adds abstraction, Better Auth CLI sufficient
2. **Manual SQL scripts** - Error-prone, CLI is standard practice
3. **TypeORM migrations** - Not needed, Better Auth handles schema

### References

- Better Auth CLI: https://www.better-auth.com/docs/concepts/database#cli
- Neon Postgres: https://neon.tech/docs/introduction

---

## Research 5: Frontend Component Architecture

### Problem

How to build authentication UI components that integrate seamlessly with Docusaurus theme while providing custom functionality?

### Decision

Create custom React pages using Docusaurus Layout component with Better Auth React SDK

### Rationale

- Docusaurus allows custom pages in `src/pages/`
- Can import `@theme/Layout` to maintain consistent theme
- Better Auth provides React hooks (`useSession`, client SDK)
- Full control over form validation and UX
- Can integrate authenticated navigation via Navbar swizzling

### Component Structure

```typescript
// Custom authentication page
import Layout from "@theme/Layout";
import SignupForm from "@site/src/components/auth/SignupForm";

export default function SignupPage() {
  return (
    <Layout
      title="Create Account"
      description="Sign up for personalized learning"
    >
      <SignupForm />
    </Layout>
  );
}

// Auth-aware navigation
import { useSession } from "@site/src/hooks/useAuth";

function NavbarAuthMenu() {
  const { data: session } = useSession();

  if (session?.user) {
    return <UserMenu user={session.user} />;
  }

  return <Link to="/signin">Sign In</Link>;
}
```

### Integration Points

1. **Navbar**: Swizzle `NavbarContent` to add auth menu
2. **Pages**: Custom pages for signup, signin, profile
3. **Protected Routes**: `AuthGuard` wrapper component
4. **Global State**: React Context for auth state management

### Alternatives Rejected

1. **Iframe external auth** - Poor UX, session isolation issues
2. **Redirect to separate auth domain** - Breaks user flow
3. **Modal-based auth** - Complex state management, accessibility issues

### References

- Docusaurus swizzling: https://docusaurus.io/docs/swizzling
- Better Auth React: https://www.better-auth.com/docs/integrations/react

---

## Technology Stack Summary

| Component       | Technology               | Version        | Justification                                    |
| --------------- | ------------------------ | -------------- | ------------------------------------------------ |
| Auth Library    | Better Auth              | 1.0+           | Requirement, production-ready, TypeScript-native |
| Database        | Neon Serverless Postgres | PostgreSQL 14+ | Existing infrastructure, serverless scaling      |
| Frontend        | React + TypeScript       | 19.0, 5.6      | Docusaurus requirement                           |
| Session Storage | HTTP-only Cookies        | N/A            | Security best practice, XSS prevention           |
| API Routes      | Docusaurus Custom Pages  | 3.9.2          | Native integration, no separate server needed    |

## Key Design Principles

1. **Security First**: HTTP-only cookies, password hashing, CSRF protection
2. **User Experience**: Minimal friction signup with integrated questionnaire
3. **Type Safety**: Full TypeScript coverage for auth state
4. **Scalability**: Serverless architecture supports growth
5. **Maintainability**: Leverages Better Auth's abstractions over custom code

## Open Questions (Deferred to Implementation)

1. **Email Provider**: Which service for verification emails (SendGrid vs AWS SES vs Resend)?

   - **Defer to**: P2 implementation after core auth works

2. **Rate Limiting**: Implementation strategy for API routes?

   - **Defer to**: Security hardening phase after MVP

3. **Password Reset Flow**: UI/UX for forgot password?
   - **Defer to**: Out of scope for initial MVP

## Success Validation

All research decisions validated against spec success criteria:

- ✅ SC-001: 3-min signup achievable with integrated questionnaire
- ✅ SC-003: HTTP-only cookies maintain session across navigation
- ✅ SC-007: Better Auth handles password hashing (bcrypt/scrypt)
- ✅ SC-009: <500ms auth response time with serverless architecture
