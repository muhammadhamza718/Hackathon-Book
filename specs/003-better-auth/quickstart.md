# Quickstart: Better Auth Authentication Setup

**Feature**: Better Auth Authentication with User Profiling  
**Time to Complete**: ~30 minutes  
**Prerequisites**: Node.js 20+, Neon Postgres database

## Overview

This guide walks you through setting up Better Auth authentication for the Docusaurus-based hackathon textbook project. You'll create user accounts, implement signup/signin flows, and capture technical background information for user personalization.

## Prerequisites

Before starting, ensure you have:

- ✅ Node.js 20.0+ installed (`node --version`)
- ✅ Git repository cloned
- ✅ Neon Serverless Postgres database (from RAG chatbot setup)
- ✅ `.env` file with `DATABASE_URL` configured
- ✅ Basic familiarity with React and TypeScript

## Step 1: Install Dependencies

Navigate to the website directory and install Better Auth:

```bash
cd website
npm install better-auth @neondatabase/serverless
```

**Verify installation**:

```bash
npm list better-auth
# Should show: better-auth@1.x.x
```

## Step 2: Configure Environment Variables

Add Better Auth configuration to `.env`:

```bash
# In website/.env

# Existing database URL (from RAG chatbot)
DATABASE_URL="postgresql://[user]:[password]@[host]/[database]?sslmode=require"

# Add these new variables:
BETTER_AUTH_SECRET="[generate-with-command-below]"
BETTER_AUTH_URL="http://localhost:3000"
```

**Generate secure secret** (run in terminal):

```bash
openssl rand -base64 32
```

Copy the output and paste it as `BETTER_AUTH_SECRET` value.

**Important**: Never commit `.env` to version control. It's already in `.gitignore`.

## Step 3: Create Better Auth Instance

Create the auth configuration file:

**File**: `website/src/lib/auth.ts`

```typescript
import { betterAuth } from "better-auth";
import { neon } from "@neondatabase/serverless";

// Initialize Neon database connection
const sql = neon(process.env.DATABASE_URL!);

export const auth = betterAuth({
  database: {
    provider: "postgres",
    sql: async (query, values) => {
      return await sql(query, values);
    },
  },

  // Email/password authentication
  emailAndPassword: {
    enabled: true,
    minPasswordLength: 8,
    maxPasswordLength: 128,
  },

  // Session configuration
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // Update every 24 hours
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5 minutes
    },
  },

  // User schema with background questionnaire fields
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

  // Security settings
  advanced: {
    cookiePrefix: "hackathon_",
    generateId: () => {
      return crypto.randomUUID();
    },
  },
});
```

## Step 4: Run Database Migrations

Better Auth needs to create tables in your Neon database:

```bash
# Generate migration SQL
npx better-auth migrate

# Follow prompts:
# 1. Select database: Postgres
# 2. Review generated SQL
# 3. Confirm migration
```

**Verify tables created** (in Neon SQL Editor):

```sql
SELECT tablename FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('user', 'session', 'account', 'verification');
```

You should see all 4 tables listed.

## Step 5: Create API Route Handler

Set up the catch-all auth endpoint:

**File**: `website/src/pages/api/auth/[...all].ts`

```typescript
import { auth } from "@/lib/auth";
import type { NextApiRequest, NextApiResponse } from "next";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    return await auth.handler(req);
  } catch (error) {
    console.error("Auth handler error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

// Disable body parsing (Better Auth handles it)
export const config = {
  api: {
    bodyParser: false,
  },
};
```

## Step 6: Create Authentication Hook

Create a custom React hook for auth state:

**File**: `website/src/hooks/useAuth.ts`

```typescript
import { useState, useEffect } from "react";

interface User {
  id: string;
  email: string;
  name?: string;
  softwareExperience: string;
  aiMlFamiliarity: string;
  hardwareExperience: string;
  learningGoals: string;
  programmingLanguages?: string;
}

interface Session {
  user: User | null;
  loading: boolean;
  error: Error | null;
}

export function useAuth(): Session & {
  signup: (data: any) => Promise<void>;
  signin: (email: string, password: string) => Promise<void>;
  signout: () => Promise<void>;
} {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Check session on mount
  useEffect(() => {
    checkSession();
  }, []);

  async function checkSession() {
    try {
      const res = await fetch("/api/auth/session");
      const data = await res.json();
      setUser(data.user || null);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }

  async function signup(userData: any) {
    const res = await fetch("/api/auth/sign-up/email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(userData),
    });

    if (!res.ok) throw new Error("Signup failed");

    const data = await res.json();
    setUser(data.user);
  }

  async function signin(email: string, password: string) {
    const res = await fetch("/api/auth/sign-in/email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) throw new Error("Signin failed");

    const data = await res.json();
    setUser(data.user);
  }

  async function signout() {
    await fetch("/api/auth/sign-out", { method: "POST" });
    setUser(null);
  }

  return { user, loading, error, signup, signin, signout };
}
```

## Step 7: Test Authentication (Quick Verification)

Create a test signup page:

**File**: `website/src/pages/test-auth.tsx`

```typescript
import React, { useState } from "react";
import Layout from "@theme/Layout";
import { useAuth } from "@site/src/hooks/useAuth";

export default function TestAuth() {
  const { user, signup, signin, signout } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignup = async () => {
    try {
      await signup({
        email,
        password,
        name: "Test User",
        softwareExperience: "intermediate",
        aiMlFamiliarity: "basic",
        hardwareExperience: "hobbyist",
        learningGoals: "skill_upgrade",
      });
      alert("Signup successful!");
    } catch (error) {
      alert("Signup failed: " + error.message);
    }
  };

  return (
    <Layout title="Auth Test">
      <div style={{ padding: "2rem" }}>
        <h1>Auth Test Page</h1>

        {user ? (
          <div>
            <p>Logged in as: {user.email}</p>
            <p>Software Experience: {user.softwareExperience}</p>
            <button onClick={signout}>Sign Out</button>
          </div>
        ) : (
          <div>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button onClick={handleSignup}>Sign Up</button>
            <button onClick={() => signin(email, password)}>Sign In</button>
          </div>
        )}
      </div>
    </Layout>
  );
}
```

## Step 8: Run Development Server

Start the development server and test:

```bash
npm run start
```

Visit `http://localhost:3000/test-auth` and try creating an account.

**Expected behavior**:

- ✅ Signup creates user in database
- ✅ Session cookie is set
- ✅ User data displayed after login
- ✅ Logout clears session

## Troubleshooting

### Issue: "Cannot find module '@/lib/auth'"

**Solution**: Update `tsconfig.json` to include path alias:

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"],
      "@site/*": ["./*"]
    }
  }
}
```

### Issue: Database connection error

**Solution**: Verify `DATABASE_URL` format:

```
postgresql://[user]:[password]@[ep-xxx].us-east-2.aws.neon.tech/[dbname]?sslmode=require
```

Enable SSL mode is required for Neon.

### Issue: CORS errors

**Solution**: Ensure API routes and pages are on same domain (localhost:3000). No CORS needed for same-origin.

### Issue: "Session not persisting"

**Solution**: Check cookie settings in browser DevTools > Application > Cookies. Should see `hackathon_session` with HttpOnly flag.

## Next Steps

Now that authentication is working:

1. **Build production UI**: Replace test page with styled components
2. **Add validation**: Implement form validation for signup/signin
3. **Profile management**: Create page for updating background information
4. **Navbar integration**: Swizzle Docusaurus navbar to show user menu
5. **Protected routes**: Add `AuthGuard` wrapper for authenticated pages

---

For full implementation, proceed to `/sp.tasks` to generate detailed task breakdown.

## Useful Commands

```bash
# Check Better Auth version
npm list better-auth

# Generate migration SQL
npx better-auth migrate

# View database schema
psql $DATABASE_URL -c "\d user"

# Clear all sessions (development only)
psql $DATABASE_URL -c "DELETE FROM session;"

# Reset database (DANGER - deletes all users)
psql $DATABASE_URL -c "DROP TABLE user, session, account, verification CASCADE;"
```

## Security Checklist

Before deploying to production:

- [ ] `BETTER_AUTH_SECRET` is cryptographically secure (32+ characters)
- [ ] `.env` file is in `.gitignore`
- [ ] Session cookies have `Secure` flag (HTTPS only)
- [ ] Rate limiting configured for auth endpoints
- [ ] Password validation meets security requirements (min 8 chars)
- [ ] CSRF protection enabled (Better Auth default)
- [ ] Database credentials not exposed in client code

---

**Estimated Time**: 20-30 minutes for setup + testing  
**Difficulty**: Intermediate

For questions or issues, refer to:
<parameter name="Complexity">7
