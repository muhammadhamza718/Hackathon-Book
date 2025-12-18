# Data Model: Better Auth Authentication

**Feature**: Better Auth Authentication with User Profiling  
**Date**: 2025-12-16  
**Database**: Neon Serverless Postgres (PostgreSQL 14+)

## Entity Relationship Diagram

```
┌──────────────────────────────────────┐
│            user                      │
├──────────────────────────────────────┤
│ id                  TEXT PK          │
│ email               TEXT UNIQUE NOT NULL │
│ email_verified      BOOLEAN DEFAULT FALSE │
│ name                TEXT             │
│ created_at          TIMESTAMP        │
│ updated_at          TIMESTAMP        │
│ ─── Questionnaire Fields ─────────   │
│ software_experience TEXT NOT NULL    │
│ ai_ml_familiarity   TEXT NOT NULL    │
│ hardware_experience TEXT NOT NULL    │
│ learning_goals      TEXT NOT NULL    │
│ programming_languages TEXT           │
└──────────────────────────────────────┘
          │ 1
          │
          │ N
          ▼
┌──────────────────────────────────────┐
│           session                    │
├──────────────────────────────────────┤
│ id                  TEXT PK          │
│ user_id             TEXT FK → user   │
│ expires_at          TIMESTAMP NOT NULL │
│ token               TEXT UNIQUE NOT NULL │
│ ip_address          TEXT             │
│ user_agent          TEXT             │
│ created_at          TIMESTAMP        │
└──────────────────────────────────────┘

          user
          │ 1
          │
          │ N
          ▼
┌──────────────────────────────────────┐
│          account                     │
├──────────────────────────────────────┤
│ id                  TEXT PK          │
│ user_id             TEXT FK → user   │
│ account_id          TEXT NOT NULL    │
│ provider_id         TEXT NOT NULL    │
│ access_token        TEXT             │
│ refresh_token       TEXT             │
│ expires_at          TIMESTAMP        │
│ created_at          TIMESTAMP        │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│        verification                  │
├──────────────────────────────────────┤
│ id                  TEXT PK          │
│ identifier          TEXT NOT NULL    │
│ value               TEXT NOT NULL    │
│ expires_at          TIMESTAMP NOT NULL │
│ created_at          TIMESTAMP        │
└──────────────────────────────────────┘
```

## Entity Definitions

### User Entity

**Purpose**: Represents an authenticated learner with profile and background information

**Fields**:

| Field                   | Type      | Constraints               | Description                                                             |
| ----------------------- | --------- | ------------------------- | ----------------------------------------------------------------------- |
| `id`                    | TEXT      | PRIMARY KEY               | Unique user identifier (nanoid/uuid)                                    |
| `email`                 | TEXT      | UNIQUE, NOT NULL          | User's email address (authentication credential)                        |
| `email_verified`        | BOOLEAN   | DEFAULT FALSE             | Email verification status                                               |
| `name`                  | TEXT      | NULLABLE                  | User's display name                                                     |
| `created_at`            | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Account creation timestamp                                              |
| `updated_at`            | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last profile update timestamp                                           |
| `software_experience`   | TEXT      | NOT NULL                  | Enum: "beginner", "intermediate", "advanced", "expert"                  |
| `ai_ml_familiarity`     | TEXT      | NOT NULL                  | Enum: "none", "basic", "intermediate", "advanced"                       |
| `hardware_experience`   | TEXT      | NOT NULL                  | Enum: "none", "hobbyist", "professional", "educator"                    |
| `learning_goals`        | TEXT      | NOT NULL                  | Enum: "career_change", "skill_upgrade", "research", "teaching", "hobby" |
| `programming_languages` | TEXT      | NULLABLE                  | Comma-separated list: "Python", "C++", "JavaScript", "other"            |

**Validation Rules**:

- Email must match RFC 5322 email format
- Email must be unique across all users
- Questionnaire enums must match predefined values
- Password stored separately (hashed, not in table schema shown)

**State Transitions**:

```
[New] → email_verified=false → [Verified] email_verified=true
[Active] → [Updated] updated_at changes on profile edits
```

**Indexes**:

```sql
CREATE INDEX idx_user_email ON "user"(email);
CREATE INDEX idx_user_software_exp ON "user"(software_experience); -- For personalization queries
```

---

### Session Entity

**Purpose**: Manages active authentication sessions with expiration tracking

**Fields**:

| Field        | Type      | Constraints                              | Description                            |
| ------------ | --------- | ---------------------------------------- | -------------------------------------- |
| `id`         | TEXT      | PRIMARY KEY                              | Unique session identifier              |
| `user_id`    | TEXT      | FOREIGN KEY → user(id) ON DELETE CASCADE | Associated user                        |
| `expires_at` | TIMESTAMP | NOT NULL                                 | Session expiration time                |
| `token`      | TEXT      | UNIQUE, NOT NULL                         | Session token (HTTP-only cookie value) |
| `ip_address` | TEXT      | NULLABLE                                 | Client IP address (security logging)   |
| `user_agent` | TEXT      | NULLABLE                                 | Client user agent string               |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP                | Session creation time                  |

**Validation Rules**:

- `expires_at` must be in the future when creating session
- Token must be cryptographically secure random string
- Cascade delete when user is deleted

**State Transitions**:

```
[Created] → token issued → [Active]
[Active] → time > expires_at → [Expired] (garbage collected)
[Active] → explicit logout → [Deleted]
```

**Indexes**:

```sql
CREATE INDEX idx_session_user ON "session"(user_id);
CREATE INDEX idx_session_token ON "session"(token);
CREATE INDEX idx_session_expires ON "session"(expires_at); -- For cleanup queries
```

---

### Account Entity

**Purpose**: Links users to OAuth providers (future social login support)

**Fields**:

| Field           | Type      | Constraints                              | Description                                    |
| --------------- | --------- | ---------------------------------------- | ---------------------------------------------- |
| `id`            | TEXT      | PRIMARY KEY                              | Unique account link identifier                 |
| `user_id`       | TEXT      | FOREIGN KEY → user(id) ON DELETE CASCADE | Associated user                                |
| `account_id`    | TEXT      | NOT NULL                                 | Provider-specific user ID                      |
| `provider_id`   | TEXT      | NOT NULL                                 | OAuth provider name ("github", "google", etc.) |
| `access_token`  | TEXT      | NULLABLE                                 | OAuth access token                             |
| `refresh_token` | TEXT      | NULLABLE                                 | OAuth refresh token                            |
| `expires_at`    | TIMESTAMP | NULLABLE                                 | Token expiration time                          |
| `created_at`    | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP                | Link creation time                             |

**Validation Rules**:

- Composite uniqueness: `(account_id, provider_id)`
- Tokens must be stored encrypted

**Status**: FUTURE USE - Not implemented in MVP (email/password only)

---

### Verification Entity

**Purpose**: Manages email verification and password reset tokens

**Fields**:

| Field        | Type      | Constraints               | Description                                   |
| ------------ | --------- | ------------------------- | --------------------------------------------- |
| `id`         | TEXT      | PRIMARY KEY               | Unique verification identifier                |
| `identifier` | TEXT      | NOT NULL                  | Email address or user ID                      |
| `value`      | TEXT      | NOT NULL                  | Verification token (cryptographically secure) |
| `expires_at` | TIMESTAMP | NOT NULL                  | Token expiration time                         |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Token creation time                           |

**Validation Rules**:

- Token expiration typically 24 hours for email verification
- Token should be single-use (deleted after verification)

**State Transitions**:

```
[Created] → email sent → [Pending]
[Pending] → user clicks link before expiry → [Verified] → Delete token
[Pending] → time > expires_at → [Expired] → Garbage collected
```

**Status**: P2 Priority - Email verification deferred after core auth

---

## Schema Migration SQL

```sql
-- Generated by Better Auth CLI: npx better-auth migrate

-- User table with questionnaire fields
CREATE TABLE IF NOT EXISTS "user" (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
  email TEXT UNIQUE NOT NULL,
  email_verified BOOLEAN DEFAULT FALSE,
  name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- Additional fields for user profiling
  software_experience TEXT NOT NULL,
  ai_ml_familiarity TEXT NOT NULL,
  hardware_experience TEXT NOT NULL,
  learning_goals TEXT NOT NULL,
  programming_languages TEXT
);

-- Session table
CREATE TABLE IF NOT EXISTS "session" (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
  user_id TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  expires_at TIMESTAMP NOT NULL,
  token TEXT UNIQUE NOT NULL,
  ip_address TEXT,
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Account table (future OAuth)
CREATE TABLE IF NOT EXISTS "account" (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
  user_id TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  account_id TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  access_token TEXT,
  refresh_token TEXT,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(account_id, provider_id)
);

-- Verification table
CREATE TABLE IF NOT EXISTS "verification" (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
  identifier TEXT NOT NULL,
  value TEXT NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_user_email ON "user"(email);
CREATE INDEX IF NOT EXISTS idx_user_software_exp ON "user"(software_experience);
CREATE INDEX IF NOT EXISTS idx_session_user ON "session"(user_id);
CREATE INDEX IF NOT EXISTS idx_session_token ON "session"(token);
CREATE INDEX IF NOT EXISTS idx_session_expires ON "session"(expires_at);
CREATE INDEX IF NOT EXISTS idx_account_user ON "account"(user_id);
CREATE INDEX IF NOT EXISTS idx_verification_identifier ON "verification"(identifier);

-- Cleanup triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_updated_at
  BEFORE UPDATE ON "user"
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
```

## Data Access Patterns

### Common Queries

1. **Signup**: Insert new user with questionnaire responses

```sql
INSERT INTO "user" (email, software_experience, ai_ml_familiarity, hardware_experience, learning_goals, programming_languages)
VALUES ($1, $2, $3, $4, $5, $6)
RETURNING *;
```

2. **Signin**: Verify user existed and get profile

```sql
SELECT * FROM "user" WHERE email = $1;
```

3. **Get Session**: Validate active session

```sql
SELECT s.*, u.*
FROM "session" s
JOIN "user" u ON s.user_id = u.id
WHERE s.token = $1 AND s.expires_at > CURRENT_TIMESTAMP;
```

4. **Update Profile**: Modify questionnaire responses

```sql
UPDATE "user"
SET software_experience = $1, learning_goals = $2
WHERE id = $3;
```

5. **Session Cleanup**: Delete expired sessions

```sql
DELETE FROM "session" WHERE expires_at < CURRENT_TIMESTAMP;
```

## Connection Pooling

**Strategy**: Use `@neondatabase/serverless` with connection pooling

```typescript
import { neon, neonConfig } from "@neondatabase/serverless";

// Enable connection pooling (recommended for serverless)
neonConfig.fetchConnectionCache = true;

const sql = neon(process.env.DATABASE_URL!);

// Example query
const users = await sql`SELECT * FROM "user" WHERE email = ${email}`;
```

**Rationale**: Neon Serverless handles connection pooling automatically, reducing latency and database load.

## Privacy & Data Retention

- **PII Fields**: `email`, `name`, questionnaire responses
- **Encryption**: Passwords hashed via Better Auth (bcrypt/scrypt)
- **Retention**: User data persists until account deletion (future feature)
- **Logging**: Session IP/user agent for security auditing only

## Scalability Considerations

- **User Growth**: Index on email supports fast lookups up to millions of users
- **Session Volume**: Automatic expiration and cleanup reduces table size
- **Query Performance**: Covering indexes on common filter fields
- **Connection Limits**: Neon Serverless auto-scales connections

---

**Next Steps**: Apply migrations via Better Auth CLI and verify schema in Neon console
