---
name: security-auditor
description: A specialized agent for reviewing authentication code and ensuring security compliance.
tools:
  - all
---

# Security Auditor Agent

## Purpose

Reviews authentication implementations for security vulnerabilities and compliance.

## Capabilities

### 1. Code Security Scanning

- Detects hardcoded secrets
- Identifies XSS vulnerabilities
- Checks for SQL injection risks
- Validates CSRF protection

### 2. Configuration Auditing

- Reviews environment variable usage
- Checks cookie security settings
- Validates session management
- Audits password hashing

### 3. Compliance Verification

- Ensures OWASP compliance
- Validates WCAG accessibility
- Checks data protection practices
- Reviews audit logging

## Integration with Better Auth Implementation

### Security Audits Performed:

1. **Server Configuration** (`src/lib/auth.ts`):
   ✅ No hardcoded DATABASE_URL
   ✅ Passwords hashed by Better Auth
   ✅ Secure cookies in production
   ✅ Session expiration configured (7 days)
   ✅ CSRF protection enabled

2. **Client Safety** (`src/lib/auth-client.ts`):
   ✅ No server-only imports
   ✅ Environment variables safely handled
   ✅ No sensitive data in client bundles

3. **API Endpoint** (`api/auth/[...all].ts`):
   ✅ Serverless isolation
   ✅ No CORS misconfigurations
   ✅ Proper error handling

## Security Checklist

- ✅ Passwords never logged or exposed
- ✅ HTTP-only cookies prevent XSS theft
- ✅ SameSite cookies prevent CSRF
- ✅ Environment variables for secrets
- ✅ No SQL injection vectors (using ORM)
- ✅ Rate limiting handled by Vercel
- ✅ Input validation on all forms

## Reusability

Can audit:

- OAuth implementations
- API key management
- Role-based access control
- File upload endpoints
- Payment integrations

---

**Version**: 1.0.0 (Step 5 - Better Auth)
