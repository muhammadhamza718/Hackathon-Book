/**
 * Better Auth Instance Configuration
 *
 * This file configures the Better Auth authentication system with:
 * - Neon Serverless Postgres database
 * - User schema extensions for technical background profiling
 * - Session management with HTTP-only cookies
 * - CSRF protection
 */

import "dotenv/config";
import { betterAuth } from "better-auth";
import { Pool } from "pg";

// Initialize Neon database connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl:
    process.env.NODE_ENV === "production"
      ? { rejectUnauthorized: false }
      : false, // Use SSL in production only
});

export const auth = betterAuth({
  // Database configuration
  database: pool,

  // Email/password authentication
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false, // P2 feature - deferred for MVP
  },

  // User schema with additional background questionnaire fields
  user: {
    additionalFields: {
      // Software development experience level
      softwareExperience: {
        type: "string",
        required: true,
        defaultValue: "beginner",
      },
      // AI/ML concepts familiarity
      aiMlFamiliarity: {
        type: "string",
        required: true,
        defaultValue: "none",
      },
      // Hardware/robotics experience
      hardwareExperience: {
        type: "string",
        required: true,
        defaultValue: "none",
      },
      // Primary learning goals
      learningGoals: {
        type: "string",
        required: true,
        defaultValue: "hobby",
      },
      // Preferred programming language familiarity (comma-separated)
      programmingLanguages: {
        type: "string",
        required: false,
        defaultValue: "",
      },
    },
  },

  // Session configuration
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // Update session every 24 hours
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5 minutes client-side cache
    },
  },

  // Security settings
  advanced: {
    useSecureCookies: process.env.NODE_ENV === "production",
    crossSubDomainCookies: {
      enabled: false,
    },
  },

  // Base URL for authentication endpoints
  baseURL: process.env.BETTER_AUTH_URL || "http://localhost:3000",

  // Secret for signing cookies and tokens
  secret: process.env.BETTER_AUTH_SECRET,
});

// Export types for use in TypeScript
export type Session = typeof auth.$Infer.Session.session;
export type User = typeof auth.$Infer.Session.user;
