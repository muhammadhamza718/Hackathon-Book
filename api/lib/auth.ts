import { betterAuth } from "better-auth";
import { Pool } from "@neondatabase/serverless";

// Initialize Neon database connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

/**
 * Get the base URL for Better Auth
 * It first checks for an explicit env var, then Vercel deployment URL,
 * then defaults to localhost for development.
 */
const getBaseURL = () => {
  if (process.env.BETTER_AUTH_URL) {
    // Ensure no trailing slash
    return process.env.BETTER_AUTH_URL.replace(/\/$/, "");
  }
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  return "http://localhost:3000";
};

const baseUrl = getBaseURL();
console.log(`[AUTH-CONFIG] Initializing with baseURL: ${baseUrl}`);

export const auth = betterAuth({
  // Database configuration
  database: pool,

  // Email/password authentication
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },

  // User schema extensions
  user: {
    additionalFields: {
      softwareExperience: {
        type: "string",
        required: true,
        defaultValue: "beginner",
      },
      aiMlFamiliarity: { type: "string", required: true, defaultValue: "none" },
      hardwareExperience: {
        type: "string",
        required: true,
        defaultValue: "none",
      },
      learningGoals: { type: "string", required: true, defaultValue: "hobby" },
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
      maxAge: 60 * 5,
    },
  },

  // Security settings
  advanced: {
    useSecureCookies: true, // Always use secure cookies on Vercel
    crossSubDomainCookies: { enabled: false },
  },

  // Base URL for authentication endpoints
  baseURL: baseUrl,

  // Secret for signing cookies and tokens
  secret: process.env.BETTER_AUTH_SECRET,
});

// Export types for use in TypeScript
export type Session = typeof auth.$Infer.Session.session;
export type User = typeof auth.$Infer.Session.user;
