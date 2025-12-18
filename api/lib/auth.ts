import { betterAuth } from "better-auth";
import { Pool } from "@neondatabase/serverless";

// Initialize Neon database connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

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
    useSecureCookies: process.env.NODE_ENV === "production",
    crossSubDomainCookies: { enabled: false },
  },

  // Base URL for authentication endpoints
  // This must be set to the absolute domain on Vercel
  baseURL: process.env.BETTER_AUTH_URL || "http://localhost:3000",

  // Secret for signing cookies and tokens
  secret: process.env.BETTER_AUTH_SECRET,
});

// Export types for use in TypeScript
export type Session = typeof auth.$Infer.Session.session;
export type User = typeof auth.$Infer.Session.user;
