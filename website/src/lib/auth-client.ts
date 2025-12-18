/**
 * Better Auth Client Configuration
 *
 * Client-side instance for interacting with the Better Auth server.
 * This file is safe to import in frontend components.
 */
import { createAuthClient } from "better-auth/react";
import { inferAdditionalFields } from "better-auth/client/plugins";
import type { auth } from "./auth";

export const authClient = createAuthClient({
  baseURL:
    typeof process !== "undefined" && process.env.REACT_APP_API_BASE_URL
      ? process.env.REACT_APP_API_BASE_URL
      : typeof window !== "undefined"
        ? window.location.origin.includes("localhost")
          ? "http://localhost:7860" // Separate auth server for local development
          : window.location.origin
        : "http://localhost:7860",
  plugins: [inferAdditionalFields<typeof auth>()],
});

export const { signIn, signUp, useSession } = authClient;
