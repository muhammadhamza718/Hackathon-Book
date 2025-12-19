/**
 * Better Auth API Route Handler
 * ESM Sandbox version at root with logging
 */
import { auth } from "../lib/auth.js";
import { toNodeHandler } from "better-auth/node";

// Disallow body parsing for auth handler
export const config = { api: { bodyParser: false } };

const authHandler = toNodeHandler(auth.handler);

export default async (req: any, res: any) => {
  // Enhanced Logging for debugging 404s
  const fullUrl = req.url || "unknown";
  console.log(`[AUTH-API] ${req.method} request to ${fullUrl}`);

  // Check if this is a POST request and if it might be missing from logs
  if (req.method === "POST") {
    console.log(`[AUTH-API] POST payload expected for ${fullUrl}`);
  }

  try {
    return await authHandler(req, res);
  } catch (error: any) {
    console.error(`[AUTH-API-ERROR]`, error);
    if (!res.headersSent) {
      res.status(500).json({
        message: "Internal Auth Error",
        error: error.message,
      });
    }
  }
};
