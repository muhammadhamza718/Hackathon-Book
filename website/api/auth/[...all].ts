/**
 * Better Auth API Route Handler
 * ESM Version for Vercel
 */
import { auth } from "../lib/auth.js";
import { toNodeHandler } from "better-auth/node";

// Disallow body parsing for auth handler
export const config = { api: { bodyParser: false } };

export default toNodeHandler(auth.handler);
