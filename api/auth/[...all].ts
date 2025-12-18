/**
 * Better Auth API Route Handler
 * ESM Sandbox version at root
 */
import { auth } from "../lib/auth.js";
import { toNodeHandler } from "better-auth/node";

// Disallow body parsing for auth handler
export const config = { api: { bodyParser: false } };

export default toNodeHandler(auth.handler);
