/**
 * Better Auth API Route Handler
 *
 * Vercel Serverless Function to handle authentication requests.
 * This file maps to /api/auth/* and processes all auth-related API calls.
 */

import { auth } from "../../website/src/lib/auth";
import { toNodeHandler } from "better-auth/node";

// Disallow body parsing, we will parse it manually
export const config = { api: { bodyParser: false } };
export default toNodeHandler(auth.handler);
