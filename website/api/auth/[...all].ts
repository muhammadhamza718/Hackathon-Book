/**
 * Better Auth API Route Handler - Node.js ESM Version
 * Located in website/api/auth/[...all].ts
 */
import { auth } from "../lib/auth.js";
import { toNodeHandler } from "better-auth/node";

// Use standard Node.js serverless function (which supports ESM via our nested package.json)
export default toNodeHandler(auth.handler);
