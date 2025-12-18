const express = require("express");
const cors = require("cors");
require("dotenv").config();
// Load the auth configuration from the built version or use tsx to run it
// For this implementation, we'll import the compiled version
const path = require("path");
const { createRequire } = require("module");
const requireFromCwd = createRequire(process.cwd());

// Use dynamic import for the ESM auth configuration
// This works with tsx and allows loading ESM from this CJS file
let auth;
let authHandler;

const initAuth = async () => {
  const authModule = await import("./api/lib/auth.ts");
  auth = authModule.auth;
  console.log("Auth module loaded:", !!auth);

  const { toNodeHandler } = require("better-auth/node");
  authHandler = toNodeHandler(auth.handler);
};

const app = express();
const port = process.env.API_PORT || 7860;

// Middleware to parse authHandler after initialization
app.use(async (req, res, next) => {
  if (!authHandler) {
    await initAuth();
  }
  next();
});

// Middleware to parse JSON - apply at app level
app.use(express.json());

// Enable CORS for local development
app.use(
  cors({
    origin: "http://localhost:3000", // Docusaurus dev server
    credentials: true,
  })
);

// Better Auth API handler
const authHandler = toNodeHandler(auth.handler);

// Handle specific auth routes first (before the general middleware)
app.get("/auth/test", (req, res) => {
  console.log("Test route hit");
  res.json({ message: "Test route working" });
});

// Handle all other auth routes under /api/auth using Better Auth
// This should come AFTER specific routes to avoid intercepting them
app.use("/api/auth", async (req, res, next) => {
  console.log(`Received ${req.method} request to ${req.originalUrl}`);
  console.log(`Processed URL: ${req.url}`);

  try {
    await authHandler(req, res);
  } catch (error) {
    console.error("Error in auth handler:", error);
    // Send error response if not already sent
    if (!res.headersSent) {
      res.status(500).json({ error: error.message || "Internal server error" });
    }
  }
});

app.get("/", (req, res) => {
  res.json({ message: "Better Auth API Server running" });
});

app.listen(port, () => {
  console.log(`Better Auth API server running at http://localhost:${port}`);
  console.log(`API endpoints available at http://localhost:${port}/api/auth/*`);
});

module.exports = app;
