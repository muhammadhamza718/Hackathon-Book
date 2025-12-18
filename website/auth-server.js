const express = require("express");
const cors = require("cors");
require("dotenv").config();
// Load the auth configuration from the built version or use tsx to run it
// For this implementation, we'll import the compiled version
const path = require("path");
const { createRequire } = require("module");
const requireFromCwd = createRequire(process.cwd());

// Since we're using tsx, we can import the TypeScript module directly
const authModule = require("./src/lib/auth");
const { auth } = authModule;

// Debug: Check if auth is properly loaded
console.log("Auth module loaded:", !!auth);
console.log("Auth object type:", typeof auth);
console.log("Auth handler type:", typeof auth?.handler);

// Import the toNodeHandler from better-auth/node
const { toNodeHandler } = require("better-auth/node");

const app = express();
const port = process.env.API_PORT || 7860;

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
