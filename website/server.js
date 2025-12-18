const express = require('express');
const path = require('path');
const { createRequestHandler } = require('@docusaurus/core/lib');
const { auth } = require('./build/server/server');
const { toNodeHandler } = require('better-auth/node');

// Create Express app
const app = express();

// Better Auth API handler
const authHandler = toNodeHandler(auth.handler);

// Handle Better Auth API routes
app.use('/api/auth', (req, res) => {
  // Pass the request to Better Auth handler
  authHandler(req, res);
});

// Serve static files from build directory
app.use(express.static(path.join(__dirname, 'build')));

// Handle all other routes with Docusaurus
app.use((req, res, next) => {
  createRequestHandler({
    staticDir: path.join(__dirname, 'build'),
    req,
    res,
  })(req, res, next);
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

module.exports = app;