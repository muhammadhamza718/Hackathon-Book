---
id: rag-backend-001
title: RAG Chatbot Backend Constitution
stage: constitution
date: 2025-12-09T15:30:00Z
surface: cli
model: gemini-2.0-flash-exp
feature: rag-chatbot-backend
branch: main
user: hamza
command: /sp.constitution
labels: [backend, rag, architecture, gemini, qdrant]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - path: backend-2/CLAUDE.md
    status: created
    lines: 250
tests: []
---

## Prompt

Create a constitution document for a RAG (Retrieval-Augmented Generation) chatbot backend for the Physical AI Textbook. The backend should:

1. Use Google Gemini 2.0 Flash for AI reasoning and generation
2. Use Qdrant for vector storage and retrieval
3. Serve via FastAPI with CORS support
4. Support two main endpoints:
   - `/api/ask` - Full RAG Q&A on textbook content
   - `/api/ask-selection` - Explain selected text without RAG
5. Be deployable to Hugging Face Spaces using Docker
6. Have zero bloat - no authentication, user management, or unnecessary features
7. Use text-embedding-004 for embeddings (768 dimensions)

The constitution should define the core principles, architecture decisions, and technical constraints for this backend service.

## Response snapshot

Created `CLAUDE.md` constitution document with the following key sections:

**Core Principles:**

- Zero Bloat Philosophy: Pure knowledge serving API
- Gemini-First: State-of-the-art reasoning with Gemini 2.0 Flash
- RAG Excellence: Efficient retrieval with Qdrant
- Deployment Simplicity: Docker-based Hugging Face Spaces deployment

**Architecture Decisions:**

- FastAPI for async performance
- OpenAI SDK compatibility layer for Gemini API
- Qdrant Cloud for managed vector storage
- Environment-based configuration
- CORS enabled for frontend integration

**Technical Stack:**

- Python 3.11+
- FastAPI + Uvicorn
- Google Gemini 2.0 Flash (via OpenAI SDK)
- Qdrant Client
- text-embedding-004 embeddings

**API Design:**

- POST /api/ask - RAG-powered Q&A
- POST /api/ask-selection - Direct text explanation
- GET /api/health - Health check endpoint

**Constraints:**

- Max 3 retrieved chunks per query
- 2-3 sentence concise answers
- 500 char limit on selections
- No authentication required
- Public API access

## Outcome

- ✅ Impact: Established clear architectural foundation and technical direction for RAG backend
- 🧪 Tests: No tests at constitution stage
- 📁 Files: Created CLAUDE.md (250 lines)
- 🔁 Next prompts: /sp.specify to create detailed technical specification
- 🧠 Reflection: Constitution provides strong foundation with clear constraints and zero-bloat philosophy

## Evaluation notes (flywheel)

- Failure modes observed: None - constitution phase is purely planning
- Graders run and results (PASS/FAIL): N/A - no code to test
- Prompt variant (if applicable): standard-constitution-v1
- Next experiment (smallest change to try): Add more specific Qdrant configuration details in spec phase
