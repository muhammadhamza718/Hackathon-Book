---
id: rag-backend-002
title: RAG Chatbot Backend Specification
stage: specify
date: 2025-12-09T15:45:00Z
surface: cli
model: gemini-2.0-flash-exp
feature: rag-chatbot-backend
branch: main
user: hamza
command: /sp.specify
labels: [backend, rag, specification, api-design]
links:
  spec: backend-2/specs/rag-backend.spec.md
  ticket: null
  adr: null
  pr: null
files:
  - path: backend-2/specs/rag-backend.spec.md
    status: created
    lines: 180
  - path: backend-2/.env.example
    status: created
    lines: 8
tests: []
---

## Prompt

Based on the constitution in CLAUDE.md, create a detailed technical specification for the RAG chatbot backend. Include:

1. **API Specification:**

   - Detailed endpoint definitions with request/response schemas
   - Error handling patterns
   - CORS configuration

2. **RAG Pipeline:**

   - Query embedding flow
   - Vector search parameters (top_k, similarity threshold)
   - Context building strategy
   - Prompt engineering templates

3. **Data Models:**

   - Pydantic models for requests/responses
   - Qdrant payload structure
   - Vector dimensions and distance metrics

4. **Environment Configuration:**

   - Required environment variables
   - Default values and validation

5. **Deployment Specification:**
   - Docker configuration for Hugging Face Spaces
   - Port and host settings
   - Health check implementation

## Response snapshot

Created comprehensive specification document covering:

**API Endpoints:**

```
POST /api/ask
  Request: { question: string }
  Response: { answer: string, sources: Array<{chapter: string, score: number}> }

POST /api/ask-selection
  Request: { question: string, selection: string }
  Response: { answer: string }

GET /api/health
  Response: { status: string, service: string, model: string }
```

**RAG Pipeline Details:**

- Embedding: text-embedding-004 (768 dims)
- Search: top_k=3, COSINE distance
- Context: Max 500 chars per chunk
- Prompt: Concise 2-3 sentence answers

**Pydantic Models:**

- AskRequest, AskSelectionRequest
- AskResponse with sources array
- Source model with chapter and score

**Environment Variables:**

```
GEMINI_API_KEY (required)
QDRANT_URL (default: http://localhost:6333)
QDRANT_API_KEY (optional for cloud)
COLLECTION_NAME (required)
```

**Docker Configuration:**

- Base: python:3.11-slim
- Port: 7860
- CMD: uvicorn app.main:app --host 0.0.0.0 --port 7860

## Outcome

- ✅ Impact: Complete technical blueprint ready for implementation
- 🧪 Tests: Test scenarios defined in spec (not yet implemented)
- 📁 Files: Created specs/rag-backend.spec.md (180 lines), .env.example (8 lines)
- 🔁 Next prompts: /sp.plan to create implementation plan
- 🧠 Reflection: Spec provides clear implementation guidance with all technical details

## Evaluation notes (flywheel)

- Failure modes observed: None - specification phase
- Graders run and results (PASS/FAIL): N/A - no code yet
- Prompt variant (if applicable): detailed-spec-v1
- Next experiment (smallest change to try): Consider adding rate limiting specs for production
