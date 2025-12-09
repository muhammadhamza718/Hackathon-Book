---
id: rag-backend-007
title: RAG Chatbot Backend Integration with Frontend
stage: implement
date: 2025-12-09T17:30:00Z
surface: cli
model: gemini-2.0-flash-exp
feature: rag-chatbot-backend
branch: main
user: hamza
command: /sp.implement
labels: [backend, rag, frontend-integration, chatbot]
links:
  spec: backend-2/specs/rag-backend.spec.md
  ticket: null
  adr: null
  pr: null
files:
  - path: website/src/components/Chatbot/Chatbot.tsx
    status: modified
    lines: 150
  - path: website/.env
    status: modified
    lines: 5
  - path: backend-2/app/main.py
    status: modified
    lines: 99
tests:
  - name: frontend-backend-integration
    status: pass
---

## Prompt

Implement the backend integration with the frontend chatbot component:

1. **Frontend Integration:**

   - Update Chatbot.tsx to connect to the deployed backend
   - Configure API endpoint URL
   - Handle /api/ask requests
   - Handle /api/ask-selection requests
   - Add error handling for backend failures

2. **Environment Configuration:**

   - Add REACT_APP_BACKEND_URL to .env
   - Point to Hugging Face Spaces deployment
   - Configure CORS on backend for frontend domain

3. **API Communication:**

   - Implement fetch calls to backend endpoints
   - Handle loading states
   - Display answers and sources
   - Handle network errors gracefully

4. **Testing:**
   - Test RAG Q&A functionality
   - Test text selection explanation
   - Verify CORS configuration
   - Test error handling

## Response snapshot

Successfully integrated backend with frontend chatbot:

**Frontend Changes (Chatbot.tsx):**

- Updated API endpoint to Hugging Face Spaces URL
- Implemented fetch calls to `/api/ask` endpoint
- Added loading states during API calls
- Display AI responses with sources
- Error handling for network failures
- Timeout handling (30 seconds)

**Environment Configuration:**

```env
REACT_APP_BACKEND_URL=https://muhammadshamza7718-physical-ai-backend.hf.space
```

**Backend CORS Update:**

- Added frontend domain to CORS allowed origins
- Enabled credentials for cross-origin requests
- Allowed all HTTP methods and headers

**API Integration Flow:**

```
User Question → Frontend (Chatbot.tsx)
  → POST /api/ask
  → Backend (RAG Engine)
  → Gemini 2.0 Flash + Qdrant
  → Response with answer + sources
  → Display in Frontend
```

**Features Implemented:**

- ✅ Full RAG Q&A integration
- ✅ Text selection explanation
- ✅ Loading indicators
- ✅ Error messages for failures
- ✅ Source citations display
- ✅ Timeout handling

**Testing Results:**

- ✅ Successfully queries backend
- ✅ Receives and displays answers
- ✅ Shows source citations
- ✅ Handles errors gracefully
- ✅ CORS working correctly

## Outcome

- ✅ Impact: Complete frontend-backend integration enabling AI-powered chatbot
- 🧪 Tests: Manual testing successful, all endpoints working
- 📁 Files: Modified Chatbot.tsx, .env, main.py (CORS config)
- 🔁 Next prompts: /sp.git.commit_pr to commit integration changes
- 🧠 Reflection: Seamless integration with deployed Hugging Face backend

## Evaluation notes (flywheel)

- Failure modes observed: Initial CORS error, fixed by updating allowed origins
- Graders run and results (PASS/FAIL):
  - Frontend-backend communication: PASS
  - CORS configuration: PASS
  - Error handling: PASS
  - User experience: PASS
- Prompt variant (if applicable): frontend-integration-v1
- Next experiment (smallest change to try): Add request caching to reduce API calls
