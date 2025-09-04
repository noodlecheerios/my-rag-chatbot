# RAG Chatbot Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(HTML/JS)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Tools as Search Tools<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    
    User->>Frontend: Enter query & click send
    Note over Frontend: script.js:45 sendMessage()
    
    Frontend->>Frontend: Disable input & show loading
    Note over Frontend: script.js:50-60
    
    Frontend->>API: POST /api/query
    Note over Frontend,API: {query, session_id}
    Note over API: app.py:56 endpoint
    
    API->>RAG: rag_system.query(query, session_id)
    Note over API: app.py:66
    
    alt No session_id provided
        API->>Session: create_session()
        Session-->>API: new session_id
    end
    
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: conversation history
    Note over RAG: rag_system.py:117-120
    
    RAG->>AI: generate_response(query, history, tools)
    Note over RAG: rag_system.py:122-127
    
    AI->>AI: Call Anthropic Claude API
    Note over AI: ai_generator.py:80
    
    alt Claude decides to search
        AI->>Tools: execute_tool("search_course_content", params)
        Note over AI: ai_generator.py:84
        
        Tools->>Vector: search(query, course_name, lesson_number)
        Note over Tools: search_tools.py:65-70
        
        Vector-->>Tools: SearchResults with documents & metadata
        
        Tools->>Tools: Format results & track sources
        Note over Tools: search_tools.py:88-114
        
        Tools-->>AI: Formatted search results
        
        AI->>AI: Generate final response with search context
        Note over AI: ai_generator.py:134-135
    end
    
    AI-->>RAG: Generated response
    
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Sources list
    Note over RAG: rag_system.py:130
    
    RAG->>Tools: reset_sources()
    Note over RAG: rag_system.py:133
    
    RAG->>Session: add_exchange(session_id, query, response)
    Note over RAG: rag_system.py:136-137
    
    RAG-->>API: (response, sources)
    Note over RAG: rag_system.py:140
    
    API-->>Frontend: QueryResponse JSON
    Note over API: app.py:68-72<br/>{answer, sources, session_id}
    
    Frontend->>Frontend: Remove loading & render response
    Note over Frontend: script.js:84-85
    
    Frontend->>Frontend: Display message with markdown & sources
    Note over Frontend: script.js:113-138 addMessage()
    
    Frontend->>User: Show AI response with sources
```

## Key File Locations

### Frontend Components
- **HTML Interface**: `frontend/index.html:62` (input field)
- **JavaScript Handler**: `frontend/script.js:45` (sendMessage function)
- **API Communication**: `frontend/script.js:63-72` (POST request)
- **Response Rendering**: `frontend/script.js:113-138` (addMessage function)

### Backend Components
- **API Endpoint**: `backend/app.py:56` (/api/query route)
- **RAG Orchestrator**: `backend/rag_system.py:102` (main query method)
- **AI Processing**: `backend/ai_generator.py:43` (generate_response)
- **Tool Execution**: `backend/search_tools.py:52` (CourseSearchTool.execute)
- **Vector Search**: `backend/vector_store.py` (semantic search)
- **Session Management**: `backend/session_manager.py` (conversation history)

## Data Flow Summary

1. **User Input** → Frontend captures and validates
2. **HTTP Request** → POST to `/api/query` with query and session
3. **Session Handling** → Create or retrieve conversation context
4. **AI Processing** → Claude decides whether to search course content
5. **Tool Execution** → Semantic search through vector store if needed
6. **Response Generation** → AI synthesizes answer with search results
7. **Source Tracking** → Collect and format source references
8. **Session Update** → Store conversation for context
9. **API Response** → Return structured JSON with answer and sources
10. **Frontend Rendering** → Display formatted response with markdown and sources