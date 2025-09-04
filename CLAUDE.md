# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
**IMPORTANT: Always use `uv` for all dependency management - never use `pip` directly**

```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add package_name

# Remove dependency
uv remove package_name

# Run Python scripts/modules
uv run python script.py
uv run module_name
```

### Development Commands
```bash
# No test suite, linting, or formatting tools configured
# This project uses direct uv dependency management without additional dev tools
```

### Environment Setup
- Create `.env` file in root with: `ANTHROPIC_API_KEY=your_key_here`
- Application loads documents from `docs/` folder automatically on startup
- ChromaDB storage created in `backend/chroma_db/`
- Requires Python 3.13+ and uv package manager

## Architecture Overview

### Core RAG Pipeline
This is a **tool-calling RAG system** where Claude decides whether to search based on query type:

1. **Query Processing**: User query → FastAPI → RAG System → AI Generator
2. **Tool Decision**: Claude API determines if search is needed via tool calling
3. **Search Execution**: CourseSearchTool → VectorStore → ChromaDB semantic search  
4. **Response Assembly**: AI generates response using search results + conversation history

### Key Component Interactions

**RAG System (`rag_system.py`)**: Main orchestrator that coordinates all components
- Manages document ingestion workflow
- Handles query processing with session context
- Integrates AI generator with tool manager

**AI Generator (`ai_generator.py`)**: Claude API integration with tool calling
- Single search per query limitation enforced via system prompt
- Handles tool execution workflow: initial response → tool calls → final response
- Temperature=0 for consistent responses

**Tool Manager + CourseSearchTool (`search_tools.py`)**: Search abstraction layer
- Abstract Tool interface for extensibility
- CourseSearchTool implements semantic search with course/lesson filtering
- Source tracking for frontend display

**Vector Store (`vector_store.py`)**: ChromaDB wrapper with dual collections
- `course_metadata`: Course titles and structure for semantic matching
- `course_content`: Chunked text content with embeddings
- Unified search interface with smart course name matching

**Document Processor (`document_processor.py`)**: Text chunking and course extraction
- Sentence-based chunking with configurable overlap (800/100 chars default)
- Regex-based course structure parsing (Course titles, lesson numbers)
- Handles multiple document formats (.pdf, .docx, .txt)

### Data Models (`models.py`)
```python
Course -> Lesson[] -> CourseChunk[]
```
- **Course**: Title (unique ID), instructor, lessons, course_link
- **Lesson**: lesson_number, title, lesson_link  
- **CourseChunk**: content, course_title, lesson_number, chunk_index

### Configuration (`config.py`)
Key settings in dataclass format:
- `CHUNK_SIZE=800, CHUNK_OVERLAP=100`: Text chunking parameters
- `MAX_RESULTS=5`: Search result limit
- `MAX_HISTORY=2`: Conversation context limit
- `ANTHROPIC_MODEL="claude-sonnet-4-20250514"`: Fixed model version

### Session Management (`session_manager.py`)
- In-memory conversation history storage
- Automatic session creation via UUID
- Limited history retention (MAX_HISTORY messages)

## Development Notes

### Adding New Tools
Implement the `Tool` abstract base class in `search_tools.py`:
```python
class NewTool(Tool):
    def get_tool_definition(self) -> Dict[str, Any]: # Anthropic tool schema
    def execute(self, **kwargs) -> str: # Tool execution logic
```

### Document Processing
- Course documents auto-processed on startup from `docs/` folder
- Duplicate detection prevents re-processing existing courses
- Course titles serve as unique identifiers

### API Structure
- Frontend serves from `/` (static files)
- API endpoints at `/api/*`
- Two main endpoints: `/api/query` (chat) and `/api/courses` (stats)

### ChromaDB Collections
- Persistent storage in `backend/chroma_db/`
- Sentence transformer embeddings: `all-MiniLM-L6-v2`
- No external vector database required

### Frontend Integration
- Vanilla HTML/JS/CSS in `frontend/`
- Markdown rendering for AI responses
- Collapsible source attribution
- Session-based conversation continuity

## Important Implementation Details

### Query Processing Flow
1. **Frontend** (`frontend/script.js:45`) → POST `/api/query`
2. **API Endpoint** (`backend/app.py:56`) → RAG System
3. **RAG System** (`backend/rag_system.py:102`) → AI Generator with tools
4. **Tool Decision**: Claude API determines search necessity
5. **Search Execution** (`backend/search_tools.py:52`) → Vector Store
6. **Response Assembly**: AI synthesizes with search results + history

### Error Handling Patterns
- Frontend: Loading states, error messages in chat
- Backend: HTTPException with 500 status for all errors
- No structured error types - relies on exception messages

### Security Considerations
- CORS enabled for all origins (`*`) - development setup
- No authentication or rate limiting implemented
- API key loaded from environment variables only