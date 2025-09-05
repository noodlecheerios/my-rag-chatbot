import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from fastapi.testclient import TestClient
import tempfile
import shutil

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    return SearchResults(
        documents=[
            "This is sample course content about MCP architecture.",
            "Here's how to implement MCP servers in Python.",
            "MCP clients can connect to multiple servers simultaneously."
        ],
        metadata=[
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 2, "chunk_index": 0},
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 3, "chunk_index": 1}, 
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 4, "chunk_index": 2}
        ],
        distances=[0.1, 0.15, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing no-results scenarios"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock VectorStore for testing"""
    store = Mock(spec=VectorStore)
    store.search.return_value = mock_search_results
    store._resolve_course_name.return_value = "MCP: Build Rich-Context AI Apps"
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course/mcp"
    return store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance for testing"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API"""
    response = Mock()
    response.stop_reason = "end_turn"
    response.content = [Mock()]
    response.content[0].text = "This is a mock AI response about MCP architecture."
    response.content[0].type = "text"
    return response


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock response from Anthropic API with tool use"""
    response = Mock()
    response.stop_reason = "tool_use"
    
    # Mock tool use content
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.input = {"query": "test query", "course_name": "MCP"}
    tool_use_block.id = "tool_123"
    
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "I'll search for that information."
    
    response.content = [text_block, tool_use_block]
    return response


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, tool_manager):
    """Mock RAGSystem for testing"""
    with patch('rag_system.VectorStore') as mock_vector_store_class, \
         patch('rag_system.AIGenerator') as mock_ai_generator_class, \
         patch('rag_system.SessionManager') as mock_session_manager_class, \
         patch('rag_system.DocumentProcessor') as mock_doc_processor_class:
        
        # Mock the dependencies
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_gen = Mock()
        mock_ai_gen.generate_with_tools.return_value = (
            "This is a test response about MCP architecture.", 
            ["source1", "source2"]
        )
        mock_ai_generator_class.return_value = mock_ai_gen
        
        mock_session_mgr = Mock()
        mock_session_mgr.create_session.return_value = "test-session-123"
        mock_session_mgr.get_session_history.return_value = []
        mock_session_manager_class.return_value = mock_session_mgr
        
        mock_doc_processor_class.return_value = Mock()
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = tool_manager
        
        # Mock the query method
        rag.query = Mock(return_value=("Test response", ["source1", "source2"]))
        rag.get_course_analytics = Mock(return_value={
            "total_courses": 2,
            "course_titles": ["MCP: Build Rich-Context AI Apps", "Another Course"]
        })
        
        return rag


@pytest.fixture
def temp_frontend_dir():
    """Create a temporary frontend directory for testing static files"""
    temp_dir = tempfile.mkdtemp()
    frontend_dir = os.path.join(temp_dir, "frontend")
    os.makedirs(frontend_dir)
    
    # Create a simple index.html for testing
    with open(os.path.join(frontend_dir, "index.html"), "w") as f:
        f.write("<html><body><h1>Test Frontend</h1></body></html>")
    
    yield frontend_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_app(mock_rag_system, temp_frontend_dir):
    """Create a FastAPI test app with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Any
    
    # Create test app without static files that don't exist
    app = FastAPI(title="Course Materials RAG System", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Any]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    class ClearSessionRequest(BaseModel):
        session_id: str
    
    class ClearSessionResponse(BaseModel):
        success: bool
        message: str
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or "test-session-123"
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/clear-session", response_model=ClearSessionResponse)
    async def clear_session(request: ClearSessionRequest):
        try:
            # Mock clearing session
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.session_id} cleared successfully"
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    # Mount static files for testing
    app.mount("/", StaticFiles(directory=temp_frontend_dir, html=True), name="static")
    
    return app


@pytest.fixture
def client(test_app):
    """FastAPI test client"""
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Ensure we don't accidentally hit real APIs during testing
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    # Cleanup after test if needed