import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
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
def error_search_results():
    """Search results with error for testing error scenarios"""
    return SearchResults.empty("No course found matching 'nonexistent'")


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock VectorStore for testing"""
    store = Mock(spec=VectorStore)
    store.search.return_value = mock_search_results
    store._resolve_course_name.return_value = "MCP: Build Rich-Context AI Apps"
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course/mcp"
    
    # Mock course catalog for outline tool
    store.course_catalog = Mock()
    store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'MCP: Build Rich-Context AI Apps',
            'instructor': 'Test Instructor',
            'course_link': 'https://example.com/course/mcp',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson/1"}]'
        }]
    }
    
    return store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance for testing"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool instance for testing"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API"""
    response = Mock()
    response.stop_reason = "end_turn"  # or "tool_use" for tool calling scenarios
    response.content = [Mock()]
    response.content[0].text = "This is a mock AI response."
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
def mock_ai_generator(mock_config, mock_anthropic_response):
    """Mock AIGenerator for testing"""
    with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        generator.client = mock_client
        return generator


@pytest.fixture
def rag_system(mock_config, mock_vector_store, tool_manager):
    """RAGSystem instance for testing"""
    with patch('rag_system.VectorStore') as mock_vector_store_class, \
         patch('rag_system.AIGenerator') as mock_ai_generator_class, \
         patch('rag_system.SessionManager') as mock_session_manager_class, \
         patch('rag_system.DocumentProcessor') as mock_doc_processor_class:
        
        # Mock the dependencies
        mock_vector_store_class.return_value = mock_vector_store
        mock_ai_generator_class.return_value = Mock()
        mock_session_manager_class.return_value = Mock()
        mock_doc_processor_class.return_value = Mock()
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = tool_manager
        return rag


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Ensure we don't accidentally hit real APIs during testing
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    # Cleanup after test if needed