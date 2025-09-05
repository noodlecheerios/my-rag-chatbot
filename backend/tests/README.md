# RAG System Testing Framework

This directory contains comprehensive tests for the RAG (Retrieval-Augmented Generation) system, including unit tests, integration tests, and API endpoint tests.

## Test Structure

```
backend/tests/
├── __init__.py                 # Python package initialization
├── conftest.py                 # Shared pytest fixtures and test configuration
├── test_api_endpoints.py       # API endpoint tests for FastAPI routes
├── test_imports.py             # Basic import and infrastructure tests
└── README.md                   # This file
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Mock external dependencies
- Fast execution

### Integration Tests (`@pytest.mark.integration`) 
- Test component interactions
- May use real dependencies in controlled environments
- Slower execution

### API Tests (`@pytest.mark.api`)
- Test FastAPI endpoint behavior
- Use TestClient for HTTP request/response testing
- Cover error handling, request validation, response format

## Configuration

pytest configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = ["-v", "--tb=short", "--strict-markers", "--disable-warnings"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "api: API endpoint tests"
]
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
uv sync  # Installs pytest, pytest-asyncio, httpx, and other test deps
```

### Run All Tests
```bash
uv run python -m pytest backend/tests/ -v
```

### Run by Category
```bash
# Unit tests only
uv run python -m pytest backend/tests/ -m unit -v

# Integration tests only  
uv run python -m pytest backend/tests/ -m integration -v

# API tests only
uv run python -m pytest backend/tests/ -m api -v
```

### Run Specific Test Files
```bash
# API endpoint tests
uv run python -m pytest backend/tests/test_api_endpoints.py -v

# Import validation tests
uv run python -m pytest backend/tests/test_imports.py -v
```

### Run with Coverage
```bash
uv run python -m pytest backend/tests/ --cov=backend --cov-report=html
```

## Test Fixtures

The `conftest.py` file provides shared fixtures for all tests:

### Core Fixtures
- `mock_config`: Mock Configuration object
- `mock_search_results`: Sample search results for testing
- `empty_search_results`: Empty search results for no-match scenarios
- `mock_vector_store`: Mock VectorStore with pre-configured responses
- `mock_rag_system`: Mock RAGSystem with all dependencies mocked

### API Testing Fixtures
- `temp_frontend_dir`: Temporary directory with test HTML files
- `test_app`: FastAPI application configured for testing (solves static file mounting issue)
- `client`: TestClient instance for making HTTP requests to test endpoints

### Test Environment
- `setup_test_environment`: Auto-used fixture that sets test environment variables

## API Endpoint Tests

The `test_api_endpoints.py` file contains comprehensive tests for all API routes:

### `/api/query` Tests
- Query processing with/without session ID
- Input validation (empty queries, invalid JSON, missing fields)
- Long text handling
- Special character handling  
- Error scenarios

### `/api/courses` Tests
- Course statistics retrieval
- Response format validation
- HTTP method validation
- Query parameter handling

### `/api/clear-session` Tests
- Session clearing functionality
- Input validation
- Error handling

### Static File Tests (/)
- Frontend file serving
- index.html access
- 404 handling for missing files

### Error Handling Tests
- Internal server errors
- CORS header validation
- Content-type handling
- Response structure validation

## Mocking Strategy

The test framework uses extensive mocking to avoid dependencies on:
- Anthropic API (AI generation)
- ChromaDB (vector storage)
- File system operations (document processing)
- Network requests

Key mocking patterns:
- `unittest.mock.Mock` for object mocking
- `unittest.mock.patch` for module-level patching
- Custom fixtures for consistent test data
- Temporary directories for file system tests

## Static File Handling Solution

The original FastAPI app mounts static files from `../frontend`, which doesn't exist during testing. The test framework solves this by:

1. Creating a separate test app (`test_app` fixture) with identical API endpoints
2. Using `temp_frontend_dir` fixture to create temporary HTML files
3. Mounting the temporary directory instead of the missing frontend directory
4. This allows testing both API endpoints and static file serving without import errors

## Adding New Tests

### For New API Endpoints
1. Add the endpoint to the `test_app` fixture in `conftest.py`
2. Create test class in `test_api_endpoints.py` following the pattern:
   ```python
   @pytest.mark.api
   class TestNewEndpoint:
       def test_success_case(self, client):
           response = client.post("/api/new-endpoint", json={...})
           assert response.status_code == 200
           # Add assertions for response content
   ```

### For New Components  
1. Create new test files following naming convention `test_component_name.py`
2. Use appropriate fixtures from `conftest.py`
3. Add new fixtures to `conftest.py` if needed
4. Mark tests with appropriate markers (`@pytest.mark.unit`, etc.)

## Troubleshooting

### Import Errors
- Ensure `sys.path` is correctly configured (handled in `conftest.py`)
- Check that backend modules are accessible
- Run `python tests/test_imports.py` for basic validation

### Fixture Errors
- Verify fixture dependencies in `conftest.py`
- Check that all required mocks are properly configured
- Use `-v` flag for verbose output to debug fixture loading

### API Test Failures
- Check that mock responses match expected format
- Verify TestClient configuration in test_app fixture
- Ensure static file mounting works with temporary directory

### Platform Issues
- Some dependencies (like onnxruntime) may have platform-specific wheels
- Tests should run independently of problematic dependencies through proper mocking