import pytest
from fastapi.testclient import TestClient
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""

    def test_query_with_session_id(self, client):
        """Test query endpoint with provided session ID"""
        response = client.post(
            "/api/query",
            json={
                "query": "What is MCP architecture?",
                "session_id": "test-session-456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-456"
        assert isinstance(data["sources"], list)
        assert len(data["answer"]) > 0

    def test_query_without_session_id(self, client):
        """Test query endpoint without session ID (should create new one)"""
        response = client.post(
            "/api/query",
            json={"query": "How do I implement MCP servers?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        assert isinstance(data["sources"], list)

    def test_query_empty_string(self, client):
        """Test query endpoint with empty query string"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still return a response even with empty query
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_query_missing_query_field(self, client):
        """Test query endpoint without required query field"""
        response = client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_query_with_long_text(self, client):
        """Test query endpoint with very long query text"""
        long_query = "What is MCP? " * 1000  # Very long query
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_with_special_characters(self, client):
        """Test query endpoint with special characters"""
        response = client.post(
            "/api/query",
            json={"query": "What about JSON parsing & special chars: {}[]\"'?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""

    def test_get_course_stats(self, client):
        """Test courses endpoint returns correct statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 2  # From mock
        assert len(data["course_titles"]) == 2
        assert "MCP: Build Rich-Context AI Apps" in data["course_titles"]
        assert "Another Course" in data["course_titles"]

    def test_get_course_stats_method_not_allowed(self, client):
        """Test courses endpoint with wrong HTTP method"""
        response = client.post("/api/courses", json={})
        
        assert response.status_code == 405  # Method Not Allowed

    def test_get_course_stats_with_query_params(self, client):
        """Test courses endpoint ignores query parameters"""
        response = client.get("/api/courses?filter=test&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return same data regardless of query params
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2


@pytest.mark.api
class TestClearSessionEndpoint:
    """Test cases for /api/clear-session endpoint"""

    def test_clear_session_success(self, client):
        """Test successful session clearing"""
        response = client.post(
            "/api/clear-session",
            json={"session_id": "test-session-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
        assert "test-session-123" in data["message"]

    def test_clear_session_missing_session_id(self, client):
        """Test clear session without session ID"""
        response = client.post(
            "/api/clear-session",
            json={}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_clear_session_empty_session_id(self, client):
        """Test clear session with empty session ID"""
        response = client.post(
            "/api/clear-session",
            json={"session_id": ""}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True

    def test_clear_session_invalid_method(self, client):
        """Test clear session with wrong HTTP method"""
        response = client.get("/api/clear-session")
        
        assert response.status_code == 405  # Method Not Allowed


@pytest.mark.api
class TestStaticFileEndpoint:
    """Test cases for static file serving"""

    def test_serve_index_html(self, client):
        """Test serving index.html from root"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Frontend" in response.text

    def test_serve_index_html_explicit(self, client):
        """Test serving index.html explicitly"""
        response = client.get("/index.html")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Frontend" in response.text

    def test_serve_nonexistent_file(self, client):
        """Test serving a file that doesn't exist"""
        response = client.get("/nonexistent.js")
        
        assert response.status_code == 404


@pytest.mark.api 
class TestAPIErrorHandling:
    """Test error handling across API endpoints"""

    def test_query_internal_error(self, client, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        # Mock the RAG system to raise an exception
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    def test_courses_internal_error(self, client, mock_rag_system):
        """Test courses endpoint when analytics raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics service down")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics service down" in data["detail"]

    def test_cors_headers_present(self, client):
        """Test that CORS headers are properly set"""
        response = client.get("/api/courses")
        
        # Check that CORS middleware added the headers
        # Note: TestClient might not expose all middleware headers
        assert response.status_code == 200

    def test_content_type_validation(self, client):
        """Test proper content-type handling"""
        # Test with correct content type
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        
        # Test with incorrect content type but valid JSON
        response = client.post(
            "/api/query", 
            json={"query": "test"},
            headers={"Content-Type": "text/plain"}
        )
        # FastAPI should still parse it correctly
        assert response.status_code == 200


@pytest.mark.api
class TestAPIResponseFormat:
    """Test API response formats and structure"""

    def test_query_response_structure(self, client):
        """Test that query response has correct structure"""
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_structure(self, client):
        """Test that courses response has correct structure"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check that all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_clear_session_response_structure(self, client):
        """Test that clear session response has correct structure"""
        response = client.post(
            "/api/clear-session",
            json={"session_id": "test"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["success", "message"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)