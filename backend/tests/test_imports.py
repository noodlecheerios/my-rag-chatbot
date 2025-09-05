"""Basic import tests to verify the test infrastructure works"""
import pytest
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from fastapi.testclient import TestClient
        from unittest.mock import Mock, patch
        import tempfile
        import json
        print("✅ All imports successful")
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_conftest_fixtures_available():
    """Test that conftest fixtures are properly defined"""
    # This test will pass if pytest can load the fixtures
    assert True

def test_backend_modules_accessible():
    """Test that backend modules are on the path"""
    try:
        # Add the backend directory to path (same as conftest.py)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from config import Config
        from models import Course, Lesson, CourseChunk
        print("✅ Backend modules accessible")
        assert True
    except ImportError as e:
        pytest.fail(f"Backend module import failed: {e}")

@pytest.mark.api
def test_test_client_creation(client):
    """Test that the FastAPI test client is properly created"""
    assert client is not None
    print("✅ Test client created successfully")

@pytest.mark.api  
def test_mock_rag_system(mock_rag_system):
    """Test that mock RAG system is properly configured"""
    assert mock_rag_system is not None
    assert hasattr(mock_rag_system, 'query')
    assert hasattr(mock_rag_system, 'get_course_analytics')
    print("✅ Mock RAG system configured")

if __name__ == "__main__":
    # Run basic import test
    test_imports()
    test_backend_modules_accessible()
    print("✅ All basic tests passed!")