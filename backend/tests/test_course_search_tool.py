import pytest
from unittest.mock import Mock, patch
from vector_store import SearchResults
from search_tools import CourseSearchTool
from tests.test_data.mock_responses import SAMPLE_COURSE_CONTENT, EXPECTED_SEARCH_RESPONSES


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""

    def test_execute_basic_query(self, course_search_tool, mock_search_results):
        """Test basic query execution without filters"""
        result = course_search_tool.execute("MCP architecture")
        
        # Should return formatted results
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should track sources
        assert len(course_search_tool.last_sources) > 0
        
        # Should call vector store search
        course_search_tool.store.search.assert_called_once_with(
            query="MCP architecture",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, course_search_tool, mock_search_results):
        """Test query execution with course name filter"""
        result = course_search_tool.execute("architecture", course_name="MCP")
        
        assert result is not None
        assert isinstance(result, str)
        
        # Should call search with course filter
        course_search_tool.store.search.assert_called_once_with(
            query="architecture",
            course_name="MCP", 
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, course_search_tool, mock_search_results):
        """Test query execution with lesson number filter"""
        result = course_search_tool.execute("MCP basics", lesson_number=1)
        
        assert result is not None
        
        # Should call search with lesson filter
        course_search_tool.store.search.assert_called_once_with(
            query="MCP basics",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, course_search_tool, mock_search_results):
        """Test query execution with both course and lesson filters"""
        result = course_search_tool.execute("servers", course_name="MCP", lesson_number=2)
        
        assert result is not None
        
        # Should call search with both filters
        course_search_tool.store.search.assert_called_once_with(
            query="servers",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_no_results(self, course_search_tool, empty_search_results):
        """Test handling when no search results are found"""
        course_search_tool.store.search.return_value = empty_search_results
        
        result = course_search_tool.execute("nonexistent topic")
        
        # Should return appropriate message
        assert "No relevant content found" in result
        assert course_search_tool.last_sources == []

    def test_execute_no_results_with_filters(self, course_search_tool, empty_search_results):
        """Test no results message includes filter information"""
        course_search_tool.store.search.return_value = empty_search_results
        
        result = course_search_tool.execute("topic", course_name="MCP", lesson_number=5)
        
        # Should mention the filters in the message
        assert "No relevant content found in course 'MCP' in lesson 5" in result

    def test_execute_error_handling(self, course_search_tool, error_search_results):
        """Test handling of search errors"""
        course_search_tool.store.search.return_value = error_search_results
        
        result = course_search_tool.execute("test query")
        
        # Should return the error message
        assert result == error_search_results.error

    def test_source_tracking(self, course_search_tool, mock_search_results):
        """Test that sources are properly tracked"""
        result = course_search_tool.execute("MCP")
        
        sources = course_search_tool.last_sources
        assert len(sources) == len(mock_search_results.documents)
        
        # Check source structure
        for source in sources:
            assert "text" in source
            # Some sources may have links, some may not
            assert isinstance(source["text"], str)

    def test_result_formatting(self, course_search_tool, mock_search_results):
        """Test that results are properly formatted with headers"""
        result = course_search_tool.execute("MCP")
        
        # Should contain formatted headers
        assert "[MCP: Build Rich-Context AI Apps" in result
        
        # Should contain the actual content
        for doc in mock_search_results.documents:
            assert doc in result

    def test_result_formatting_with_lesson(self, course_search_tool):
        """Test formatting includes lesson information when available"""
        # Mock search results with lesson information
        results_with_lessons = SearchResults(
            documents=["Content about MCP servers"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 3, "chunk_index": 0}],
            distances=[0.1]
        )
        course_search_tool.store.search.return_value = results_with_lessons
        
        result = course_search_tool.execute("servers")
        
        # Should include lesson number in header
        assert "Lesson 3" in result
        assert "[MCP Course - Lesson 3]" in result

    def test_empty_query(self, course_search_tool, empty_search_results):
        """Test handling of empty query string"""
        course_search_tool.store.search.return_value = empty_search_results
        
        result = course_search_tool.execute("")
        
        # Should still call search (let vector store handle empty queries)
        course_search_tool.store.search.assert_called_once()
        assert "No relevant content found" in result

    def test_source_links_included(self, course_search_tool):
        """Test that source links are included when available"""
        # Setup mock to return lesson link
        course_search_tool.store.get_lesson_link.return_value = "https://example.com/lesson/1"
        
        result = course_search_tool.execute("MCP")
        
        sources = course_search_tool.last_sources
        
        # Should have at least one source with a link
        has_link = any("link" in source for source in sources)
        assert has_link

    def test_source_links_fallback(self, course_search_tool):
        """Test source handling when no links are available"""
        # Setup mock to return no links
        course_search_tool.store.get_lesson_link.return_value = None
        course_search_tool.store.get_course_link.return_value = None
        
        result = course_search_tool.execute("MCP")
        
        sources = course_search_tool.last_sources
        
        # Should still have sources, just without links
        assert len(sources) > 0
        for source in sources:
            assert "text" in source
            # Links are optional
            
    def test_multiple_results_formatting(self, course_search_tool, mock_search_results):
        """Test formatting when multiple results are returned"""
        result = course_search_tool.execute("MCP")
        
        # Should separate multiple results with double newlines
        result_parts = result.split("\n\n")
        assert len(result_parts) == len(mock_search_results.documents)

    def test_course_name_resolution(self, course_search_tool, mock_search_results):
        """Test that course name resolution is handled by vector store"""
        course_search_tool.execute("test", course_name="partial name")
        
        # Should pass the course name as provided (resolution happens in vector store)
        course_search_tool.store.search.assert_called_once_with(
            query="test",
            course_name="partial name",
            lesson_number=None
        )

    def test_tool_definition(self, course_search_tool):
        """Test that tool definition is properly formatted"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        # Check required parameters
        required = definition["input_schema"]["required"]
        assert "query" in required
        
        # Check optional parameters exist
        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
        assert "lesson_number" in properties

    def test_sources_reset_between_searches(self, course_search_tool, mock_search_results):
        """Test that sources are properly reset between searches"""
        # First search
        course_search_tool.execute("first query")
        first_sources = course_search_tool.last_sources.copy()
        
        # Setup different results for second search
        different_results = SearchResults(
            documents=["Different content"],
            metadata=[{"course_title": "Different Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.3]
        )
        course_search_tool.store.search.return_value = different_results
        
        # Second search
        course_search_tool.execute("second query")
        second_sources = course_search_tool.last_sources
        
        # Sources should be different and not accumulated
        assert second_sources != first_sources
        assert len(second_sources) == 1  # Only one result in different_results

    @pytest.mark.parametrize("query,course,lesson", [
        ("basic query", None, None),
        ("filtered query", "MCP", None),
        ("lesson query", None, 1),
        ("fully filtered", "Chroma", 2)
    ])
    def test_execute_parameter_combinations(self, course_search_tool, mock_search_results, query, course, lesson):
        """Test various parameter combinations"""
        result = course_search_tool.execute(query, course_name=course, lesson_number=lesson)
        
        assert result is not None
        course_search_tool.store.search.assert_called_once_with(
            query=query,
            course_name=course,
            lesson_number=lesson
        )