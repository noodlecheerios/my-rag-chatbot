import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from tests.test_data.mock_responses import SAMPLE_QUERIES


class TestRAGSystemIntegration:
    """Integration tests for RAG system content query handling"""

    def test_end_to_end_content_query(self, rag_system, tool_manager):
        """Test complete content query flow from input to response"""
        # Mock AI generator to simulate tool calling
        mock_response = "Based on the course materials, MCP architecture provides..."
        rag_system.ai_generator.generate_response.return_value = mock_response
        
        # Mock tool manager to return sources
        mock_sources = [{"text": "MCP Course - Lesson 2", "link": "https://example.com/lesson/2"}]
        tool_manager.get_last_sources.return_value = mock_sources
        
        query = "How does MCP architecture work?"
        session_id = "test_session"
        
        response, sources = rag_system.query(query, session_id)
        
        # Should return AI response and sources
        assert response == mock_response
        assert sources == mock_sources
        
        # Should have called AI generator with correct parameters
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args
        
        # Check that tools were provided
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]
        
        # Should have retrieved and reset sources
        tool_manager.get_last_sources.assert_called_once()
        tool_manager.reset_sources.assert_called_once()

    def test_outline_query_flow(self, rag_system, tool_manager):
        """Test outline query processing"""
        mock_response = "Here's the complete MCP course outline..."
        rag_system.ai_generator.generate_response.return_value = mock_response
        
        mock_sources = [{"text": "MCP: Build Rich-Context AI Apps", "link": "https://example.com/course"}]
        tool_manager.get_last_sources.return_value = mock_sources
        
        query = "Show me the MCP course outline"
        
        response, sources = rag_system.query(query)
        
        assert response == mock_response
        assert sources == mock_sources

    def test_query_prompt_formatting(self, rag_system):
        """Test that query is properly formatted for AI"""
        query = "Test query"
        rag_system.query(query)
        
        # Should wrap query in instruction prompt
        call_args = rag_system.ai_generator.generate_response.call_args
        formatted_query = call_args[1]["query"]
        
        assert "Answer this question about course materials:" in formatted_query
        assert query in formatted_query

    def test_session_management_integration(self, rag_system):
        """Test session management in query processing"""
        query = "Test query"
        session_id = "test_session_123"
        
        # Mock session manager
        mock_history = "Previous conversation history"
        rag_system.session_manager.get_conversation_history.return_value = mock_history
        
        mock_response = "Test response"
        rag_system.ai_generator.generate_response.return_value = mock_response
        
        response, sources = rag_system.query(query, session_id)
        
        # Should retrieve conversation history
        rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Should pass history to AI generator
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == mock_history
        
        # Should update conversation history with exchange
        rag_system.session_manager.add_exchange.assert_called_once_with(session_id, query, mock_response)

    def test_session_management_without_session_id(self, rag_system):
        """Test query processing without session ID"""
        query = "Test query"
        
        response, sources = rag_system.query(query)
        
        # Should not attempt session operations
        rag_system.session_manager.get_conversation_history.assert_not_called()
        rag_system.session_manager.add_exchange.assert_not_called()
        
        # Should pass None for history
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] is None

    def test_source_aggregation(self, rag_system, tool_manager):
        """Test that sources are properly aggregated from tools"""
        # Test different source scenarios
        test_cases = [
            # Multiple sources
            [
                {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson/1"},
                {"text": "MCP Course - Lesson 2", "link": "https://example.com/lesson/2"}
            ],
            # Sources without links
            [
                {"text": "Course Title"},
                {"text": "Another Source"}
            ],
            # Empty sources
            []
        ]
        
        for expected_sources in test_cases:
            tool_manager.get_last_sources.return_value = expected_sources
            
            response, sources = rag_system.query("Test query")
            
            assert sources == expected_sources

    def test_tool_definitions_passed_correctly(self, rag_system, tool_manager):
        """Test that tool definitions are properly passed to AI generator"""
        mock_tools = [
            {"name": "search_course_content", "description": "Search content"},
            {"name": "get_course_outline", "description": "Get outline"}
        ]
        tool_manager.get_tool_definitions.return_value = mock_tools
        
        rag_system.query("Test query")
        
        # Should get tool definitions from manager
        tool_manager.get_tool_definitions.assert_called_once()
        
        # Should pass tools to AI generator
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["tools"] == mock_tools
        assert call_args[1]["tool_manager"] == tool_manager

    def test_error_handling_in_ai_generation(self, rag_system):
        """Test error handling when AI generation fails"""
        # Mock AI generator to raise exception
        rag_system.ai_generator.generate_response.side_effect = Exception("API Error")
        
        # Should propagate the exception (or handle gracefully depending on implementation)
        with pytest.raises(Exception, match="API Error"):
            rag_system.query("Test query")

    def test_error_handling_in_tool_execution(self, rag_system, tool_manager):
        """Test behavior when tools return errors"""
        # Mock AI response
        rag_system.ai_generator.generate_response.return_value = "I couldn't find that information."
        
        # Mock tool manager to return error sources
        tool_manager.get_last_sources.return_value = []
        
        query = "Find nonexistent information"
        response, sources = rag_system.query(query)
        
        # Should still return response and empty sources
        assert response == "I couldn't find that information."
        assert sources == []

    def test_sources_reset_after_query(self, rag_system, tool_manager):
        """Test that sources are properly reset after each query"""
        mock_sources = [{"text": "Test source"}]
        tool_manager.get_last_sources.return_value = mock_sources
        
        # First query
        rag_system.query("First query")
        
        # Should reset sources
        tool_manager.reset_sources.assert_called()
        
        # Reset the mock for second query
        tool_manager.reset_sources.reset_mock()
        tool_manager.get_last_sources.return_value = []
        
        # Second query
        rag_system.query("Second query")
        
        # Should reset sources again
        tool_manager.reset_sources.assert_called()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_rag_system_initialization(self, mock_session_manager, mock_ai_generator, 
                                     mock_vector_store, mock_doc_processor, mock_config):
        """Test RAG system proper initialization"""
        rag = RAGSystem(mock_config)
        
        # Should initialize all components
        mock_doc_processor.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS)
        mock_ai_generator.assert_called_once_with(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        mock_session_manager.assert_called_once_with(mock_config.MAX_HISTORY)
        
        # Should register tools
        assert hasattr(rag, 'tool_manager')
        assert hasattr(rag, 'search_tool')
        assert hasattr(rag, 'outline_tool')

    def test_multiple_queries_in_session(self, rag_system):
        """Test multiple queries in the same session"""
        session_id = "multi_query_session"
        
        # Mock session manager to track conversation
        conversation_history = []
        def mock_get_history(sid):
            return "\n".join(conversation_history) if conversation_history else None
        
        def mock_add_exchange(sid, query, response):
            conversation_history.append(f"User: {query}")
            conversation_history.append(f"Assistant: {response}")
        
        rag_system.session_manager.get_conversation_history.side_effect = mock_get_history
        rag_system.session_manager.add_exchange.side_effect = mock_add_exchange
        
        # First query
        rag_system.ai_generator.generate_response.return_value = "First response"
        response1, _ = rag_system.query("First query", session_id)
        
        # Second query should include history
        rag_system.ai_generator.generate_response.return_value = "Second response"
        response2, _ = rag_system.query("Second query", session_id)
        
        # Check that history was passed on second query
        second_call = rag_system.ai_generator.generate_response.call_args_list[1]
        assert "First query" in str(second_call[1]["conversation_history"])
        assert "First response" in str(second_call[1]["conversation_history"])

    @pytest.mark.parametrize("query,expected_in_prompt", [
        ("How to implement MCP?", "How to implement MCP?"),
        ("Show course outline", "Show course outline"),
        ("", ""),  # Edge case
        ("What is AI?", "What is AI?")
    ])
    def test_query_content_preservation(self, rag_system, query, expected_in_prompt):
        """Test that query content is preserved in prompt formatting"""
        rag_system.query(query)
        
        call_args = rag_system.ai_generator.generate_response.call_args
        formatted_query = call_args[1]["query"]
        
        assert expected_in_prompt in formatted_query

    def test_tool_manager_lifecycle(self, rag_system, tool_manager):
        """Test tool manager operations during query lifecycle"""
        mock_sources = [{"text": "test"}]
        tool_manager.get_last_sources.return_value = mock_sources
        
        query = "Test query"
        response, sources = rag_system.query(query)
        
        # Should follow the complete lifecycle
        tool_manager.get_tool_definitions.assert_called_once()  # Get tools for AI
        tool_manager.get_last_sources.assert_called_once()     # Get sources after AI
        tool_manager.reset_sources.assert_called_once()       # Reset for next query

    def test_concurrent_query_handling(self, rag_system):
        """Test behavior with concurrent queries (if applicable)"""
        # This test checks that the system doesn't have shared state issues
        session1 = "session_1"
        session2 = "session_2"
        
        # Mock different responses for different sessions
        responses = ["Response 1", "Response 2"]
        rag_system.ai_generator.generate_response.side_effect = responses
        
        # Simulate concurrent queries
        response1, _ = rag_system.query("Query 1", session1)
        response2, _ = rag_system.query("Query 2", session2)
        
        assert response1 == "Response 1"
        assert response2 == "Response 2"
        
        # Should have managed sessions separately
        assert rag_system.session_manager.add_exchange.call_count == 2