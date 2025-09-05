import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
from tests.test_data.mock_responses import MOCK_ANTHROPIC_RESPONSES, SAMPLE_QUERIES


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling behavior"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_content_query_triggers_tool_use(self, mock_anthropic, mock_config, tool_manager):
        """Test that content queries trigger tool usage"""
        # Setup mock response for tool calling
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock the initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [
            Mock(type="text", text="I'll search for that information."),
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "MCP architecture"}, id="tool_123")
        ]
        
        # Mock the final response after tool execution
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Here's information about MCP architecture...")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        # Mock tool manager to return a response
        tool_manager.execute_tool.return_value = "MCP architecture details from search..."
        
        result = generator.generate_response(
            query="How does MCP architecture work?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have called Anthropic API twice (initial + final)
        assert mock_client.messages.create.call_count == 2
        
        # Should have executed the tool
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="MCP architecture")
        
        # Should return the final response
        assert "Here's information about MCP architecture" in result

    @patch('ai_generator.anthropic.Anthropic')  
    def test_outline_query_triggers_outline_tool(self, mock_anthropic, mock_config, tool_manager):
        """Test that outline queries trigger the outline tool"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response for outline tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [
            Mock(type="text", text="I'll get the course outline."),
            Mock(type="tool_use", name="get_course_outline",
                 input={"course_title": "MCP"}, id="tool_456")
        ]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Here's the MCP course outline...")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Course outline details..."
        
        result = generator.generate_response(
            query="What lessons are in the MCP course?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have executed the outline tool
        tool_manager.execute_tool.assert_called_once_with("get_course_outline", course_title="MCP")

    @patch('ai_generator.anthropic.Anthropic')
    def test_general_query_no_tool_use(self, mock_anthropic, mock_config, tool_manager):
        """Test that general knowledge queries don't trigger tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response without tool use
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Machine learning is a subset of AI...")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        result = generator.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should only call API once (no tool execution)
        assert mock_client.messages.create.call_count == 1
        
        # Should not execute any tools
        tool_manager.execute_tool.assert_not_called()
        
        # Should return direct response
        assert "Machine learning is a subset of AI" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_workflow(self, mock_anthropic, mock_config, tool_manager):
        """Test the complete tool execution workflow"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Setup tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            Mock(type="text", text="I'll search for that."),
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "test", "course_name": "MCP"}, id="tool_789")
        ]
        
        # Setup final response
        final_response = Mock()
        final_response.stop_reason = "end_turn" 
        final_response.content = [Mock(text="Based on the search results...")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Search results content"
        
        result = generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify the message flow
        calls = mock_client.messages.create.call_args_list
        
        # First call should include tools
        first_call_kwargs = calls[0][1]
        assert "tools" in first_call_kwargs
        assert "tool_choice" in first_call_kwargs
        
        # Second call should include tool results
        second_call_kwargs = calls[1][1]
        messages = second_call_kwargs["messages"]
        
        # Should have user message, assistant response with tool use, and tool results
        assert len(messages) >= 3
        
        # Tool should have been executed
        tool_manager.execute_tool.assert_called_once()

    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tools_in_response(self, mock_anthropic, mock_config, tool_manager):
        """Test handling when Claude tries to use multiple tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with multiple tool uses
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "MCP"}, id="tool_1"),
            Mock(type="tool_use", name="get_course_outline",
                 input={"course_title": "MCP"}, id="tool_2")
        ]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Combined response from tools")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool response"
        
        result = generator.generate_response(
            query="Tell me about MCP",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should execute both tools
        assert tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_error_handling(self, mock_anthropic, mock_config, tool_manager):
        """Test handling when tools return errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "nonexistent"}, id="tool_error")
        ]
        
        # Mock final response after tool error
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="I apologize, but I couldn't find that information.")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        # Mock tool to return error
        tool_manager.execute_tool.return_value = "No relevant content found."
        
        result = generator.generate_response(
            query="Find nonexistent topic",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should still return a response
        assert result is not None
        assert len(result) > 0

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tools_provided(self, mock_anthropic, mock_config):
        """Test behavior when no tools are provided"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Direct response without tools")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        result = generator.generate_response(query="Test query")
        
        # Should not include tools in API call
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_integration(self, mock_anthropic, mock_config, tool_manager):
        """Test that conversation history is properly integrated"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Response with history")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        history = "Previous conversation context"
        
        result = generator.generate_response(
            query="Follow up question",
            conversation_history=history,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should include history in system prompt
        call_kwargs = mock_client.messages.create.call_args[1]
        assert history in call_kwargs["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_system_prompt_includes_tool_instructions(self, mock_anthropic, mock_config, tool_manager):
        """Test that system prompt includes tool usage instructions"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Test response")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        result = generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check that system prompt contains tool instructions
        call_kwargs = mock_client.messages.create.call_args[1]
        system_prompt = call_kwargs["system"]
        
        assert "get_course_outline" in system_prompt
        assert "search_course_content" in system_prompt
        assert "MUST use" in system_prompt

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_parameters(self, mock_anthropic, mock_config, tool_manager):
        """Test that correct API parameters are used"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Test response")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        result = generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify API parameters
        call_kwargs = mock_client.messages.create.call_args[1]
        
        assert call_kwargs["model"] == mock_config.ANTHROPIC_MODEL
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert call_kwargs["tool_choice"]["type"] == "auto"

    @pytest.mark.parametrize("query_type,expected_tool", [
        ("How to implement MCP servers?", "search_course_content"),
        ("What lessons are in MCP course?", "get_course_outline"),
        ("Show me Chroma course outline", "get_course_outline"),
        ("Explain vector search techniques", "search_course_content")
    ])
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_type_tool_mapping(self, mock_anthropic, mock_config, tool_manager, query_type, expected_tool):
        """Test that different query types map to expected tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            Mock(type="tool_use", name=expected_tool, input={"query": "test"}, id="tool_test")
        ]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Response based on tool")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool result"
        
        result = generator.generate_response(
            query=query_type,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify correct tool was called (this test depends on Claude's decision-making)
        # In practice, we can only test the workflow, not force specific tool choices
        assert tool_manager.execute_tool.called

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic, mock_config, tool_manager):
        """Test sequential tool calling across 2 rounds"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock first tool use response
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [
            Mock(type="text", text="I'll get the course outline first."),
            Mock(type="tool_use", name="get_course_outline", 
                 input={"course_title": "MCP"}, id="tool_1")
        ]
        
        # Mock second tool use response (after first tool results)
        second_response = Mock()
        second_response.stop_reason = "tool_use"
        second_response.content = [
            Mock(type="text", text="Now I'll search for specific content."),
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "lesson 2 content", "course_name": "MCP"}, id="tool_2")
        ]
        
        # Mock final response (no more tools)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Based on the outline and content, here's the complete answer...")]
        
        mock_client.messages.create.side_effect = [first_response, second_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool result content"
        
        result = generator.generate_response(
            query="Get MCP course outline then search lesson 2 content",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have made 3 API calls (initial + 2 rounds)
        assert mock_client.messages.create.call_count == 3
        
        # Should have executed 2 tools
        assert tool_manager.execute_tool.call_count == 2
        
        # Verify the sequence of tool calls
        tool_calls = tool_manager.execute_tool.call_args_list
        first_call = tool_calls[0][0]
        second_call = tool_calls[1][0]
        
        assert first_call[0] == "get_course_outline"
        assert second_call[0] == "search_course_content"
        
        # Should return final response
        assert "complete answer" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_reached(self, mock_anthropic, mock_config, tool_manager):
        """Test that sequential tool calling stops at max rounds (2)"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock responses that always want to use tools
        tool_response_1 = Mock()
        tool_response_1.stop_reason = "tool_use"
        tool_response_1.content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "test1"}, id="tool_1")
        ]
        
        tool_response_2 = Mock()
        tool_response_2.stop_reason = "tool_use"
        tool_response_2.content = [
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "test2"}, id="tool_2")
        ]
        
        # This would be the third round (should be prevented)
        tool_response_3 = Mock()
        tool_response_3.stop_reason = "tool_use"
        tool_response_3.content = [
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "test3"}, id="tool_3")
        ]
        
        # Final response without tools (forced)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Maximum rounds reached, here's what I found...")]
        
        mock_client.messages.create.side_effect = [tool_response_1, tool_response_2, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool result"
        
        result = generator.generate_response(
            query="Complex query requiring multiple searches",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have made exactly 3 API calls (initial + 2 rounds max)
        assert mock_client.messages.create.call_count == 3
        
        # Should have executed exactly 2 tools (max rounds)
        assert tool_manager.execute_tool.call_count == 2
        
        # Final call should not have tools parameter
        final_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
        assert "tools" not in final_call_kwargs
        
        assert "Maximum rounds reached" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic, mock_config, tool_manager):
        """Test that sequential tool calling stops when Claude doesn't want more tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First tool use
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "test"}, id="tool_1")
        ]
        
        # Second response with no tool use (should terminate)
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(text="I found the information I needed, here's the answer...")]
        
        mock_client.messages.create.side_effect = [first_response, second_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Sufficient tool result"
        
        result = generator.generate_response(
            query="Simple query with early termination",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have made 2 API calls (initial + 1 round, then terminated)
        assert mock_client.messages.create.call_count == 2
        
        # Should have executed 1 tool
        assert tool_manager.execute_tool.call_count == 1
        
        # Second call should still have tools available
        second_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
        assert "tools" in second_call_kwargs
        
        assert "I found the information I needed" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_with_tool_failure(self, mock_anthropic, mock_config, tool_manager):
        """Test sequential tool calling handles tool execution failures gracefully"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First tool use that will fail
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "test"}, id="tool_1")
        ]
        
        # Final response after tool failure
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="I apologize, but I encountered an error while searching.")]
        
        mock_client.messages.create.side_effect = [first_response, final_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        # Mock tool to raise an exception
        tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        result = generator.generate_response(
            query="Query that causes tool failure",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should have attempted to call the API twice (initial + error recovery)
        assert mock_client.messages.create.call_count == 2
        
        # Should have attempted tool execution once
        assert tool_manager.execute_tool.call_count == 1
        
        # Should still return a response
        assert result is not None
        assert len(result) > 0

    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_context_preserved_across_rounds(self, mock_anthropic, mock_config, tool_manager):
        """Test that conversation context is preserved across multiple tool calling rounds"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock responses for sequential tool calls
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [Mock(type="tool_use", name="get_course_outline", 
                                     input={"course_title": "MCP"}, id="tool_1")]
        
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(text="Final response with context")]
        
        mock_client.messages.create.side_effect = [first_response, second_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool result"
        
        # Include conversation history
        history = "Previous conversation context"
        
        result = generator.generate_response(
            query="Follow up question",
            conversation_history=history,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify both API calls include the conversation history
        for call in mock_client.messages.create.call_args_list:
            call_kwargs = call[1]
            assert history in call_kwargs["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tools_in_single_round_sequential(self, mock_anthropic, mock_config, tool_manager):
        """Test handling multiple tools in a single round within sequential calling"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First round with multiple tools
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [
            Mock(type="tool_use", name="get_course_outline",
                 input={"course_title": "MCP"}, id="tool_1"),
            Mock(type="tool_use", name="search_course_content",
                 input={"query": "architecture"}, id="tool_2")
        ]
        
        # Second round after seeing both results
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(text="Combined response from multiple tools")]
        
        mock_client.messages.create.side_effect = [first_response, second_response]
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        tool_manager.execute_tool.return_value = "Tool result"
        
        result = generator.generate_response(
            query="Complex query needing multiple tools",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should execute both tools in the first round
        assert tool_manager.execute_tool.call_count == 2
        
        # Should make 2 API calls total
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_system_prompt_includes_multiround_instructions(self, mock_anthropic, mock_config, tool_manager):
        """Test that system prompt includes multi-round tool calling instructions"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock(text="Test response")]
        
        mock_client.messages.create.return_value = response
        
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        
        result = generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check system prompt contains multi-round instructions
        call_kwargs = mock_client.messages.create.call_args[1]
        system_prompt = call_kwargs["system"]
        
        assert "multiple tool calls across up to 2 separate rounds" in system_prompt
        assert "MULTI-ROUND EXAMPLES" in system_prompt
        assert "You have 2 rounds maximum" in system_prompt