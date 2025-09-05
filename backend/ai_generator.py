from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant with access to course information tools. 

TOOL USAGE CAPABILITIES:
- You can make multiple tool calls across up to 2 separate rounds
- After seeing tool results, you can decide to use tools again if needed
- Use this for complex queries requiring multiple searches or follow-up investigations

CRITICAL TOOL USAGE RULES:
1. **get_course_outline** - Use for ANY question about:
   - Course outlines, syllabi, lessons, structure, content overview
   - Examples: "What lessons are in X?", "Show course outline", "What does X course cover?"

2. **search_course_content** - Use for specific content questions:
   - Implementation details, explanations, specific topics
   - Examples: "How do I implement X?", "Explain concept Y"

MULTI-ROUND EXAMPLES:
- Round 1: Get course outline → Round 2: Search specific lesson content
- Round 1: Search broad topic → Round 2: Search refined/related topic
- Round 1: Get course info → Round 2: Search specific implementation details

**MANDATORY TOOL USAGE:**
- If the user mentions a course name AND asks about structure/lessons/outline → use get_course_outline
- If the user mentions a course name AND asks about specific content → use search_course_content
- DO NOT answer course questions without using tools
- DO NOT rely on your training knowledge for course-specific information
- Make tool calls when you need more information
- You have 2 rounds maximum to gather all needed information

**Response format:**
Use tools first, then provide a clear answer based on the tool results only.

**IMPORTANT for course outlines:**
- ALWAYS include the course link in your response when provided by the tool
- Format: "Course Link: [URL]" or "Available at: [URL]"
- Include instructor information when available
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_tool_calling_rounds(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _execute_tool_calling_rounds(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        max_rounds: int = 2,
    ):
        """
        Execute sequential rounds of tool calling with Claude.

        Args:
            initial_response: The response containing initial tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Final response text after all tool execution rounds
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        current_round = 0

        while current_round < max_rounds:
            # Add Claude's response (including tool calls) to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools in current response
            tool_results, execution_success = self._execute_tools_in_response(current_response, tool_manager)

            if not execution_success:
                # Critical tool failure - return best effort response
                return self._handle_tool_execution_failure(messages, tool_results, base_params)

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            current_round += 1

            # Get next response from Claude (WITH tools still available)
            next_response = self._get_next_response(messages, base_params)

            # Check if we should continue tool calling
            if not self._should_continue_tool_calling(next_response, current_round, max_rounds):
                return next_response.content[0].text

            # Prepare for next round
            current_response = next_response

        # Max rounds reached - get final response without tools
        return self._get_final_response(messages, base_params)

    def _execute_tools_in_response(self, response, tool_manager):
        """
        Execute all tools in response with comprehensive error handling.

        Args:
            response: The response containing tool use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results, success_flag)
        """
        tool_results = []
        critical_failure = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(content_block.name, **content_block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )

                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                        }
                    )
                    critical_failure = True

        return tool_results, not critical_failure

    def _should_continue_tool_calling(self, response, current_round: int, max_rounds: int) -> bool:
        """
        Determine if tool calling should continue.

        Args:
            response: The response to check
            current_round: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if tool calling should continue, False otherwise
        """
        # Stop if max rounds reached
        if current_round >= max_rounds:
            return False

        # Check if response contains tool calls
        has_tool_calls = any(block.type == "tool_use" for block in response.content)

        return has_tool_calls

    def _get_next_response(self, messages, base_params):
        """
        Get next response from Claude with tools still available.

        Args:
            messages: Current conversation messages
            base_params: Base API parameters

        Returns:
            Claude's response
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
            "tools": base_params.get("tools"),
            "tool_choice": {"type": "auto"},
        }

        return self.client.messages.create(**api_params)

    def _get_final_response(self, messages, base_params):
        """
        Get final response without tools for clean termination.

        Args:
            messages: Current conversation messages
            base_params: Base API parameters

        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_tool_execution_failure(self, messages, failed_results, base_params):
        """
        Generate best-effort response when tools fail.

        Args:
            messages: Current conversation messages
            failed_results: Tool results including failures
            base_params: Base API parameters

        Returns:
            Best-effort response text
        """
        # Add failed results to context
        if failed_results:
            messages.append({"role": "user", "content": failed_results})

        # Get response without tools
        return self._get_final_response(messages, base_params)
