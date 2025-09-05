from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(query=query, course_name=course_name, lesson_number=lesson_number)

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI (now structured with links)

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Create source object with display text and optional link
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
                # Try to get lesson link from vector store
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    # Store structured source with link
                    sources.append({"text": source_text, "link": lesson_link})
                else:
                    # No link available, store as plain text
                    sources.append({"text": source_text})
            else:
                # Course level result, try to get course link
                course_link = self.store.get_course_link(course_title)
                if course_link:
                    sources.append({"text": source_text, "link": course_link})
                else:
                    # No link available
                    sources.append({"text": source_text})

            formatted.append(f"{header}\n{doc}")

        # Store structured sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines with lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last outline request

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline of a course including all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "The title of the course to get the outline for (partial matches work)",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the outline tool with given course title.

        Args:
            course_title: Course title to get outline for

        Returns:
            Formatted course outline or error message
        """
        # First resolve the course name using the existing method
        resolved_title = self.store._resolve_course_name(course_title)

        if not resolved_title:
            return f"No course found matching '{course_title}'"

        # Get course metadata
        try:
            results = self.store.course_catalog.get(ids=[resolved_title])
            if not results or not results.get("metadatas") or not results["metadatas"]:
                return f"Course '{course_title}' found but no metadata available"

            metadata = results["metadatas"][0]
            course_title = metadata.get("title", "Unknown Course")
            course_link = metadata.get("course_link")
            instructor = metadata.get("instructor")

            # Parse lessons from JSON
            import json

            lessons_json = metadata.get("lessons_json", "[]")
            lessons = json.loads(lessons_json)

            # Format the outline
            outline_parts = [f"**{course_title}**"]

            if instructor:
                outline_parts.append(f"*Instructor: {instructor}*")

            if course_link:
                outline_parts.append(f"*Course Link: {course_link}*")

            # Create source object for tracking
            source_obj = {"text": course_title}
            if course_link:
                source_obj["link"] = course_link
            self.last_sources = [source_obj]

            if lessons:
                outline_parts.append(f"\n**Course Outline ({len(lessons)} lessons):**")
                for lesson in lessons:
                    lesson_num = lesson.get("lesson_number", "?")
                    lesson_title = lesson.get("lesson_title", "Untitled")
                    lesson_link = lesson.get("lesson_link")

                    lesson_line = f"{lesson_num}. {lesson_title}"
                    if lesson_link:
                        lesson_line += f" - [Link]({lesson_link})"
                    outline_parts.append(lesson_line)
            else:
                outline_parts.append("\nNo lessons found for this course.")

            return "\n".join(outline_parts)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
