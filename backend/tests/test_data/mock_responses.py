"""
Mock data and responses for RAG system testing
"""

# Sample course content for testing
SAMPLE_COURSE_CONTENT = {
    "mcp_course": {
        "title": "MCP: Build Rich-Context AI Apps with Anthropic",
        "instructor": "Elie Schoppik",
        "course_link": "https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/",
        "lessons": [
            {
                "lesson_number": 0,
                "lesson_title": "Introduction",
                "lesson_link": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/fkbhh/introduction"
            },
            {
                "lesson_number": 1,
                "lesson_title": "Why MCP",
                "lesson_link": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp"
            },
            {
                "lesson_number": 2,
                "lesson_title": "MCP Architecture",
                "lesson_link": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/xtt6w/mcp-architecture"
            }
        ],
        "content_chunks": [
            {
                "content": "Model Context Protocol (MCP) is a new standard for connecting AI assistants to data sources.",
                "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "content": "MCP servers provide standardized interfaces for AI assistants to access various data sources.",
                "course_title": "MCP: Build Rich-Context AI Apps with Anthropic", 
                "lesson_number": 2,
                "chunk_index": 1
            },
            {
                "content": "Creating an MCP server involves implementing the MCP protocol handlers for your specific data source.",
                "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 3,
                "chunk_index": 2
            }
        ]
    },
    "chroma_course": {
        "title": "Advanced Retrieval for AI with Chroma",
        "instructor": "Anton Troynikov",
        "course_link": "https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/",
        "lessons": [
            {
                "lesson_number": 0,
                "lesson_title": "Introduction",
                "lesson_link": "https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/intro/introduction"
            },
            {
                "lesson_number": 1,
                "lesson_title": "Overview Of Embeddings Based Retrieval",
                "lesson_link": "https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/overview/overview"
            }
        ],
        "content_chunks": [
            {
                "content": "Chroma is a vector database designed for AI applications and embeddings storage.",
                "course_title": "Advanced Retrieval for AI with Chroma",
                "lesson_number": 1,
                "chunk_index": 0
            }
        ]
    }
}

# Sample queries for testing different scenarios
SAMPLE_QUERIES = {
    "content_queries": [
        "How do I create an MCP server?",
        "What is the MCP architecture?", 
        "How to implement vector search with Chroma?",
        "Explain embedding-based retrieval methods",
        "Show me MCP server code examples"
    ],
    "outline_queries": [
        "What lessons are in the MCP course?",
        "Show me the course outline for Chroma", 
        "What topics does the MCP course cover?",
        "Get the syllabus for Advanced Retrieval course",
        "What's the structure of the MCP course?"
    ],
    "general_queries": [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the benefits of AI?",
        "Explain neural networks",
        "What is deep learning?"
    ]
}

# Expected tool responses
EXPECTED_SEARCH_RESPONSES = {
    "mcp_architecture_query": {
        "formatted_response": """[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 2]
MCP servers provide standardized interfaces for AI assistants to access various data sources.

[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 3]  
Creating an MCP server involves implementing the MCP protocol handlers for your specific data source.""",
        "sources": [
            {"text": "MCP: Build Rich-Context AI Apps with Anthropic - Lesson 2", "link": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/xtt6w/mcp-architecture"},
            {"text": "MCP: Build Rich-Context AI Apps with Anthropic - Lesson 3"}
        ]
    },
    "no_results": {
        "formatted_response": "No relevant content found in course 'Nonexistent Course'.",
        "sources": []
    }
}

EXPECTED_OUTLINE_RESPONSES = {
    "mcp_course_outline": {
        "formatted_response": """**MCP: Build Rich-Context AI Apps with Anthropic**
*Instructor: Elie Schoppik*
*Course Link: https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/*

**Course Outline (3 lessons):**
0. Introduction - [Link](https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/fkbhh/introduction)
1. Why MCP - [Link](https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp)  
2. MCP Architecture - [Link](https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/xtt6w/mcp-architecture)""",
        "sources": [
            {"text": "MCP: Build Rich-Context AI Apps with Anthropic", "link": "https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/"}
        ]
    }
}

# Mock Anthropic API responses
MOCK_ANTHROPIC_RESPONSES = {
    "content_tool_use": {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "I'll search for information about MCP architecture."},
            {
                "type": "tool_use", 
                "name": "search_course_content",
                "input": {"query": "MCP architecture", "course_name": "MCP"},
                "id": "tool_123"
            }
        ]
    },
    "outline_tool_use": {
        "stop_reason": "tool_use", 
        "content": [
            {"type": "text", "text": "I'll get the course outline for you."},
            {
                "type": "tool_use",
                "name": "get_course_outline", 
                "input": {"course_title": "MCP"},
                "id": "tool_456"
            }
        ]
    },
    "no_tool_use": {
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."}
        ]
    },
    "final_response_with_tools": {
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": "Based on the course materials, MCP (Model Context Protocol) architecture provides standardized interfaces for AI assistants to connect to various data sources. The architecture involves MCP servers that implement protocol handlers for specific data sources, allowing AI assistants to access and utilize external information seamlessly."}
        ]
    }
}

# Error scenarios for testing
ERROR_SCENARIOS = {
    "vector_store_error": "ChromaDB connection failed",
    "course_not_found": "No course found matching 'NonexistentCourse'",
    "api_error": "Anthropic API rate limit exceeded",
    "malformed_tool_response": "Invalid tool response format"
}