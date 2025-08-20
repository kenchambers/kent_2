# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a self-improving AI agent built with LangGraph and Google's Gemini 2.5 Pro, featuring advanced memory architecture and emotional intelligence. The agent dynamically creates memory layers based on conversations and maintains long-term episodic memory through FAISS vector stores.

## Development Commands

### Running the Agent
```bash
uv run kent
```

### Package Management
```bash
# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Add new dependencies
uv add package_name

# Development dependencies
uv add --dev package_name
```

### Code Quality
```bash
# Lint with ruff
uv run ruff check .
uv run ruff check --fix .

# Format with ruff
uv run ruff format .
```

### Environment Setup
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Architecture Overview

### Core Components

- **Agent (`src/self_improving_agent/agent.py`)**: Main SelfImprovingAgent class implementing the LangGraph workflow
- **Memory (`src/self_improving_agent/memory.py`)**: FAISS vector store management for long-term memory
- **Config (`src/self_improving_agent/config.py`)**: Configuration management and file I/O operations
- **Main (`src/self_improving_agent/main.py`)**: CLI entry point with async event loop

### Memory Architecture

The agent uses a sophisticated multi-layered memory system:

1. **Short-term Summary**: Dynamic conversation summary to maintain context efficiency
2. **Long-term Memory**: FAISS vector stores for semantic search across conversation history
3. **Topic-specific Layers**: Automatically created specialized memory layers (stored in `vector_stores/`)
4. **Core Identity Layers**: Identity concepts stored as searchable memory layers

### Key Files

- `agent_config.json`: Dynamic configuration tracking memory layers and versions
- `core_identity.json`: Agent's core identity with hybrid static/dynamic references
- `conversation_history.json`: Raw conversation history
- `vector_stores/`: Directory containing FAISS indices for each memory layer

### Agent State Management

The `AgentState` TypedDict includes:
- `conversation_history`: Message history
- `short_term_summary`: Condensed recent context
- `episodic_buffer`: Retrieved context from vector stores
- `core_identity`: Agent's identity and beliefs
- `emotional_context`: Emotional intelligence layer
- `active_layer`: Currently selected memory layer
- `needs_new_layer`: Flag for dynamic layer creation

### Dynamic Memory Layer Creation

The agent automatically creates new memory layers when encountering novel topics:
1. Router node analyzes user input
2. Determines if existing layer applies or new layer needed
3. Generates layer description and file-safe name
4. Creates new FAISS vector store
5. Updates `agent_config.json` with incremented version

### Self-Correction Mechanism

The agent implements a "Conscience" module that critiques responses before delivery, enabling self-correction loops for improved accuracy.

## Development Guidelines

### Memory Layer Management
- Memory layers are automatically managed - avoid manual vector store manipulation
- Layer names are generated using `create_layer_name()` utility for file-safe naming
- Each layer has a description and dedicated FAISS vector store

### Configuration Changes
- Agent configuration auto-increments version numbers
- Core identity supports both static values and dynamic layer references
- Use `beliefs_layer` key to reference belief-specific memory layers

### LangGraph Workflow
- The agent uses StateGraph with conditional routing
- Nodes can access and modify the complete AgentState
- Self-correction loops prevent infinite recursion through careful state management

### Error Handling
- API key validation occurs at startup
- Vector store deserialization uses `allow_dangerous_deserialization=True` (required for FAISS)
- Missing files are handled with sensible defaults

### Logging
- `log_thinking()`: Dark green output for internal processing
- `log_response()`: Bold purple output for agent responses