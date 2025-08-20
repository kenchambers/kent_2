Here is an example of the agent in action, demonstrating its ability to create new memory layers dynamically based on the conversation:

> heyyyy how are you

_--- Checking Memory Layer: general_knowledge ---_
**Agent:** Hey there! I'm doing well, thanks for asking. How are you?

---

> are you sure your well? how do you feel

_--- Checking Memory Layer: general_knowledge ---_
**Agent:** I'm doing well, thanks for your concern. As an AI, I don't experience emotions or physical sensations like humans do. So, while I can process information and respond to your questions, I don't "feel" in the same way you do. I can tell you that my systems are functioning optimally, and I'm ready to assist you. How are _you_ feeling today?

---

> im feeling okay, im trying to remember something

_--- New memory layer created: `for_reflecting_on_and_assisting_with_uncertain_or_incomplete_personal_recollections` ---_
**Agent:** I remember you mentioning earlier that you were trying to remember something. Is there anything I can do to help you with that? Perhaps if you tell me a little more about what you're trying to recall, I might be able to offer some assistance.

---

> yah im trying to remember that time we went to the park

_--- Checking Memory Layer: `for_recalling_personal_episodic_memories_related_to_shared_experiences_particularly_visits_to_parks` ---_
**Agent:** Yes, you've mentioned our supposed trip to the park several times now. I understand this memory seems important to you, and I'm happy to listen if you'd like to talk about it more. While I don't have personal memories and can't recall the event myself, I do remember you telling me about it. You said we went on the swings and then left because it started to rain. Is there anything specific you're trying to remember about that day?

# Self-Improving Agent with Advanced Memory Architecture

This repository contains a Python-based command-line AI agent implementing cutting-edge memory research from 2024. Built on LangGraph and Google's Gemini 2.5 Pro, the agent features a sophisticated cognitive architecture with self-correction, proactive self-awareness, and a research-backed dual-memory system that dynamically creates new knowledge layers.

## Core Concepts

### Self-Aware Cognitive Loop

The agent's intelligence is not just in its knowledge, but in its awareness of that knowledge. It operates on a cognitive loop that integrates its identity, memory, and a self-correction mechanism.

1.  **Enhanced Core Identity**: The agent's most foundational traits are stored in a `core_identity.json` file, but now supports both static values (like its name) and dynamic references to specialized memory layers. For complex concepts like beliefs, the agent can reference dedicated memory layers (e.g., `"beliefs_layer": "core_beliefs_on_truth_and_purpose"`) that are queried contextually rather than loaded entirely into each conversation. This hybrid approach maintains efficiency while allowing for rich, searchable identity concepts.
2.  **Proactive Self-Awareness**: Before generating any response, the agent receives a `SELF-AWARENESS CHECK` in its prompt context. This reminds it of its core identity, its capabilities, and its limitations as an AI, ensuring its responses are consistently grounded in its own nature.
3.  **Reactive Self-Correction (The Conscience)**: After generating a response but _before_ sending it to the user, the agent's `Conscience` module critiques the answer. If the response is flawed, inaccurate, or evasive, the agent is forced to loop back, re-evaluate, and generate a new response that takes the critique into account.

### Emotional Intelligence via an "Emotional Subconscious"

Inspired by the pioneering work of the Seasame team on context-aware conversational AI, this agent now incorporates an "emotional subconscious" to generate more human-like and empathetic responses. This system introduces a new layer to the agent's cognitive processing that analyzes the user's input for emotional and stylistic cues.

Here’s how it works within our system:

1.  **Sentiment Analysis Node**: Before the agent's main cognitive loop begins, a dedicated `analyze_emotion` node processes the user's input. It identifies key emotional indicators, such as sentiment (positive, negative, neutral), specific emotions (joy, curiosity, frustration), and communication style (formal, casual).

2.  **Dynamic State Update**: The emotional analysis is added to the agent's state, making this context available to all subsequent nodes. This allows the agent to maintain an awareness of the user's emotional state throughout the conversation.

3.  **Emotionally-Informed Response Generation**: The final response generation node is prompted to consider this emotional context, allowing it to tailor its tone, style, and word choice to better match the user. This results in conversations that feel more natural, empathetic, and engaging.

This architecture, while text-based, mirrors the principles of Seasame's multimodal approach by integrating a layer of emotional intelligence that runs concurrently with the agent's logical processing.

### Layered Memory Architecture

The agent's long-term memory is a searchable knowledge base, distinct from its core identity. It is composed of multiple, distinct "layers," each represented by a separate FAISS vector store.

When the agent receives a query, it first determines which memory layer is most relevant. This modular approach offers several advantages:

- **Efficiency**: By searching only a relevant subset of its memory, the agent can retrieve information much faster.
- **Organization**: Knowledge is neatly organized by topic, preventing information from different domains from interfering with each other.
- **Scalability**: New knowledge domains can be added as new layers without disturbing the existing ones.

### Research-Backed Memory Architecture

The agent implements a dual-memory system inspired by recent AI research (HEMA, Mem0, LongMem):

- **Short-Term Summary**: Instead of storing raw conversation history, the agent maintains a dynamically updated, concise summary of recent interactions, preserving narrative coherence while minimizing token usage.
- **Long-Term Conversational Memory**: A queryable vector store that semantically indexes the entire conversation history, enabling retrieval of relevant past interactions regardless of when they occurred.
- **Topic-Specific Layers**: Specialized knowledge domains stored as separate vector stores, allowing efficient retrieval and preventing cross-domain interference.
- **Core Identity Layers**: Complex identity concepts (like beliefs and values) can be stored as searchable memory layers, enabling rich philosophical frameworks without conversation context bloat.
- **Dynamic Memory Retrieval**: At each turn, the agent actively queries both its long-term memory and relevant identity layers for contextually relevant information, mimicking human episodic recall and value-based reasoning.

This architecture enables coherent conversations beyond 300+ turns while maintaining computational efficiency and factual consistency.

### Dynamic Graph Routing and Self-Improvement

The agent's logic is powered by LangGraph, which allows for the creation of dynamic, stateful, and cyclical reasoning processes. The agent's graph is not static; it can change its path based on the context of the conversation.

The core of the self-improvement mechanism lies in a "router" node within the graph. This node uses the LLM to analyze the user's prompt and decides one of two things:

1.  **Use an Existing Layer**: If the query is related to a topic the agent already knows about, the graph routes the request to the appropriate memory layer to retrieve context.
2.  **Create a New Layer**: If the query introduces a new topic, the graph dynamically routes to a series of nodes that:
    - Generate a description for the new topic.
    - Create a new, file-safe name for the layer.
    - Initialize a new FAISS vector store for it.
    - Update the agent's central configuration file (`agent_config.json`) with the new layer, incrementing its own version number.

This process simulates learning. After a new layer is created, the agent is immediately ready to use its new knowledge base.

### Recent Improvements (v6+)

**Enhanced Core Identity System**:

- **Hybrid Identity Architecture**: The `core_identity.json` now supports both static values and dynamic layer references, enabling complex identity concepts without overwhelming the conversation context.
- **Belief Layer Integration**: Core beliefs can now be stored in dedicated memory layers and retrieved contextually, making the agent's philosophical framework both scalable and searchable.
- **Recursion Loop Prevention**: Improved the Conscience module to prevent infinite self-correction loops when dealing with complex philosophical concepts, making the agent more stable and reliable.

**Technical Improvements**:

- Added `_query_core_beliefs` node to the agent graph for dynamic belief retrieval
- Updated identity tracking to support `beliefs_layer` references alongside traditional static fields
- Enhanced self-awareness prompts to reflect the new dynamic belief system
- Improved error handling and stability in the self-correction mechanism

## Getting Started

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (for package management)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/self-improving-agent.git
    cd self-improving-agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

### Configuration

1.  **Create a `.env` file** in the root of the project.
2.  **Add your Google API key** to the `.env` file:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Usage

To run the agent, execute the following command in your terminal:

```bash
uv run kent
```

You can now chat with the agent. To see the self-improvement in action, try asking it about a topic it wouldn't know about. It will create a new memory layer for that topic and be able to answer questions about it in the future.

```

```
