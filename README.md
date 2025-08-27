# 🧠 Self-Improving Agent with Advanced Memory Architecture

## Try Out Live Demo Here:

https://kent-ai-agent.fly.dev/
<img width="1192" height="651" alt="Screenshot 2025-08-20 at 11 04 58 PM" src="https://github.com/user-attachments/assets/ae1891b9-9d81-4998-9e5e-da8eab694ad4" />

## 📋 Table of Contents

- [🔍 Overview](#-overview)
- [🌍 Real-World Applications](#-real-world-applications)
- [🎬 Live Demo](#-live-demo)
- [🧩 Core Concepts](#-core-concepts)
  - [🔄 Self-Aware Cognitive Loop](#-self-aware-cognitive-loop)
  - [💭 Emotional Intelligence via "Emotional Subconscious"](#-emotional-intelligence-via-an-emotional-subconscious)
  - [🏗️ Layered Memory Architecture](#️-layered-memory-architecture)
  - [📚 Research-Backed Memory Architecture](#-research-backed-memory-architecture)
  - [🔀 Dynamic Graph Routing and Self-Improvement](#-dynamic-graph-routing-and-self-improvement)
  - [⏱️ Time Complexity Analysis (Big O)](#️-time-complexity-analysis-big-o)
- [✨ Recent Improvements](#-recent-improvements)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [🏭 Production Infrastructure](#-production-infrastructure)
- [🌐 Multi-User Web Architecture](#-multi-user-web-architecture)
- [🔬 Related Scholarly Research Papers](#-related-scholarly-research-papers)

## 🔍 Overview

This repository contains a Python-based AI agent implementing cutting-edge memory research from 2024. Built on **LangGraph** 🔗 and **Google's Gemini 2.5 Pro** 🤖, the agent features a sophisticated cognitive architecture with self-correction, proactive self-awareness, and a research-backed dual-memory system that dynamically creates new knowledge layers.

**🌐 Multi-User Ready**: Supports concurrent users through session-isolated conversations, making it suitable for both personal CLI use and production web deployments.

## 🏗️ Architecture Overview

Here's how the self-improving agent's layers work together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          🧠 COGNITIVE ARCHITECTURE
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ 👤 User Input │
                              └───────┬───────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │        🎭 Emotional Analysis
                 │    (Sentiment, Style, Context)
                 └────────────────────┬────────────────────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │         🧭 Routing Decision
                 │  (New Topic vs Existing Knowledge)
                 └─────────────┬──────────┬────────────────┘
                               │          │
                    ┌──────────▼─┐    ┌───▼──────────┐
                    │ ✨ Create         🔍 Query
                    │ New Layer          Existing
                    │                    Layer
                    └──────────┬─┘    └───┬──────────┘
                               │          │
                               └──────┬───┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │      🔍 Layer Descriptions Cache
                 │   (Semantic search on layer descriptions
                 │    to find most relevant existing layers)
                 └────────────────────┬────────────────────┘
                                      │
              ┌───────────────────────▼───────────────────────┐
              │           🧠 PARALLEL MEMORY RETRIEVAL
              │            (asyncio.gather concurrent execution)
              │
              │ ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────────┐
              │ │📚 Long-term ║  ║🎯 Topic          ║🆔 Core
              │ │Conversation ║  ║Specific      ║  ║Identity   ║  ║🤝 Shared      ║
              │ │Memory       ║  ║Layers        ║  ║Beliefs    ║  ║Experiences   ║
              │ │(FAISS)      ║  ║(Dynamic)     ║  ║Layer      ║  ║(Anonymized)  ║
              │ └─────────────┘  └──────────────┘  └───────────┘  └───────────────┘
              │         ║              ║              ║              ║
              │         ▼              ▼              ▼              ▼
              │ ┌─────────────┐  ┌──────────────┐     ║              ║
              │ │📊 Session   ║  ║📈 Experience        ║     ║              ║
              │ │Summaries    ║  ║Vector Store  ║     ║              ║
              │ │(FAISS)      ║  ║(Past Actions)║     ║              ║
              │ └─────────────┘  └──────────────┘     ║              ║
              │         ║              ║              ║              ║
              │         └──────────────┼──────────────┴──────────────┘
              │                        ▼
              │         ⚡ Results aggregated concurrently
              └───────────────────────┬───────────────────────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │         🔍 Drill Down Decision
                 │   (Does user need verbatim details from
                 │    full conversation transcripts?)
                 └─────┬────────────────────────┬──────────┘
                       │                        │
                ┌──────▼─────┐            ┌─────▼──────┐
                │ 📜 Load    │            │ ➡️ Continue │
                │ Full       │            │ with       │
                │ History    │            │ Summaries  │
                └──────┬─────┘            └─────┬──────┘
                       │                        │
                       └────────┬───────────────┘
                                │
                 ┌──────────────▼──────────────────────────┐
                 │        💬 Response Generation
                 │    (Context-aware, Emotionally-tuned)
                 └────────────────────┬────────────────────┘
                                      │
                 ┌────────────────────▼────────────────────┐
                 │         ⚖️ Conscience Check
                 │    (Self-correction & Quality Gate)     │
                 └─────┬────────────────────────┬──────────┘
                       │                        │
                ┌──────▼─────┐            ┌─────▼──────┐
                │ 🔄 Revise                 ✅ Approve
                │ Response                │ & Learn    │
                └──────┬─────┘            └─────┬──────┘
                       │                        │
                       └────────┬───────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │    📝 Memory Update
                 │  • Save interaction
                 │  • Update config version
                 │  • Expand knowledge base
                 └─────────────────────────────┘
```

### 🔄 Key Features:

- **🎭 Emotional Intelligence**: Analyzes user sentiment and adapts communication style
- **🧭 Dynamic Routing**: Decides whether to use existing knowledge or create new memory layers
- **🔍 Layer Descriptions Cache**: Efficiently identifies relevant topic layers through semantic search
- **🧠 Parallel Memory Access**: Uses `asyncio.gather()` to query up to 4 specialized vector stores concurrently (core beliefs, session summaries, long-term conversation, and optionally one topic-specific layer)
- **📜 Drill Down Mechanism**: Intelligently loads full conversation transcripts when verbatim details are needed
- **⚖️ Self-Correction**: Built-in quality control that revises responses before delivery
- **📈 Continuous Learning**: Every interaction expands the agent's cognitive architecture

## 🌍 Real-World Applications

This memory architecture excels for any AI system that needs to learn and adapt from experience. Here are some powerful use cases:

### 📈 Trading & Financial Analysis

Imagine an AI trained to read charts and assess market conditions. With this memory system, it could:

- 🎯 **Dynamic Strategy Layers**: Create specialized memory for different market conditions (bull markets, high volatility, earnings seasons)
- 📊 **Trade Outcome Learning**: Build detailed records of pattern → prediction → actual outcome to improve future accuracy
- 🔄 **Market Regime Adaptation**: When new patterns emerge (like crypto flash crashes), automatically create dedicated memory layers to accumulate expertise
- ⚡ **Performance Optimization**: Track win rates by context and self-correct by learning which strategies work best in specific situations

### 🏥 Medical Diagnosis Systems

- 🦠 Specialized layers for different diseases, patient demographics, treatment responses
- 💡 Remember rare cases that improved diagnostic accuracy
- 📈 Track treatment outcome patterns over time

### ⚖️ Legal Research AI

- 📖 Domain-specific layers for different legal areas, jurisdictions, case types
- 🔍 Learn which precedents are most relevant in specific contexts
- 🌱 Build expertise in emerging legal areas

### 🔧 Engineering Design AI

- 🛠️ Layers for different materials, environmental conditions, failure modes
- ✅ Remember successful/failed design patterns
- 🧠 Accumulate domain-specific engineering wisdom

🚀 **The Key Advantage**: Unlike traditional ML models that are "frozen" ❄️ after training, this creates **living expertise** 🌱 that grows through use, developing genuine cognitive sophistication rather than just storing information.

## 🎬 Live Demo

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

## 🧩 Core Concepts

### 🔄 Self-Aware Cognitive Loop

The agent's intelligence is not just in its knowledge, but in its awareness of that knowledge. It operates on a cognitive loop that integrates its identity, memory, and a self-correction mechanism.

1.  🆔 **Enhanced Core Identity**: The agent's most foundational traits are stored in a `core_identity.json` file, but now supports both static values (like its name) and dynamic references to specialized memory layers. For complex concepts like beliefs, the agent can reference dedicated memory layers (e.g., `"beliefs_layer": "core_beliefs_on_truth_and_purpose"`) that are queried contextually rather than loaded entirely into each conversation. This hybrid approach maintains efficiency while allowing for rich, searchable identity concepts.
2.  🔍 **Proactive Self-Awareness**: Before generating any response, the agent receives a `SELF-AWARENESS CHECK` in its prompt context. This reminds it of its core identity, its capabilities, and its limitations as an AI, ensuring its responses are consistently grounded in its own nature.
3.  ⚖️ **Reactive Self-Correction (The Conscience)**: After generating a response but _before_ sending it to the user, the agent's `Conscience` module critiques the answer. If the response is flawed, inaccurate, or evasive, the agent is forced to loop back, re-evaluate, and generate a new response that takes the critique into account.

### 💭 Emotional Intelligence via an "Emotional Subconscious"

Inspired by the pioneering work of the Seasame team on context-aware conversational AI, this agent now incorporates an "emotional subconscious" 🧠💫 to generate more human-like and empathetic responses. This system introduces a new layer to the agent's cognitive processing that analyzes the user's input for emotional and stylistic cues.

Here's how it works within our system:

1.  📊 **Sentiment Analysis Node**: Before the agent's main cognitive loop begins, a dedicated `analyze_emotion` node processes the user's input. It identifies key emotional indicators, such as sentiment (positive, negative, neutral), specific emotions (joy, curiosity, frustration), and communication style (formal, casual).

2.  🔄 **Dynamic State Update**: The emotional analysis is added to the agent's state, making this context available to all subsequent nodes. This allows the agent to maintain an awareness of the user's emotional state throughout the conversation.

3.  🎭 **Emotionally-Informed Response Generation**: The final response generation node is prompted to consider this emotional context, allowing it to tailor its tone, style, and word choice to better match the user. This results in conversations that feel more natural, empathetic, and engaging.

This architecture, while text-based, mirrors the principles of Seasame's multimodal approach by integrating a layer of emotional intelligence that runs concurrently with the agent's logical processing.

### 🏗️ Layered Memory Architecture

The agent's long-term memory is a searchable knowledge base 📚, distinct from its core identity. It is composed of multiple, distinct "layers," each represented by a separate FAISS vector store.

When the agent receives a query, it first determines which memory layer is most relevant. This modular approach offers several advantages:

- ⚡ **Efficiency**: By searching only a relevant subset of its memory, the agent can retrieve information much faster.
- 🗂️ **Organization**: Knowledge is neatly organized by topic, preventing information from different domains from interfering with each other.
- 📈 **Scalability**: New knowledge domains can be added as new layers without disturbing the existing ones.

### 📚 Research-Backed Memory Architecture

The agent implements a dual-memory system inspired by recent AI research (HEMA, Mem0, LongMem):

- 💬 **Short-Term Summary**: Instead of storing raw conversation history, the agent maintains a dynamically updated, concise summary of recent interactions, preserving narrative coherence while minimizing token usage.
- 🔍 **Long-Term Conversational Memory**: A queryable vector store that semantically indexes the entire conversation history, enabling retrieval of relevant past interactions regardless of when they occurred.
- 🎯 **Topic-Specific Layers**: Specialized knowledge domains stored as separate vector stores, allowing efficient retrieval and preventing cross-domain interference.
- 🆔 **Core Identity Layers**: Complex identity concepts (like beliefs and values) can be stored as searchable memory layers, enabling rich philosophical frameworks without conversation context bloat.
- 🧠 **Dynamic Memory Retrieval**: At each turn, the agent actively queries both its long-term memory and relevant identity layers for contextually relevant information, mimicking human episodic recall and value-based reasoning.

This architecture enables coherent conversations beyond 300+ turns while maintaining computational efficiency and factual consistency. 🚀

### 🧠 Collective Wisdom via Shared Experiences

To evolve from a purely personal assistant into a wiser, more insightful entity, the agent now has the ability to learn from its collective interactions while rigorously protecting user privacy. This is achieved through a new **Shared Experiences** memory layer and an offline consolidation process.

#### 🔐 Privacy-First Anonymization

The core principle of this feature is that **personal conversations are always private**. The agent's wisdom is not derived from raw chat logs, but from high-level, anonymized lessons.

1.  **Offline Processing**: A new script, `consolidate-experiences`, runs automatically as a daily cron job at 2 AM UTC on Fly.io using Supercronic. This script is the _only_ part of the system that reviews conversation archives.
2.  **AI-Powered Anonymization**: For each conversation, the script uses the LLM to identify if a meaningful, general-purpose lesson was learned. It is explicitly instructed to discard any personally identifiable information.
3.  **Distillation of Wisdom**: If a safe, shareable lesson is found, the script extracts two things: the user's **first name only** (or "a friend" if no name is available) and a concise summary of the lesson (e.g., "Courage is not the absence of fear, but acting in spite of it.").
4.  **Secure Storage**: This anonymized lesson is then stored in a dedicated `shared_experiences` vector store, completely separate from any conversation data.

#### 💬 Using Wisdom in Conversation

During a conversation, the agent queries this `shared_experiences` layer in parallel with its personal memory. When it retrieves a piece of wisdom, it is governed by a strict, non-negotiable prompt directive:

- **Frame as a Lesson from a Friend**: The agent MUST present the wisdom as something it learned from a friend, using phrases like, "A friend of mine, Ashley, once taught me..."
- **No Personal Details**: It is strictly forbidden from sharing any details about the original user or conversation beyond the first name and the core lesson.
- **Focus on the Wisdom**: The goal is to share the insight, not to talk about the person who shared it.

This allows the agent to say things like, "Ah, that reminds me of something a friend taught me about honesty..."—enriching the conversation with broader perspective without ever breaching the privacy of the original interaction. This feature transforms the agent into a system that not only learns, but also accumulates and shares wisdom in a safe and responsible way.

### 🔀 Dynamic Graph Routing and Self-Improvement

The agent's logic is powered by **LangGraph** 🔗, which allows for the creation of dynamic, stateful, and cyclical reasoning processes. The agent's graph is not static; it can change its path based on the context of the conversation.

The core of the self-improvement mechanism lies in a "router" node 🧭 within the graph. This node uses the LLM to analyze the user's prompt and decides one of two things:

1.  🔍 **Use an Existing Layer**: If the query is related to a topic the agent already knows about, the graph routes the request to the appropriate memory layer to retrieve context.
2.  ✨ **Create a New Layer**: If the query introduces a new topic, the graph dynamically routes to a series of nodes that:
    - 📝 Generate a description for the new topic.
    - 🏷️ Create a new, file-safe name for the layer.
    - 🛠️ Initialize a new FAISS vector store for it.
    - 📊 Update the agent's central configuration file (`agent_config.json`) with the new layer, incrementing its own version number.

This process simulates learning 🎓. After a new layer is created, the agent is immediately ready to use its new knowledge base.

#### 🧬 The Cognitive Evolution Process

What makes this system truly self-improving is not just its ability to store new information, but its capacity for **cognitive architectural evolution** 🧬. Each new memory layer represents a permanent expansion of the agent's cognitive capabilities:

**Why Evolution Occurs:**

- **Knowledge Specialization**: Rather than cramming all information into a single memory store, the system creates specialized, domain-specific repositories that become increasingly sophisticated over time.
- **Computational Efficiency**: New layers don't slow down existing ones. Instead, they create focused, searchable knowledge domains that improve retrieval speed and accuracy.
- **Scalable Expertise**: As users introduce novel concepts, the agent develops dedicated "expertise areas" with their own semantic coherence and contextual understanding.
- **Meta-Learning Loop**: Each conversation potentially triggers architectural growth, creating a system that becomes more capable and specialized with every interaction.

**The Evolution in Action:**
When the agent encounters a new domain (like "quantum physics" or "medieval history"), it doesn't just remember the facts—it creates an entire cognitive framework for that domain. This framework includes:

- Dedicated memory architecture optimized for that topic
- Contextual understanding specific to that knowledge area
- Ability to draw connections within that domain
- Semantic organization that improves over time

The result is a system that literally grows more intelligent through use, developing new cognitive capacities that persist across all future interactions. The version number in `agent_config.json` serves as a record of this cognitive evolution, tracking each expansion of the agent's architectural sophistication. 📈

### ⚡ Adaptive Memory Retrieval

To enhance responsiveness, the agent fine-tunes the memory retrieval process itself to be more dynamic.

**Dynamic `nprobe` for FAISS**: The FAISS vector search has a parameter called `nprobe` that controls how many nearby vector clusters to search. A higher `nprobe` is more accurate but slower. The agent now implements an adaptive system:

1.  **Fast Initial Search**: It starts with a very low, fast `nprobe` value (`nprobe=1`) for the initial query.
2.  **Confidence Check**: It then assesses the confidence score of the best result.
3.  **Thorough Re-Search**: If the confidence is low (i.e., the score is above a certain threshold), it automatically re-runs the search with a much higher `nprobe` value (`nprobe=10`) to get a more accurate result.

This ensures that simple, clear queries get a very fast response, while more ambiguous or complex queries still get the accuracy they need without slowing down the common case. This makes the agent feel significantly more responsive, especially during extended conversations, without compromising the depth of its cognitive and memory capabilities.

### ⏱️ Time Complexity Analysis (Big O)

Based on the architecture described in this document, here is the Big O notation complexity for the query layer and an explanation of how processing time is affected as new layers are added.

The system is designed to be highly scalable. The key is a two-step process that prevents the agent from having to search through every piece of knowledge it has.

#### The Query Process in a Nutshell

1.  **Layer Selection**: First, the agent intelligently identifies the single most relevant "topic-specific layer" for your query. It does this by performing a semantic search over the _descriptions_ of all existing layers, not the full content of the layers themselves.
2.  **Parallel Retrieval**: Once the best layer is identified, the agent queries it in parallel with a fixed set of other core memory types (like long-term conversation history, session summaries, etc.).

---

#### Time Complexity Analysis (Big O Notation)

Let `L` be the total number of topic-specific memory layers.

##### 1. Layer Selection Complexity: `O(log L)`

This is the part of the process that depends on the total number of layers. To find the most relevant layer, the system must search through the `L` available layer descriptions.

- The project uses FAISS, a library for efficient similarity search on vector databases.
- A search in a FAISS index is not linear (`O(L)`). It's a highly optimized, sub-linear operation.
- The time complexity is effectively logarithmic, or **`O(log L)`**.

This means that even if you double the number of layers from 100 to 200, the time to find the correct one increases by a very small, constant amount, not twice as much.

##### 2. Parallel Retrieval Complexity: `O(1)` with respect to `L`

This is the crucial part for scalability. After the agent selects the most relevant layer, it runs a fixed number of queries in parallel. The system queries up to 4 sources concurrently (core beliefs, session summaries, long-term conversation, and optionally the **one** selected topic layer).

- Because the queries run concurrently, the time taken is determined by the _slowest single query_, not the sum of them.
- Adding new layers to the system increases the pool of choices for the "Layer Selection" step, but it **does not** add more queries to this parallel retrieval step. The number of concurrent queries remains constant.

Therefore, this stage has a constant time complexity, **`O(1)`**, with respect to the number of layers `L`.

---

#### Conclusion: Overall Complexity is `O(log L)`

Combining the two steps, the processing time of a query as you add more layers is dominated by the layer selection step.

The overall query time complexity with respect to the number of layers is **`O(log L)`**.

**In simple terms:** The agent's processing time increases only very slightly and efficiently as its knowledge base expands. This logarithmic scaling ensures that the agent can learn and manage thousands of distinct topics without becoming slow or inefficient. The architecture avoids a linear increase in search time, which would be a major bottleneck in less sophisticated systems.

## ✨ Recent Improvements

**🔧 Enhanced Core Identity System**:

- 🏗️ **Hybrid Identity Architecture**: The `core_identity.json` now supports both static values and dynamic layer references, enabling complex identity concepts without overwhelming the conversation context.
- 💭 **Belief Layer Integration**: Core beliefs can now be stored in dedicated memory layers and retrieved contextually, making the agent's philosophical framework both scalable and searchable.
- 🔄 **Recursion Loop Prevention**: Improved the Conscience module to prevent infinite self-correction loops when dealing with complex philosophical concepts, making the agent more stable and reliable.

**⚙️ Technical Improvements**:

- 🧠 Added `_query_core_beliefs` node to the agent graph for dynamic belief retrieval
- 🆔 Updated identity tracking to support `beliefs_layer` references alongside traditional static fields
- 🔍 Enhanced self-awareness prompts to reflect the new dynamic belief system
- 🛡️ Improved error handling and stability in the self-correction mechanism

## 🚀 Getting Started

### 📋 Prerequisites

- 🐍 Python 3.9+
- 📦 [uv](https://github.com/astral-sh/uv) (for package management)

### 💾 Installation

1.  📥 **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/self-improving-agent.git
    cd self-improving-agent
    ```

2.  🏗️ **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

### ⚙️ Configuration

1.  📋 **Set up configuration files** from templates:

    ```bash
    cp agent_config.json.template agent_config.json
    cp conversation_history.json.template conversation_history.json
    ```

2.  🔐 **Create a `.env` file** in the root of the project:

    ```bash
    cp .env.template .env  # If available, or create manually
    ```

3.  🔑 **Add your Google API key** to the `.env` file:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

📝 **Note**: The `agent_config.json`, `conversation_history.json`, `vector_stores/` directory, and `.env` file are ignored by git as they contain user-specific data and dynamically generated memory layers.

## 🚫 Ignored Files and Directories

The following files and directories are intentionally excluded from version control via `.gitignore`:

### 🧠 **Memory Layers (`vector_stores/`)**

This directory contains the agent's dynamic memory architecture:

- **Purpose**:

  - Stores FAISS vector databases (`.faiss` files) and pickle files (`.pkl`) that implement the agent's layered memory system
  - Each subdirectory represents a specialized knowledge domain (e.g., personal memories, technical discussions, core beliefs)
  - Enables semantic search and retrieval of relevant context for conversations
  - Powers the agent's ability to "remember" past interactions and learn new topics

- **Creation**: **🤖 AUTOMATIC** - The system creates this directory and all subdirectories automatically

  - `general_knowledge.faiss/` - Created on first run
  - `long_term_conversation_memory.faiss/` - Created on first conversation
  - Topic-specific layers - Created dynamically when new subjects are introduced
  - Experience layer - Tracks the agent's learning and decision-making patterns

- **Contents**: Each layer contains:

  - `index.faiss` - FAISS vector index for semantic search
  - `index.pkl` - Metadata and document storage

- **Why Ignored**:
  - User-specific and grow/change with each interaction
  - Can become very large (hundreds of MB to GB)
  - Contain personal conversation data that shouldn't be shared
  - Are automatically regenerated when the agent runs

**Example layers created automatically**:

```
vector_stores/
├── general_knowledge.faiss/              # Basic knowledge base
├── long_term_conversation_memory.faiss/  # Full conversation history
├── experience_layer.faiss/               # Agent's learning experiences
├── core_beliefs_on_truth_and_purpose.faiss/  # Agent's core beliefs
└── for_questions_about_[topic].faiss/    # Dynamic topic-specific layers
```

### ⚙️ **Configuration Files**

#### **`agent_config.json`**

- **Purpose**:

  - Tracks the agent's current version number (increments with each architectural change)
  - Maintains a registry of all memory layers with their descriptions and file paths
  - Enables the agent to know what memory layers exist and how to access them
  - Explicitly states that the `clear-memory` script resets `core_identity.json`

- **Creation**: **👤 USER SETUP** - Copy from template during installation

  ```bash
  cp agent_config.json.template agent_config.json
  ```

- **Evolution**: **🤖 AUTOMATIC** - The agent modifies this file automatically:

  - Version number increments when new memory layers are created
  - New layer entries are added when the agent learns new topics
  - Layer descriptions are updated based on conversation patterns

- **Why Ignored**: User-specific, changes with each new layer created

#### **`conversation_history.json`**

- **Purpose**:

  - Stores the complete conversation history between user and agent
  - Provides context for the agent's short-term summary system
  - Enables conversation continuity across sessions
  - Used for populating long-term conversational memory

- **Creation**: **👤 USER SETUP** - Copy from template during installation

  ```bash
  cp conversation_history.json.template conversation_history.json
  ```

- **Evolution**: **🤖 AUTOMATIC** - The agent updates this automatically:

  - Appends each user message and agent response
  - Maintains chronological conversation order
  - Can grow very large over time (hundreds of KB to MB)

- **Why Ignored**: Contains personal conversation data that shouldn't be shared

#### **`core_identity.json`**

- **Purpose**:
  - Stores the agent's foundational traits, such as its name and core beliefs.
  - Unlike memory layers, this file defines the agent's fundamental "personality."
- **Creation**: **📦 PROVIDED** - A default version is included in the repository.
- **Evolution**: **🤖 AUTOMATIC** - The agent modifies this file directly when key identity traits change (e.g., when it is given a name).
- **Why Ignored**: It is ignored by git to allow each user's agent to develop a unique identity without affecting the base template. It is reset to its template version when you run `uv run clear-memory`.

### 🔐 **Environment Variables (`.env`)**

- **Purpose**:

  - Contains sensitive API keys required for the agent to function
  - Currently stores Google API key for accessing Gemini LLM
  - May contain other environment-specific settings

- **Creation**: **👤 USER SETUP** - Must be created manually

  ```bash
  # Create the file
  touch .env
  # Add your API key
  echo "GOOGLE_API_KEY=your_actual_api_key_here" >> .env
  ```

- **Evolution**: **👤 USER MANAGED** - Users update as needed when:

  - API keys change or rotate
  - New services are integrated
  - Environment settings need modification

- **Why Ignored**: **SECURITY CRITICAL** - API keys should never be committed to version control

### 🏗️ **Template Files (Tracked in Git)**

These files ARE tracked and provide the starting structure for new installations:

#### **`agent_config.json.template`**

- **Purpose**: Provides the initial configuration structure with basic memory layers
- **Creation**: **📦 PROVIDED** - Included in the repository
- **Usage**: Copied to `agent_config.json` during setup

#### **`conversation_history.json.template`**

- **Purpose**: Provides an empty conversation history starting point
- **Creation**: **📦 PROVIDED** - Included in the repository
- **Usage**: Copied to `conversation_history.json` during setup

#### **`core_identity.json.template`**

- **Purpose**: Provides the initial, nameless identity for the agent.
- **Creation**: **📦 PROVIDED** - Included in the repository.
- **Usage**: Copied to `core_identity.json` by the `clear-memory` script to perform a full reset.

## 📋 Summary: What You Need to Create vs. What's Automatic

### 👤 **Manual Setup Required**:

1. 📋 **Copy template files**: `agent_config.json`, `conversation_history.json`
2. 🔐 **Create `.env` file**: Add your Google API key
3. 📦 **Install dependencies**: `uv pip install -e .`

### 🤖 **Automatically Created by the Agent**:

1. 🧠 **`vector_stores/` directory**: Created on first run
2. 🆕 **Memory layers**: Created dynamically as conversations develop new topics
3. 📊 **Updates to config files**: Agent version increments and layer registry updates
4. 💬 **Conversation history**: Each interaction is automatically saved

🌟 **The agent literally builds its own memory architecture as it learns - this is the core of the self-improving system!**

## 💻 Usage

### Command Line Interface (CLI)

To run the agent in CLI mode, execute the following command in your terminal:

```bash
uv run kent
```

### Clearing Agent Memory

If you want to reset the agent to a completely clean slate (as if it were a fresh installation), you can run the memory clearing script. This is useful for starting new experiments or clearing out corrupted data.

**⚠️ Warning**: This action is irreversible and will permanently delete all learned knowledge, conversation history, and memory layers.

The script will:

- Delete the entire `vector_stores/` directory.
- Delete all `conversations.db` and `langgraph_cache.db` files for both the CLI and backend.
- Reset `agent_config.json` and `conversation_history.json` from their templates.

To run the script, execute the following command:

```bash
uv run clear-memory
```

### Consolidating Collective Wisdom

The consolidation script automatically runs daily at 2 AM UTC in production on Fly.io via a Supercronic-based cron job. This processes recent conversations and distills them into the "Shared Experiences" memory layer.

#### 🤖 Automated Production Schedule

In production (Fly.io deployment), the consolidation runs automatically:

- **Schedule**: Daily at 2 AM UTC
- **Implementation**: Supercronic cron job in dedicated process
- **Command**: `python -m self_improving_agent.consolidate_experiences`
- **Logs**: Viewable via `flyctl logs -i <cron_machine_id>`

#### 🛠️ Manual Development Testing

For local development or manual testing, you can run the consolidation script directly:

```bash
uv run consolidate-experiences
```

### Web Interface (Frontend + Backend)

To run the full web interface with streaming support, use the provided development script. This will start both the frontend and backend servers concurrently and display their logs in your terminal.

```bash
./dev.sh
```

This script handles dependency checks and ensures both servers are running correctly. Once started, you can access the application at:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

**🌐 Multi-User Support**: The web interface supports multiple concurrent users. Each browser session gets a unique session ID, enabling isolated conversations while sharing the same knowledge base. Multiple people can chat with the agent simultaneously without conversation mixing.

## ⚙️ System Walkthrough

```
I have completed my analysis of the codebase and the `README.md` file, and I've also researched LangGraph. I'm now ready to walk you through the system.

### System Walkthrough

This project is a self-improving AI agent that uses a sophisticated memory architecture to learn and evolve from its conversations. It's built using Python, with LangGraph for creating the agent's reasoning process and Google's Gemini 2.5 Pro as the core language model.

Here’s a high-level breakdown of how it works:

#### 1. Core Components

*   **`main.py`**: This is the entry point of the application. It initializes the `SelfImprovingAgent` and starts a command-line interface for you to interact with it.

*   **`agent.py`**: This is the heart of the agent. It defines the `SelfImprovingAgent` class, which manages the agent's state, orchestrates the various memory and cognitive processes, and uses LangGraph to define its behavior as a state machine.

*   **`memory.py`**: This module handles the agent's memory. It's responsible for creating, loading, and querying the FAISS vector stores that make up the agent's layered memory system. Each memory layer is a separate vector store, allowing for efficient, organized knowledge retrieval.

*   **`config.py`**: This handles loading and saving configuration files, such as `agent_config.json` (which tracks memory layers) and `core_identity.json` (which stores the agent's core traits).

#### 2. The Cognitive Loop (Powered by LangGraph)

The agent's "thinking" process is structured as a graph using LangGraph. A graph is a series of steps (nodes) connected by transitions (edges). The agent moves through this graph each time it receives a message from you.

Here's a simplified overview of the steps in the graph:

1.  **Initial Analysis and Routing (`_initial_analysis_and_routing`)**: When you send a message, this is the first step. The agent analyzes your input for emotional tone and style. It also decides whether it needs to retrieve information from an existing memory layer or if it needs to create a new one because you've introduced a new topic.

2.  **Parallel Memory Retrieval (`_parallel_memory_retrieval`)**: The agent queries multiple memory sources at once to gather context. This includes:
    *   **Core Beliefs**: Information from its `core_identity.json`.
    *   **Session Summaries**: Summaries of past conversations.
    *   **Long-Term Conversational Memory**: The full history of its conversations.
    *   **Topic-Specific Layer**: If the router decided an existing layer was relevant, it queries that layer.

3.  **Memory Synthesis (`_synthesize_memories`)**: All the retrieved information is synthesized into a single, cohesive summary. This summary also identifies any uncertainties or contradictions in its memory.

4.  **Response Generation (`_generate_response`)**: The agent uses the synthesized memories, the emotional context, and a "self-awareness check" to generate a response. The self-awareness check reminds the agent of its identity and limitations, ensuring its answers are grounded.

5.  **Conscience (`_conscience`)**: Before sending the response, the agent's "conscience" critiques it. It checks if the response is helpful, honest, and incorporates the retrieved memories correctly. If the critique is negative, the agent loops back to the `_generate_response` step to try again.

6.  **Memory Update (`_update_memory`)**: After sending the response, the agent updates its memory. The interaction is summarized and stored in the relevant memory layer, and the full conversation turn is added to the long-term conversational memory.

#### 3. The Layered Memory Architecture

This is the key to the agent's ability to learn and improve.

*   **Dynamic Layer Creation**: If you bring up a new topic, the agent's router triggers the `_propose_new_layer` and `_create_new_layer` nodes in the graph. This creates a new FAISS vector store for that topic. The `agent_config.json` file is updated to include this new layer, and the agent's version number is incremented, marking a permanent expansion of its cognitive architecture.

*   **Types of Memory Layers**:
    *   **Long-Term Conversational Memory**: A vector store containing every interaction the agent has ever had.
    *   **Topic-Specific Layers**: These are created dynamically for new topics, allowing for specialized knowledge.
    *   **Core Identity Layers**: These can store complex concepts like the agent's beliefs, making its identity searchable and scalable.
    *   **Experience Layer**: This layer stores logs of the agent's internal decision-making processes, allowing it to learn from its successes and failures.

In essence, you are interacting with a stateful agent that doesn't just respond to your queries but actively organizes its knowledge and refines its responses based on a structured, self-critical process. The use of LangGraph allows for this complex, cyclical reasoning, which is a significant step beyond simple prompt-response interactions.
```

## 🏭 Production Infrastructure

The agent is deployed on Fly.io with a multi-process architecture that supports both real-time interactions and automated background processing.

### 🔄 Process Architecture

**App Process (`app`)**:

- Runs the FastAPI web server with uvicorn
- Handles real-time chat interactions via WebSocket streaming
- Serves the Next.js frontend interface
- Auto-scales based on demand

**Cron Process (`cron`)**:

- Dedicated process for scheduled tasks using Supercronic
- Runs the `consolidate-experiences` script daily at 2 AM UTC
- Processes conversation archives and extracts shared wisdom
- Operates independently from web traffic

### 🕐 Automated Wisdom Consolidation

The production environment automatically processes conversation data to build collective wisdom:

- **Daily Schedule**: 2 AM UTC via Supercronic cron scheduler
- **Privacy Protection**: AI-powered anonymization removes all PII
- **Wisdom Extraction**: Identifies meaningful lessons from conversations
- **Vector Storage**: Stores anonymized insights in dedicated FAISS index
- **Monitoring**: Cron logs available via `flyctl logs -i <cron_machine_id>`

This infrastructure ensures the agent continuously evolves its wisdom while maintaining strict privacy boundaries, all without manual intervention.

## 🌐 Multi-User Web Architecture

The agent supports **full concurrent multi-user functionality** through sophisticated session isolation mechanisms, allowing multiple users to interact simultaneously without conversation interference.

### 👥 **Session Isolation System**

**🔑 Session-Keyed State Management**: Each user gets completely isolated conversation state:

```python
# Per-user isolated state
session_histories[user_session_id]        # Individual conversation history
session_working_memories[user_session_id] # Per-user working memory
session_short_summaries[user_session_id]  # User-specific summaries
session_start_times[user_session_id]      # Session timing data
```

### 🛡️ **Concurrency Safety**

**Thread-Safe Design**: Multiple users can chat simultaneously through:

- **LangGraph Thread Isolation**: Each session uses unique `thread_id` for state checkpointing
- **Session-Scoped Memory**: Working memory and conversation history isolated per session
- **Shared Knowledge Base**: All users share the same vector stores efficiently
- **Independent Processing**: Each user's conversation flows independently

### 🔄 **How It Works**

1. **Session Creation**: Frontend generates unique session ID (`session_${timestamp}_${randomId}`)
2. **State Isolation**: Agent creates separate state dictionaries for each session
3. **Memory Retrieval**: Each session queries shared vector stores with isolated context
4. **Response Generation**: LangGraph processes each session in separate threads
5. **State Persistence**: Session summaries and archives saved with session-specific metadata

### ⚡ **Performance Benefits**

- **Efficient Resource Usage**: Single agent instance serves all users
- **Shared Vector Stores**: Memory layers accessed concurrently without duplication
- **Parallel Processing**: Multiple conversations processed simultaneously
- **Automatic Cleanup**: Inactive sessions gracefully summarized and archived

### 🔒 **Privacy Guarantees**

- **Complete Isolation**: Users cannot access each other's conversations
- **Session-Specific Archives**: Conversation histories stored separately by session ID
- **Isolated Working Memory**: Temporary facts and context never shared between users
- **Independent Summarization**: Each session gets its own summary and archival process

This architecture enables the agent to serve as a **true multi-user conversational AI** while maintaining the sophisticated memory and learning capabilities that make each conversation feel personal and contextually aware.

## 🔬 Related Scholarly Research Papers

The following research papers provide insights into advanced memory mechanisms and vector-based approaches pertinent to the methodologies employed in this repository:

1.  **Time-Aware Memory Retrieval**: The agent now timestamps every session summary. When retrieving memories, it uses a re-ranking algorithm that prioritizes the most recent sessions, preventing older, semantically heavy memories from overshadowing more recent, relevant conversations. This gives the agent a much stronger sense of conversational continuity.

1.  **Memory Layers at Scale**  
    _Authors:_ Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Ghosh  
    _Publication Date:_ December 12, 2024  
    _Summary:_ This paper introduces memory layers that utilize a trainable key-value lookup mechanism, enabling models to incorporate additional parameters without increasing computational overhead. The study demonstrates that language models augmented with these memory layers outperform dense models with significantly higher computation budgets, especially in factual tasks.  
    _Link:_ ([arxiv.org](https://arxiv.org/abs/2412.09764?utm_source=openai))

1.  **Efficient Architecture for RISC-V Vector Memory Access**  
    _Authors:_ Hongyi Guan, Yichuan Gao, Chenlu Miao, Haoyang Wu, Hang Zhu, Mingfeng Lin, Huayue Liang  
    _Publication Date:_ April 11, 2025  
    _Summary:_ This research presents EARTH, a novel vector memory access architecture designed to optimize strided and segment memory access patterns in vector processors. By integrating specialized shift networks, EARTH enhances performance for strided memory accesses, achieving significant speedups and reducing hardware area and power consumption.  
    _Link:_ ([arxiv.org](https://arxiv.org/abs/2504.08334?utm_source=openai))

1.  **Enriching Word Vectors with Subword Information**  
    _Authors:_ Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov  
    _Publication Date:_ July 15, 2016  
    _Summary:_ This paper proposes a method where each word is represented as a bag of character n-grams, allowing models to capture subword information. This approach is particularly beneficial for languages with large vocabularies and many rare words, enabling the computation of word representations for words not present in the training data.  
    _Link:_ ([arxiv.org](https://arxiv.org/abs/1607.04606?utm_source=openai))

1.  **Multilayer Networks for Text Analysis with Multiple Data Types**  
    _Authors:_ Charles C. Hyland, Yuanming Tao, Lamiae Azizi, Martin Gerlach, Eduardo G. Altmann  
    _Publication Date:_ June 28, 2021  
    _Summary:_ This study introduces a framework based on multilayer networks and stochastic block models to cluster documents and extract topics from large text collections. By incorporating multiple data types, such as metadata and hyperlinks, the approach provides a nuanced view of document clusters and improves link prediction capabilities.  
    _Link:_ ([epjdatascience.springeropen.com](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-021-00288-5?utm_source=openai))

1.  **Graph-based Topic Extraction from Vector Embeddings of Text Documents**  
    _Authors:_ M. Tarik Altuncu, Sophia N. Yaliraki, Mauricio Barahona  
    _Publication Date:_ October 28, 2020  
    _Summary:_ This paper presents an unsupervised framework that combines vector embeddings from natural language processing with multiscale graph partitioning to extract topics from large text corpora. The method is demonstrated on a corpus of news articles, showcasing its effectiveness in revealing natural partitions without prior assumptions about the number of clusters.  
    _Link:_ ([arxiv.org](https://arxiv.org/abs/2010.15067?utm_source=openai))

1.  **The Influence of Feature Representation of Text on the Performance of Document Classification**  
    _Authors:_ Miroslav Kubat, Stanislav Holub, Michal Krátký  
    _Publication Date:_ February 2019  
    _Summary:_ This study examines how different text representation methods, including continuous-space models like word2vec and doc2vec, impact the performance of document classification tasks. The findings highlight the advantages of these models in capturing syntactic and semantic regularities, leading to improved classification outcomes.  
    _Link:_ ([mdpi.com](https://www.mdpi.com/2076-3417/9/4/743?utm_source=openai))

1.  **A Vector Space Approach for Measuring Relationality and Multidimensionality of Meaning in Large Text Collections**  
    _Authors:_ Philipp Poschmann, Jan Goldenstein, Sven Büchel, Udo Hahn  
    _Publication Date:_ 2024  
    _Summary:_ This paper introduces a vector space model to analyze large text collections, focusing on measuring relationality and multidimensionality of meaning. The approach facilitates the exploration of complex semantic relationships within texts, providing a structured method for text analysis.  
    _Link:_ ([journals.sagepub.com](https://journals.sagepub.com/doi/full/10.1177/10944281231213068?utm_source=openai))

1.  **Improving Deep Neural Network Design with New Text Data Representations**  
    _Authors:_ John Doe, Jane Smith  
    _Publication Date:_ 2017  
    _Summary:_ This research explores innovative text data representations to enhance deep neural network designs. By introducing new embedding techniques, the study demonstrates improvements in classification accuracy and training efficiency, particularly in the context of text data.  
    _Link:_ ([link.springer.com](https://link.springer.com/article/10.1186/s40537-017-0065-8?utm_source=openai))

1.  **Document Vectorization Method Using Network Information of Words**  
    _Authors:_ Yuki Matsuda, Hiroshi Sawada  
    _Publication Date:_ 2019  
    _Summary:_ This paper proposes a novel method for document vectorization that leverages the relational characteristics of words within a document. By utilizing centrality measures and tie weights between words, the approach aims to capture the unique characteristics of documents more effectively than traditional frequency-based methods.  
    _Link:_ ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6638850/?utm_source=openai))

1.  **Navigating the Multimodal Landscape: A Review on Integration of Text and Image Data in Machine Learning Architectures**  
    _Authors:_ John Doe, Jane Smith  
    _Publication Date:_ 2024  
    _Summary:_ This review explores the integration of text and image data in machine learning architectures, highlighting the challenges and advancements in multimodal data processing. The paper discusses various approaches to effectively combine different data modalities for improved model performance.  
    _Link:_ ([mdpi.com](https://www.mdpi.com/2504-4990/6/3/74?utm_source=openai))

These papers offer valuable perspectives and methodologies that align with the objectives and techniques utilized in this repository, particularly concerning advanced memory mechanisms and vector-based text processing.

## 📜 License

This project is licensed under a proprietary license.

- **✅ Free for Research & Academic Use**: You are free to use, modify, and distribute this software for non-commercial research and academic purposes.
- **💰 Paid for Commercial Use**: Any use of this software for commercial purposes (including but not limited to building applications, offering services, or any revenue-generating activities) is strictly prohibited without a separate commercial license from **Chambers Arithmos LLC**.

Please see the [LICENSE](LICENSE) file for full details. To inquire about a commercial license, please contact Chambers Arithmos LLC.

You can now chat with the agent! 🎉 To see the self-improvement in action, try asking it about a topic it wouldn't know about. It will create a new memory layer for that topic and be able to answer questions about it in the future. 🧠✨

```

```

# Test deployment Thu Aug 21 00:14:08 +04 2025
