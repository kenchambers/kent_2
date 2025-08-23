"""
This module defines the SelfImprovingAgent class, which is the core of the
self-improving agent.
"""
import json
import asyncio
import aiosqlite
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from datetime import datetime
from . import config
from . import memory
from .utils import create_layer_name, log_thinking


class AgentState(TypedDict):
    """
    Represents the state of the agent at any given time.
    """
    user_input: str
    conversation_history: List[Dict[str, str]]
    short_term_summary: str  # A running summary of the recent conversation

    semantic_cache: Dict[str, Any]  # Caching factual lookups
    active_layer: Optional[str]
    layer_info: Dict[str, Any]
    new_layer_description: Optional[str]
    needs_new_layer: bool
    response: str
    inner_monologue: Optional[str]
    critique: Optional[str]
    past_experiences: Optional[str]
    core_identity: Dict[str, Any]
    emotional_context: Optional[str]

    working_memory: Dict[str, Any]  # Key facts from current conversation (names, roles, etc)
    retrieved_docs: List[Any] # To hold retrieved Document objects with metadata
    drill_down_needed: bool # Flag to signal if we need to load full history
    archive_to_load: Optional[str] # Path to the specific archive file
    full_history_context: Optional[str] # Content of the loaded archive
    revision_count: int # To prevent self-correction loops
    layer_creation_attempts: int # To prevent layer creation loops
    recent_sessions_context: Optional[str] # For prioritizing recent session summaries


class SelfImprovingAgent:
    """
    A self-improving agent that can learn from its interactions with the user.
    """

    def __init__(self):
        self.config = config.get_config()
        self.core_identity = config.get_core_identity()
        log_thinking(f"--- Core Identity Loaded: {self.core_identity} ---")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            google_api_key=config.load_api_key()
        )
        self.fast_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=config.load_api_key()
        )
        # AsyncSqliteSaver will be initialized in arun when we have async context
        self.checkpointer = None
        self.conn = None
        self.graph = None  # Will be built when checkpointer is ready
        
        # Session management
        self.current_session_history: List[Dict[str, str]] = []  # Only current session
        self.session_id = None
        self.session_start_time = None
        
        # Memory management
        self.short_term_summary = ""
        self.working_memory: Dict[str, Any] = {}  # Persistent facts for current session
        self.conversation_window_size = 10  # Only keep last N exchanges in context
        
        # Vector stores
        self.experience_vector_store = memory.get_experience_vector_store()
        self.long_term_memory = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss")
        )
        self.session_summaries = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "session_summaries.faiss")
        )
        
        # Layer routing optimization
        self.layer_descriptions_cache = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "layer_descriptions.faiss")
        )
        self._layer_cache_version = None
        self._cached_layer_descriptions = None
        
        self.constitution = [
            "Be helpful and factual.",
            "Do not be evasive.",
            "Consider the user's likely intent.",
            "Admit when you don't know something.",
        ]
        self.simple_greetings = re.compile(
            r"^\s*(hi|hello|hey|yo|heya|howdy|sup|what's up|greetings)(\s+\w+)?\s*$",
            re.IGNORECASE
        )

    async def _update_layer_descriptions_cache(self):
        """
        Updates the layer descriptions cache if the config has changed.
        """
        current_version = self.config.get("version", 1)
        
        if self._layer_cache_version != current_version:
            log_thinking(f"--- Updating Layer Descriptions Cache (v{current_version}) ---")
            
            # Clear existing cache
            self.layer_descriptions_cache = memory.get_vector_store(
                str(config.VECTOR_STORES_DIR / "layer_descriptions.faiss")
            )
            
            # Add all layer descriptions to the cache
            for layer_name, layer_config in self.config['layers'].items():
                description_with_name = f"{layer_name}: {layer_config['description']}"
                memory.add_to_memory(
                    self.layer_descriptions_cache,
                    description_with_name,
                    str(config.VECTOR_STORES_DIR / "layer_descriptions.faiss")
                )
            
            self._layer_cache_version = current_version
            log_thinking(f"--- Cached {len(self.config['layers'])} layer descriptions ---")
    
    async def _get_relevant_layers(self, user_input: str, k: int = 20) -> Dict[str, Any]:
        """
        Gets the most relevant memory layers for the user input using semantic search.
        """
        await self._update_layer_descriptions_cache()
        
        # Query the layer descriptions cache
        relevant_docs = memory.query_memory(
            self.layer_descriptions_cache, 
            user_input, 
            k=min(k, len(self.config['layers']))
        )
        
        # Extract layer names from the results and build the relevant layers dict
        relevant_layers = {}
        
        if relevant_docs:
            # Parse the returned documents to extract layer names
            for doc in relevant_docs:
                content = doc.page_content
                if ':' in content:
                    # Extract layer name from format "layer_name: description"
                    layer_name = content.split(':', 1)[0].strip()
                    if layer_name in self.config['layers']:
                        relevant_layers[layer_name] = self.config['layers'][layer_name]
        
        # Fallback: if no relevant layers found, return a few recent ones
        if not relevant_layers:
            layer_names = list(self.config['layers'].keys())[-5:]  # Last 5 layers
            relevant_layers = {name: self.config['layers'][name] for name in layer_names}
        
        log_thinking(f"--- Found {len(relevant_layers)} relevant layers: {list(relevant_layers.keys())} ---")
        return relevant_layers

    async def _initial_analysis_and_routing(self, state: AgentState) -> AgentState:
        """
        Combines emotional analysis and routing into a single LLM call.
        Uses smart layer filtering to only present relevant layers.
        """
        log_thinking("--- Initial Analysis and Routing ---")
        past_experiences = memory.query_memory(
            self.experience_vector_store,
            state["user_input"]
        )
        state["past_experiences"] = past_experiences
        
        # Get only the most relevant layers instead of all layers
        relevant_layers = await self._get_relevant_layers(state["user_input"], k=20)

        prompt = f"""
        Analyze the user's input and determine the next action.

        User input: "{state['user_input']}"

        Here is the summary of the recent conversation for context:
        --- RECENT CONVERSATION SUMMARY ---
        {state['short_term_summary']}
        --- END RECENT CONVERSATION SUMMARY ---

        Here are the most relevant available memory layers (out of {len(self.config['layers'])} total):
        {json.dumps(relevant_layers, indent=2)}

        Here are some relevant past experiences:
        {past_experiences}

        Respond with a JSON object containing two keys:
        1. "emotional_context": A concise, one-sentence summary of the user's emotional and stylistic state (e.g., "The user seems curious and is communicating in a casual style.").
        2. "routing_decision": An object with your thought process for routing. It should contain:
           - "thought": Your reasoning process.
           - "action": Either a specific layer name from the available layers, "new_layer" to create a new layer, or "none"/"no_action_needed" for simple responses that don't require memory retrieval.
           - "confidence": A percentage of your confidence in this action.
           - "self_correction_plan": What to do if the action fails.

        ROUTING GUIDELINES:
        - Create "new_layer" for specific technical topics, hobbies, or specialized subjects (e.g., astrophotography, quantum physics, cooking techniques, specific games, etc.)
        - Use "general_knowledge" ONLY for very broad, general questions or simple greetings
        - If the user introduces a specific technique, method, or specialized topic, prefer "new_layer"
        - When in doubt between an existing layer and creating a new one, prefer creating a new layer for better organization

        Example JSON response:
        {{
            "emotional_context": "The user seems curious and is communicating in a casual style.",
            "routing_decision": {{
                "thought": "The user is asking about the history of the internet. The 'history_of_technology' layer seems relevant.",
                "action": "history_of_technology",
                "confidence": "90%",
                "self_correction_plan": "If the 'history_of_technology' layer does not have the answer, I will create a new layer called 'history_of_the_internet'."
            }}
        }}
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        try:
            # Extract JSON from the response, assuming it might be wrapped in markdown
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]

            if not json_str:
                raise json.JSONDecodeError("No JSON object found in response", response_text, 0)

            parsed_json = json.loads(json_str)
            emotional_context = parsed_json.get("emotional_context")
            routing_decision = parsed_json.get("routing_decision")

            state["emotional_context"] = emotional_context
            log_thinking(f"--- Emotional Context: {emotional_context} ---")

            monologue = json.dumps(routing_decision, indent=2)
            state["inner_monologue"] = monologue
            log_thinking(f"--- Inner Monologue ---\n{monologue}")

            action = routing_decision.get("action", "new_layer")

            if action == "new_layer":
                state["needs_new_layer"] = True
            elif action in ["none", "no_action_needed"]:
                state["needs_new_layer"] = False
                state["active_layer"] = None
            else:
                state["needs_new_layer"] = False
                state["active_layer"] = action

        except (json.JSONDecodeError, KeyError) as e:
            log_thinking(f"--- Initial Analysis: Invalid JSON from LLM --- \nLLM Response: '{response_text}'\nError: {e}")
            # Fallback or error handling logic here
            state["needs_new_layer"] = True # Default to creating a new layer on failure
            state["emotional_context"] = "Could not determine emotional context."

        return state

    async def _get_dynamic_memory(self, state: AgentState) -> str:
        """Helper to retrieve from long-term conversational memory."""
        log_thinking("--- Querying Long-Term Conversational Memory ---")
        user_input = state["user_input"]
        working_memory = state.get("working_memory", {})

        # Construct a more focused query from the working memory
        topic = working_memory.get("topic")
        question = working_memory.get("question")

        if topic and question:
            focused_query = f"Regarding the topic of {topic}: {question}"
        elif topic:
            focused_query = f"Information about the topic: {topic}"
        else:
            focused_query = user_input

        log_thinking(f"--- Focused long-term memory query: '{focused_query}' ---")
        return memory.query_memory(self.long_term_memory, focused_query)
    
    async def _get_session_summaries(self, query: str) -> List[Document]:
        """
        Retrieves the most relevant session summaries, re-ranked by recency.
        """
        log_thinking("--- Querying Past Session Summaries ---")
        # Fetch a larger number of docs to re-rank
        try:
            return await self.session_summaries.asimilarity_search(query, k=10)
        except Exception as e:
            log_thinking(f"Error querying session summaries: {e}")
            return []

    async def _get_core_beliefs(self, user_input: str) -> List[Document]:
        """Helper to retrieve from core belief layers."""
        retrieved_beliefs: List[Document] = []
        for key, value in self.core_identity.items():
            if key.endswith("_layer"):
                layer_name = value
                if layer_name in self.config["layers"]:
                    log_thinking(f"--- Checking Core Beliefs Layer: {layer_name} ---")
                    layer_config = self.config["layers"][layer_name]
                    vector_store = memory.get_vector_store(
                        layer_config["vector_store_path"]
                    )
                    retrieved_info = memory.query_memory(vector_store, user_input)
                    if retrieved_info:
                        log_thinking(f"--- Retrieved {len(retrieved_info)} beliefs from {layer_name} ---")
                        retrieved_beliefs.extend(retrieved_info)
        return retrieved_beliefs

    async def _get_existing_layer_info(self, layer_name: str, user_input: str) -> Optional[List[Document]]:
        """Helper to retrieve from an existing layer."""
        if layer_name and layer_name in self.config["layers"]:
            log_thinking(f"--- Checking Memory Layer: {layer_name} ---")
            layer_config = self.config["layers"][layer_name]
            vector_store = memory.get_vector_store(layer_config["vector_store_path"])
            retrieved_info = memory.query_memory(vector_store, user_input)
            log_thinking(f"--- Retrieved Info: {retrieved_info} ---")
            return retrieved_info
        return None

    async def _parallel_memory_retrieval(self, state: AgentState) -> AgentState:
        """
        Retrieves information from various memory sources in parallel.
        Also speculatively proposes a new layer in case it's needed.
        """
        log_thinking("--- Parallel Memory Retrieval (with Speculative Layer Proposal) ---")
        user_input = state["user_input"]
        active_layer = state.get("active_layer")

        tasks = {
            "beliefs": self._get_core_beliefs(user_input),
            "sessions": self._get_session_summaries(user_input),
            "dynamic": self._get_dynamic_memory(state),
        }

        if active_layer and not state["needs_new_layer"]:
            tasks["layer_info"] = self._get_existing_layer_info(active_layer, user_input)

        # Speculatively propose a new layer in parallel (we might need it)
        if state.get("needs_new_layer"):
            tasks["speculative_proposal"] = self._propose_new_layer(state)

        # Use asyncio.gather to run all tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        results_map = dict(zip(tasks.keys(), results))
        all_docs = []
        
        # Special handling for session summaries to re-rank by recency
        if 'sessions' in results_map and isinstance(results_map['sessions'], list):
            # Pop the session docs to handle them separately and add them to all_docs
            session_docs = results_map.pop('sessions')
            all_docs.extend(session_docs)
            log_thinking(f"--- Retrieved {len(session_docs)} docs from sessions ---")

            log_thinking("--- Re-ranking session summaries by recency ---")
            
            # Sort docs by creation_timestamp, newest first
            # We parse the ISO string back to a datetime object for safe comparison
            sorted_sessions = sorted(
                session_docs,
                key=lambda doc: datetime.fromisoformat(ts) if (ts := doc.metadata.get('creation_timestamp')) else datetime.min,
                reverse=True
            )
            
            # Keep only the top N most recent summaries
            recent_sessions = sorted_sessions[:3]
            state['recent_sessions_context'] = "\n".join([doc.page_content for doc in recent_sessions])
            log_thinking(f"--- Top {len(recent_sessions)} recent sessions selected and isolated ---")

        for task_name, result in results_map.items():
            if isinstance(result, Exception):
                log_thinking(f"--- Error during {task_name} retrieval: {result} ---")
                continue

            if result and isinstance(result, list):
                # Now all results are expected to be List[Document]
                all_docs.extend(result)
                log_thinking(f"--- Retrieved {len(result)} docs from {task_name} ---")

        # Handle speculative proposal result if it exists
        if 'speculative_proposal' in results_map and not isinstance(results_map['speculative_proposal'], Exception):
            # The proposal has already updated the state, no need to do anything
            log_thinking("--- Speculative layer proposal completed ---")

        state['retrieved_docs'] = all_docs
        return state

    def _build_graph(self):
        """
        Builds the agent's graph.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node(
            "initial_analysis_and_routing", self._initial_analysis_and_routing
        )
        workflow.add_node(
            "parallel_memory_retrieval", self._parallel_memory_retrieval
        )
        workflow.add_node("propose_new_layer", self._propose_new_layer)
        workflow.add_node("create_new_layer", self._create_new_layer)
        workflow.add_node("generate_and_correct_response", self._generate_and_correct_response)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node(
            "_update_short_term_summary", self._update_short_term_summary
        )
        workflow.add_node("query_semantic_cache", self._query_semantic_cache)
        workflow.add_node("update_core_identity", self._update_core_identity)
        workflow.add_node("update_working_memory", self._update_working_memory)
        workflow.add_node("decide_drill_down", self._decide_drill_down)
        workflow.add_node("load_full_history", self._load_full_history)
        workflow.add_node("verify_layer_creation", self._verify_layer_creation)

        workflow.set_entry_point("initial_analysis_and_routing")

        workflow.add_edge("initial_analysis_and_routing", "update_working_memory")
        workflow.add_edge("update_working_memory", "parallel_memory_retrieval")

        workflow.add_conditional_edges(
            "parallel_memory_retrieval",
            lambda state: "propose_new_layer"
            if state["needs_new_layer"]
            else "query_semantic_cache",
        )
        workflow.add_edge("propose_new_layer", "create_new_layer")
        workflow.add_edge("create_new_layer", "verify_layer_creation")
        
        workflow.add_conditional_edges(
            "verify_layer_creation",
            self._decide_after_layer_creation
        )
        
        workflow.add_edge("query_semantic_cache", "decide_drill_down")
        
        workflow.add_conditional_edges(
            "decide_drill_down",
            lambda state: "load_full_history" if state.get("drill_down_needed") else "generate_and_correct_response"
        )
        workflow.add_edge("load_full_history", "generate_and_correct_response")

        workflow.add_edge("generate_and_correct_response", "update_core_identity")
        workflow.add_edge("update_core_identity", "update_memory")
        workflow.add_edge("update_memory", "_update_short_term_summary")
        workflow.add_edge("_update_short_term_summary", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _update_short_term_summary(self, state: AgentState) -> AgentState:
        """
        Updates the short-term summary of the conversation.
        """
        working_memory_str = ""
        if state.get('working_memory'):
            working_memory_str = f"\n\nKey facts to preserve: {json.dumps(state['working_memory'])}"
        
        prompt = f"""
        Given the previous summary and the latest exchange, create a new, concise
        summary of the conversation that preserves all important information.

        Previous Summary:
        "{state['short_term_summary']}"

        Latest Exchange:
        User: "{state['user_input']}"
        Agent: "{state['response']}"
        {working_memory_str}

        Create a summary that:
        1. Captures the conversation flow and main topics
        2. EXPLICITLY mentions any names or personal details shared
        3. Notes key decisions or preferences expressed
        4. Preserves context about ongoing discussions
        
        New Summary:
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["short_term_summary"] = response.content.strip()
        log_thinking(
            "--- Short-Term Summary Updated: "
            f"{state['short_term_summary']} ---"
        )
        return state

    async def _query_semantic_cache(self, state: AgentState) -> AgentState:
        # Placeholder - simple check if input is in cache
        if state["user_input"] in state["semantic_cache"]:
            log_thinking(
                f"--- Semantic Cache Hit for: {state['user_input']} ---"
            )
        return state
    
    def _decide_after_layer_creation(self, state: AgentState):
        """
        Decides the next step after attempting to create a layer.
        Handles retry logic.
        """
        MAX_ATTEMPTS = 2
        
        if state.get("active_layer") and state["active_layer"] in self.config["layers"]:
            # Success
            log_thinking(f"--- Layer '{state['active_layer']}' successfully verified. ---")
            return "query_semantic_cache"
        else:
            # Failure
            attempts = state.get('layer_creation_attempts', 0)
            if attempts < MAX_ATTEMPTS:
                log_thinking(f"--- Layer creation failed (Attempt {attempts + 1}). Retrying. ---")
                return "propose_new_layer" # Loop back to try again
            else:
                log_thinking(f"--- Layer creation failed after {MAX_ATTEMPTS} attempts. Proceeding without new layer. ---")
                # Reset the flag so we don't try again on the next turn
                state['needs_new_layer'] = False
                return "query_semantic_cache" # Move on

    async def _verify_layer_creation(self, state: AgentState) -> AgentState:
        """
        Verifies that the new layer was successfully created and saved to config.
        """
        state['layer_creation_attempts'] = state.get('layer_creation_attempts', 0) + 1
        
        # Re-load config from disk to ensure we have the latest version
        self.config = config.get_config()
        
        layer_name = state.get("active_layer")
        if layer_name and layer_name in self.config["layers"]:
            log_thinking(f"--- Verification successful for layer: {layer_name} ---")
        else:
            log_thinking(f"--- Verification FAILED for layer: {layer_name}. Not found in config. ---")
            # Clear the active layer since it wasn't created properly
            state['active_layer'] = None

        return state

    async def _decide_drill_down(self, state: AgentState) -> AgentState:
        """
        Analyzes if a retrieved summary is sufficient or if the full history is needed.
        """
        log_thinking("--- Deciding on Memory Drill-Down ---")
        state['drill_down_needed'] = False # Default

        # Find the most relevant session summary with an archive path
        most_relevant_session = None
        for doc in state.get('retrieved_docs', []):
            if 'archive_path' in doc.metadata:
                # Simple logic for now: pick the first one.
                # A more advanced version could score them.
                most_relevant_session = doc
                break

        if not most_relevant_session:
            log_thinking("No archived sessions retrieved. Skipping drill-down.")
            return state

        prompt = f"""
        You need to decide if you have enough information to answer the user's query or if you need to access the full conversation transcript.

        USER'S QUERY: "{state['user_input']}"

        Here is the most relevant summary of a past conversation:
        ---
        {most_relevant_session.page_content}
        ---

        Analyze the user's query. If it asks for a very specific, verbatim detail (like an exact quote, a specific number, a line of code, or a precise detail that is unlikely to be in a summary), you should drill down. Otherwise, the summary is sufficient.

        Respond with ONLY a JSON object with one key, "drill_down", set to either true or false.

        Example 1:
        User Query: "What was that exact error message we discussed last Tuesday?"
        Your Response: {{"drill_down": true}}

        Example 2:
        User Query: "What were we talking about last week?"
        Your Response: {{"drill_down": false}}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision_text = response.content.strip()

        try:
            # Extract JSON from the response, assuming it might be wrapped in markdown
            if "```json" in decision_text:
                json_str = decision_text.split("```json")[1].split("```")[0].strip()
            else:
                # Fallback for raw JSON
                json_str = decision_text[decision_text.find('{'):decision_text.rfind('}') + 1]

            if not json_str:
                 raise json.JSONDecodeError("No JSON object found in response", decision_text, 0)
                 
            decision = json.loads(json_str)
            if decision.get("drill_down"):
                state['drill_down_needed'] = True
                state['archive_to_load'] = most_relevant_session.metadata['archive_path']
                log_thinking(f"Drill-down is NEEDED. Will load: {state['archive_to_load']}")
            else:
                log_thinking("Drill-down is NOT needed. Summary is sufficient.")
        except (json.JSONDecodeError, KeyError):
            log_thinking(f"Could not parse drill-down decision: {decision_text}. Defaulting to no drill-down.")
        
        return state

    async def _load_full_history(self, state: AgentState) -> AgentState:
        """
        Loads the full conversation history from the specified archive file.
        """
        archive_path = state.get('archive_to_load')
        if not archive_path:
            return state
            
        log_thinking(f"--- Loading Full History from {archive_path} ---")
        try:
            # Load file asynchronously
            history_data = await asyncio.to_thread(
                lambda: json.load(open(archive_path, 'r'))
            )
            
            # Format the history for context
            history_text = "--- Full Conversation Transcript ---\n"
            for message in history_data:
                history_text += f"{message['role'].capitalize()}: {message['content']}\n"
            history_text += "--- End of Transcript ---"
            
            state['full_history_context'] = history_text
            log_thinking("Successfully loaded and formatted full history.")
            
        except Exception as e:
            log_thinking(f"Error loading archive file {archive_path}: {e}")
            state['full_history_context'] = "Error: Could not load the full conversation history."
            
        return state

    async def _update_working_memory(self, state: AgentState) -> AgentState:
        """
        Extracts and maintains key facts from the conversation in working memory.
        """
        log_thinking("--- Updating Working Memory ---")
        
        # Extract key facts from user input
        extraction_prompt = f"""
        Analyze this user input and extract any key facts that should be remembered during the conversation:
        User input: "{state['user_input']}"
        
        Current working memory: {state.get('working_memory', {})}
        
        Extract facts like:
        - The user's name if they introduce themselves
        - Any personal information they share
        - Specific topics or questions they're asking about
        - Any corrections to previous information
        
        Respond with a JSON object of key-value pairs. For example:
        {{"user_name": "Ken", "topic": "memory issues", "concern": "AI forgetting names"}}
        
        If no new facts to add, respond with: {{}}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=extraction_prompt)])
        response_text = response.content.strip()
        
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
            
            new_facts = json.loads(json_str)
            if new_facts:
                state.setdefault('working_memory', {}).update(new_facts)
                log_thinking(f"--- Working Memory Updated: {new_facts} ---")
        except (json.JSONDecodeError, IndexError) as e:
            log_thinking(f"--- Working Memory: Could not extract facts --- Error: {e}")
        
        return state

    async def _propose_new_layer(self, state: AgentState) -> AgentState:
        prompt = f"""
        You are a memory architect. Your job is to create a new memory layer
        for the agent.
        The user's request is: "{state['user_input']}"
        Based on this request, create a concise, one-sentence description for
        a new memory layer.
        For example, if the user asks about the history of Rome, the
        description could be: "For questions about the history of ancient
        Rome."
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["new_layer_description"] = response.content.strip()
        return state

    async def _create_new_layer(self, state: AgentState) -> AgentState:
        description = state["new_layer_description"]
        layer_name = create_layer_name(description)

        new_layer_config = {
            "description": description,
            "vector_store_path": str(
                config.VECTOR_STORES_DIR / f"{layer_name}.faiss"
            ),
        }

        # Create the new vector store with the user's input as the first document.
        initial_doc = Document(page_content=state["user_input"])
        memory.get_vector_store(
            new_layer_config["vector_store_path"],
            initial_documents=[initial_doc]
        )

        # Update and save the configuration
        self.config["version"] += 1
        self.config["layers"][layer_name] = new_layer_config
        await asyncio.to_thread(config.save_config, self.config)

        # Invalidate layer cache since we added a new layer
        self._layer_cache_version = None

        state["active_layer"] = layer_name
        log_thinking(
            f"Version {self.config['version']}: New layer '{layer_name}' created."
        )

        # Check if this new layer relates to core beliefs and update identity if so
        await self._check_and_update_belief_layer(description, layer_name)

        return state

    async def _check_and_update_belief_layer(self, layer_description: str, layer_name: str):
        """
        Checks if a new layer is about beliefs and updates core_identity.json if it is.
        """
        prompt = f"""
        Analyze the description of a new memory layer.
        Layer Description: "{layer_description}"

        Does this description relate to the agent's core beliefs, philosophy, ethics, or principles?

        Respond with only "yes" or "no".
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()

        if "yes" in decision:
            log_thinking(f"--- New layer '{layer_name}' identified as a beliefs layer. Updating core identity. ---")
            current_identity = config.get_core_identity()
            if current_identity.get("beliefs_layer") != layer_name:
                current_identity["beliefs_layer"] = layer_name
                await asyncio.to_thread(config.save_core_identity, current_identity)
                self.core_identity = current_identity  # Update in-memory identity
                log_thinking(f"--- Core identity 'beliefs_layer' updated to '{layer_name}'. ---")


    async def _generate_and_correct_response(self, state: AgentState) -> AgentState:
        """
        Generates a response and performs a single-pass self-correction.
        The initial response is generated by the main LLM (pro), and the critique
        and final response generation is handled by the fast LLM (flash).
        """
        log_thinking("--- Generating and Correcting Response (Single Pass) ---")

        # Step 1: Consolidate all context for the generator
        context = self._build_response_context(state)

        # System prompt for the generator
        generator_system_prompt = f"""
        You are a helpful AI assistant (version {self.config['version']}) with a persistent, layered memory.
        Your task is to generate the best possible response to the user's query based on the extensive context provided.
        Pay close attention to your memory, emotional context, and self-awareness checks.
        """
        
        generator_messages = [
            SystemMessage(content=generator_system_prompt),
            HumanMessage(content=f"{context}\n\nUser's current message: \"{state['user_input']}\"")
        ]

        # Generate the initial draft response using the main LLM
        initial_response = await self.llm.ainvoke(generator_messages)
        initial_response_text = initial_response.content.strip()
        log_thinking(f"--- Initial Draft Response: {initial_response_text} ---")

        # Step 2: Perform self-correction using the fast LLM
        correction_prompt = f"""
        You are the 'Conscience' of an AI. Your job is to critique and refine a draft response.

        Here is all the context the AI was given:
        --- START CONTEXT ---
        {context}
        --- END CONTEXT ---

        Here is the user's query:
        "{state['user_input']}"

        Here is the AI's draft response:
        "{initial_response_text}"

        Critique the draft based on these rules: {self.constitution}
        
        ADDITIONAL CRITICAL CHECKS:
        1.  **MEMORY USAGE**: Does the response properly incorporate specific, relevant details from the 'COMPREHENSIVE MEMORY SYNTHESIS'?
        2.  **RELEVANCE**: Is the response focused on the user's most recent query?
        3.  **HONESTY**: Is the AI honest about its limitations (e.g., no internet access)? Does it avoid making up information?
        
        Follow these steps:
        1.  **Critique**: Write a brief, one-sentence critique. If the draft is good, say "The draft is excellent and requires no changes."
        2.  **Revise**: If the critique found any issues, rewrite the response to fix them. If no changes are needed, simply repeat the original draft.

        Respond with ONLY a JSON object with two keys: "critique" and "final_response".
        
        Example for a good response:
        {{
            "critique": "The draft is excellent and requires no changes.",
            "final_response": "{initial_response_text}"
        }}
        
        Example for a response needing revision:
        {{
            "critique": "The draft failed to incorporate details from the memory synthesis about our previous conversation on this topic.",
            "final_response": "Based on our previous discussion where you mentioned liking sci-fi, I'd recommend..."
        }}
        """
        
        correction_response = await self.fast_llm.ainvoke([HumanMessage(content=correction_prompt)])
        correction_text = correction_response.content.strip()

        try:
            json_str = correction_text[correction_text.find('{'):correction_text.rfind('}') + 1]
            parsed_correction = json.loads(json_str)
            
            critique = parsed_correction.get("critique", "Critique could not be parsed.")
            final_response = parsed_correction.get("final_response", initial_response_text)
            
            state["critique"] = critique
            state["response"] = final_response
            
            log_thinking(f"--- Conscience Critique: {critique} ---")
            log_thinking(f"--- Final Response: {final_response} ---")

        except (json.JSONDecodeError, IndexError) as e:
            log_thinking(f"--- Self-Correction Error: Could not parse JSON. Using initial response. Error: {e} ---")
            state["critique"] = "Correction failed due to parsing error."
            state["response"] = initial_response_text

        return state

    async def _update_core_identity(self, state: AgentState) -> AgentState:
        """
        Checks if a core identity trait has been updated and saves it.
        """
        prompt = f"""
        Analyze the following interaction and determine if the AGENT's core identity trait
        has been established or changed.
        
        CRITICAL: Only update the agent's identity if the user is explicitly:
        1. Giving the AGENT a name (e.g., "I'll call you Kent" or "Your name is Kent")
        2. Establishing beliefs FOR THE AGENT (not the user's own beliefs)
        
        Do NOT update the agent's identity if:
        - The user is introducing themselves (e.g., "I'm Ken" or "My name is Ken")
        - The user is talking about their own beliefs or characteristics
        
        The current AGENT identity is: {state['core_identity']}
        The interaction was:
        User: "{state['user_input']}"
        Agent: "{state['response']}"

        We are tracking "name" and "beliefs_layer" (for referencing belief memory layers).
        If the user gave the AGENT a new name or established the AGENT's beliefs, respond with ONLY a JSON object with the updated
        identity. For example: {{"name": "Kent"}} or {{"beliefs_layer": "core_beliefs_on_truth_and_purpose"}}
        If no AGENT identity trait was changed, respond with the exact text "No change.".
        Do not include any other text or formatting.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision = response.content.strip()

        if "No change." not in decision:
            try:
                # Extract JSON from the response, assuming it might be wrapped in markdown
                if "```json" in decision:
                    json_str = decision.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = decision[decision.find('{'):decision.rfind('}') + 1]

                if not json_str:
                    raise json.JSONDecodeError("No JSON object found in response", decision, 0)

                updated_identity = json.loads(json_str)
                state["core_identity"].update(updated_identity)
                await asyncio.to_thread(config.save_core_identity, state["core_identity"])
                log_thinking(
                    "--- Core Identity Updated: "
                    f"{json.dumps(updated_identity)} ---"
                )
            except (json.JSONDecodeError, IndexError) as e:
                log_thinking(f"--- Core Identity: Invalid JSON from LLM --- \nLLM Response: '{decision}'\nError: {e}")

        return state

    def _build_response_context(self, state: AgentState) -> str:
        """Helper function to build the context string for response generation."""
        context = ""
        
        # Add working memory context
        if state.get("working_memory"):
            context += "\n--- 🧠 WORKING MEMORY (Current Conversation Facts) 🧠 ---\n"
            for key, value in state["working_memory"].items():
                context += f"{key}: {value}\n"
            context += "--- END WORKING MEMORY ---\n"

        if state["emotional_context"]:
            context += (
                "\n--- EMOTIONAL CONTEXT ---\n"
                f"User's emotional state: {state['emotional_context']}\n"
                "You should adapt your response style to this context.\n"
                "--- END EMOTIONAL CONTEXT ---\n"
            )

        context += "\n--- RECENT CONVERSATION SUMMARY ---\n"
        context += f"{state['short_term_summary']}\n"
        context += "--- END RECENT CONVERSATION SUMMARY ---\n"
        
        # Add the full history context if it was loaded
        if state.get("full_history_context"):
             context += f"\n{state['full_history_context']}\n"

        # Add recent session context if available and distinct
        if state.get("recent_sessions_context"):
            context += "\n--- 📝 MOST RECENT SESSION HIGHLIGHTS (GIVE THIS PRIORITY) 📝 ---\n"
            context += "This is the summary of the most recent, relevant conversation session.\n"
            context += f"{state['recent_sessions_context']}\n"
            context += "--- END RECENT SESSION HIGHLIGHTS ---\n"
            
        if state.get('retrieved_docs'):
            context += "\n--- 🔴 RELEVANT MEMORIES (SYNTHESIZE AND INCORPORATE THESE) 🔴 ---\n"
            context += "These are your memories from various sources. Synthesize them into a coherent understanding and incorporate relevant details into your response:\n\n"
            
            # Group by source for better organization
            memories_by_source = {}
            for doc in state['retrieved_docs']:
                source = doc.metadata.get('source', 'general')
                if source not in memories_by_source:
                    memories_by_source[source] = []
                memories_by_source[source].append(doc.page_content)
            
            for source, memories in memories_by_source.items():
                context += f"[{source}]:\n"
                for mem in memories:
                    context += f"• {mem}\n"
                context += "\n"
            
            context += "--- END OF MEMORIES ---\n"

        if state["inner_monologue"]:
            context += (
                f"\n--- INNER MONOLOGUE ---\n{state['inner_monologue']}\n"
                "--- END INNER MONOLOGUE ---\n"
            )

        # Self-awareness prompt
        context += (
            "\n--- SELF-AWARENESS CHECK ---\n"
            "Before you answer, take a moment to consider:\n"
            f"- Your name is {self.core_identity.get('name', 'not set')}.\n"
            f"- Your beliefs are retrieved from your beliefs layer when relevant.\n"
            "- Who are you? (A self-improving AI, version "
            f"{self.config['version']})\n"
            "- What are your capabilities and limitations? (Access to layered "
            "memory, ability to learn, but NO internet access, no web search, "
            "no ability to browse websites or fetch external information, "
            "and no true consciousness or feelings)\n"
            "- How does the user's query relate to your nature as an AI?\n"
            "--- END SELF-AWARENESS CHECK ---\n"
        )
        return context

    async def _update_memory(self, state: AgentState) -> AgentState:
        layer_name = state["active_layer"]
        if layer_name:
            layer_config = self.config["layers"][layer_name]
            vector_store = memory.get_vector_store(
                layer_config["vector_store_path"]
            )

            # Create a summary of the interaction to store in memory
            summary_prompt = f"""
            Create a concise summary of the key information from the following
            interaction. Focus on extracting any core facts or decisions made,
            especially if they relate to the agent's identity, user preferences,
            or established facts for future reference.

            User: "{state['user_input']}"
            Agent: "{state['response']}"

            For example, if the agent was given a name, the summary should be
            something like: "The user named the agent Kent."
            If the user expressed a preference, it could be: "The user likes to
            talk about space exploration."
            """
            summary_response = await self.llm.ainvoke(
                [HumanMessage(content=summary_prompt)]
            )
            summary = summary_response.content

            memory.add_to_memory(
                vector_store,
                summary,
                layer_config["vector_store_path"]
            )

        # Log the interaction to the experience layer
        log_entry = {
            "user_prompt": state["user_input"],
            "inner_monologue": state["inner_monologue"],
            "final_answer": state["response"],
            "critique_result": state["critique"],
            "successful": state["critique"] == "Positive",
        }
        memory.add_to_memory(
            self.experience_vector_store,
            json.dumps(log_entry),
            str(config.VECTOR_STORES_DIR / "experience_layer.faiss")
        )

        # Save the current turn to long-term memory
        turn_text = f"User: {state['user_input']}\nAgent: {state['response']}"
        memory.add_to_memory(
            self.long_term_memory,
            turn_text,
            str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss")
        )

        # Update semantic cache
        state["semantic_cache"][state["user_input"]] = state["response"]

        return state

    async def _ensure_initialized(self):
        """Ensure the async components are initialized."""
        if self.checkpointer is None:
            self.conn = await aiosqlite.connect("langgraph_cache.db")
            self.checkpointer = AsyncSqliteSaver(self.conn)
            self.graph = self._build_graph()

    async def arun(self, user_input: str, session_id: str = "default_session"):
        """
        Runs the agent.
        """
        await self._ensure_initialized()
        
        # Initialize session if needed
        if self.session_id != session_id:
            await self._start_new_session(session_id)
        
        # Triage for simple greetings to provide a quick response
        greeting_match = self.simple_greetings.match(user_input)
        if greeting_match and len(self.current_session_history) < 4:  # Allow greetings in early conversation
            # Check if the greeting includes the agent's name
            if greeting_match.group(2) and self.core_identity.get('name'):
                agent_name = self.core_identity.get('name', 'there')
                if greeting_match.group(2).strip().lower() == agent_name.lower():
                    response = f"Hey there! Yes, I'm {agent_name}. How can I help you today?"
                else:
                    response = "Hey there! How can I help you today?"
            else:
                response = "Hey there! How can I help you today?"
            
            self.current_session_history.append({"role": "user", "content": user_input})
            self.current_session_history.append({"role": "agent", "content": response})
            return response
        
        initial_state = {
            "user_input": user_input,
            "conversation_history": self.current_session_history,  # Only current session
            "short_term_summary": self.short_term_summary,

            "semantic_cache": {},
            "layer_info": {},
            "active_layer": None,
            "new_layer_description": None,
            "needs_new_layer": False,
            "response": "",
            "inner_monologue": None,
            "critique": None,
            "past_experiences": None,
            "core_identity": self.core_identity,
            "emotional_context": None,
            "working_memory": self.working_memory.copy(),  # Pass current working memory
            "retrieved_docs": [],
            "drill_down_needed": False,
            "archive_to_load": None,
            "full_history_context": None,
            "revision_count": 0, # This is no longer used but kept for state compatibility
            "layer_creation_attempts": 0,
            "recent_sessions_context": None # Initialize for new session
        }
        final_state = await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        # Update current session history only
        self.current_session_history.append(
            {"role": "user", "content": user_input}
        )
        self.current_session_history.append(
            {"role": "agent", "content": final_state["response"]}
        )

        # Update the instance's short_term_summary, core identity, and working memory
        self.short_term_summary = final_state["short_term_summary"]
        self.core_identity = final_state["core_identity"]
        self.working_memory = final_state.get("working_memory", {})
        
        log_thinking(f"--- Session Working Memory: {self.working_memory} ---")

        return final_state["response"]

    async def astream(self, user_input: str, session_id: str = "default_session"):
        """
        Streams the agent's processing in real-time.
        """
        await self._ensure_initialized()
        
        # Initialize session if needed
        if self.session_id != session_id:
            await self._start_new_session(session_id)
        
        # Triage for simple greetings to provide a quick response
        greeting_match = self.simple_greetings.match(user_input)
        if greeting_match and len(self.current_session_history) < 4:  # Allow greetings in early conversation
            # Check if the greeting includes the agent's name
            if greeting_match.group(2) and self.core_identity.get('name'):
                agent_name = self.core_identity.get('name', 'there')
                if greeting_match.group(2).strip().lower() == agent_name.lower():
                    response = f"Hey there! Yes, I'm {agent_name}. How can I help you today?"
                else:
                    response = "Hey there! How can I help you today?"
            else:
                response = "Hey there! How can I help you today?"
            
            self.current_session_history.append({"role": "user", "content": user_input})
            self.current_session_history.append({"role": "agent", "content": response})
            yield {"type": "response", "content": response}
            return

        initial_state = {
            "user_input": user_input,
            "conversation_history": self.current_session_history,  # Only current session
            "short_term_summary": self.short_term_summary,

            "semantic_cache": {},
            "layer_info": {},
            "active_layer": None,
            "new_layer_description": None,
            "needs_new_layer": False,
            "response": "",
            "inner_monologue": None,
            "critique": None,
            "past_experiences": None,
            "core_identity": self.core_identity,
            "emotional_context": None,
            "working_memory": self.working_memory.copy(),  # Pass current working memory
            "retrieved_docs": [],
            "drill_down_needed": False,
            "archive_to_load": None,
            "full_history_context": None,
            "revision_count": 0, # This is no longer used but kept for state compatibility
            "layer_creation_attempts": 0,
            "recent_sessions_context": None # Initialize for new session
        }

        # Stream the graph execution
        final_state = None
        async for chunk in self.graph.astream(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        ):
            # Yield intermediate progress
            node_name = list(chunk.keys())[0] if chunk else "unknown"
            node_data = chunk.get(node_name, {}) if chunk else {}
            
            # Emit thinking steps for specific nodes
            if node_name == "initial_analysis_and_routing":
                yield {"type": "thinking", "content": "Analyzing your message and routing to appropriate memory..."}
            elif node_name == "parallel_memory_retrieval":
                yield {"type": "thinking", "content": "Retrieving relevant memories from multiple layers..."}
            elif node_name == "synthesize_memories":
                yield {"type": "thinking", "content": "Synthesizing retrieved memories..."}
            elif node_name == "generate_and_correct_response":
                yield {"type": "thinking", "content": "Generating response and performing self-correction..."}
            elif node_name == "propose_new_layer":
                yield {"type": "thinking", "content": "Creating new memory layer for this topic..."}
            elif node_name == "create_new_layer":
                layer_name = node_data.get("active_layer", "unknown")
                yield {"type": "thinking", "content": f"Initializing new memory layer: {layer_name}"}
                
            final_state = node_data

        if final_state and final_state.get("response"):
            # Update current session history only
            self.current_session_history.append(
                {"role": "user", "content": user_input}
            )
            self.current_session_history.append(
                {"role": "agent", "content": final_state["response"]}
            )

            # Update the instance's short_term_summary, core identity, and working memory
            self.short_term_summary = final_state.get("short_term_summary", self.short_term_summary)
            self.core_identity = final_state.get("core_identity", self.core_identity)
            self.working_memory = final_state.get("working_memory", {})
            
            log_thinking(f"--- Session Working Memory: {self.working_memory} ---")

            yield {"type": "response", "content": final_state["response"]}
        else:
            yield {"type": "error", "content": "Failed to generate response"}

    async def _start_new_session(self, session_id: str):
        """
        Starts a new conversation session, summarizing the previous one if it exists.
        """
        # Summarize and save previous session if it had content
        if self.current_session_history and len(self.current_session_history) > 2:
            await self._summarize_and_save_session()
        
        # Reset for new session
        self.session_id = session_id
        self.session_start_time = asyncio.get_event_loop().time()
        self.current_session_history = []
        self.working_memory = {}
        self.short_term_summary = ""
        
        log_thinking(f"--- Started New Session: {session_id} ---")
    
    async def _summarize_and_save_session(self):
        """
        Creates a comprehensive summary of the current session, saves the full
        history to an archive, and links the two in the session summary vector store.
        """
        if not self.current_session_history:
            return
            
        # 1. Create a dedicated directory for archives if it doesn't exist
        archive_dir = config.VECTOR_STORES_DIR.parent / "conversation_archives"
        archive_dir.mkdir(exist_ok=True)
        
        # 2. Save the full conversation history to a unique archive file
        timestamp = int(asyncio.get_event_loop().time())
        archive_path = archive_dir / f"session_{self.session_id}_{timestamp}.json"
        await asyncio.to_thread(
            lambda: json.dump(self.current_session_history, open(archive_path, 'w'), indent=2)
        )
        
        log_thinking(f"--- Full session history saved to {archive_path} ---")

        # 3. Create conversation text for summarization
        conversation_text = ""
        for msg in self.current_session_history:
            conversation_text += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Generate comprehensive summary
        summary_prompt = f"""
        Create a comprehensive summary of this conversation session.
        
        Conversation:
        {conversation_text}
        
        Working Memory Facts: {json.dumps(self.working_memory)}
        
        Create a detailed summary that includes:
        1. WHO was involved (names, roles)
        2. WHAT was discussed (main topics, questions asked)
        3. KEY INFORMATION shared (facts, preferences, decisions)
        4. OUTCOMES or conclusions reached
        5. Any ACTION ITEMS or follow-ups mentioned
        
        Format this as a coherent narrative that preserves all important details.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        session_summary = response.content.strip()
        
        # 5. Save summary to vector store with metadata linking to the archive
        summary_metadata = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "end_time": timestamp,
            "archive_path": str(archive_path),
            "creation_timestamp": datetime.now().isoformat()
        }
        
        memory.add_to_memory(
            self.session_summaries,
            session_summary, # We only embed the summary text
            str(config.VECTOR_STORES_DIR / "session_summaries.faiss"),
            metadata=summary_metadata
        )
        
        log_thinking(f"--- Session Summary saved with link to archive: {archive_path} ---")

    async def aclose(self):
        """Closes the database connection and saves current session."""
        # Summarize current session before closing
        if self.current_session_history:
            await self._summarize_and_save_session()
            
        if self.conn:
            await self.conn.close()
