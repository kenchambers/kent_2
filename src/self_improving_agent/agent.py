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
from . import user_profile
from .utils import create_layer_name, log_thinking


class AgentState(TypedDict):
    """
    Represents the state of the agent at any given time.
    """
    session_id: str
    user_id: str  # Add a persistent user_id
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
    user_profile: Dict[str, Any]  # User profile data including name, facts, vector store path
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
        
        # Session management - now keyed by session_id for proper isolation
        self.session_histories: Dict[str, List[Dict[str, str]]] = {}
        self.session_start_times: Dict[str, float] = {}
        
        # Memory management - now keyed by session_id
        self.session_short_summaries: Dict[str, str] = {}
        self.session_working_memories: Dict[str, Dict[str, Any]] = {}
        self.session_user_profiles: Dict[str, Dict[str, Any]] = {}  # Cache user profiles per session
        self.conversation_window_size = 10  # Only keep last N exchanges in context
        
        # Add a lock for thread-safe session initialization
        self._session_init_lock = asyncio.Lock()

        # Vector stores
        self.experience_vector_store = memory.get_experience_vector_store()
        self.long_term_memory = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss")
        )
        self.session_summaries = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "session_summaries.faiss")
        )
        self.shared_experiences = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "shared_experiences.faiss")
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

    async def _analyze_and_retrieve_concurrently(self, state: AgentState) -> AgentState:
        """
        Runs emotional analysis, routing, and memory retrieval in parallel to reduce latency.
        """
        log_thinking("--- Concurrently Analyzing and Retrieving Memories ---")
        user_input = state["user_input"]

        # 1. Define the analysis and routing task
        async def analyze_and_route():
            past_experiences = memory.query_memory(
                self.experience_vector_store,
                user_input
            )
            relevant_layers = await self._get_relevant_layers(user_input, k=20)
            
            prompt = f"""
            Analyze the user's input and determine the next action.

            User input: "{user_input}"
            Recent conversation summary: "{state['short_term_summary']}"
            Relevant memory layers: {json.dumps(list(relevant_layers.keys()))}
            Relevant past experiences: {past_experiences}

            Respond with a JSON object with two keys:
            1. "emotional_context": A concise, one-sentence summary of the user's emotional and stylistic state.
            2. "routing_decision": An object with "thought", "action" (layer name, "new_layer", or "none"), "confidence", and "self_correction_plan".

            ROUTING GUIDELINES:
            - Use "new_layer" for specific, novel topics.
            - Always check belief layers for philosophical questions.
            - Use "general_knowledge" for simple greetings.
            - When in doubt, prefer creating a new layer.
            """
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            try:
                json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
                parsed_json = json.loads(json_str)
                return parsed_json, past_experiences
            except (json.JSONDecodeError, KeyError) as e:
                log_thinking(f"--- Analysis/Routing Error: {e} ---")
                return None, past_experiences

        # 2. Define memory retrieval tasks
        active_layer = state.get("active_layer")
        memory_tasks = {
            "beliefs": self._get_core_beliefs(user_input),
            "sessions": self._get_session_summaries(user_input),
            "dynamic": self._get_dynamic_memory(state),
            "experiences": self._get_shared_experiences(state),
            "user_profile": self._get_user_profile_info(state),
        }
        if active_layer:
            memory_tasks["layer_info"] = self._get_existing_layer_info(active_layer, user_input)

        # 3. Run analysis and memory tasks concurrently
        analysis_task = analyze_and_route()
        all_tasks = { "analysis": analysis_task, **memory_tasks }
        
        results = await asyncio.gather(*all_tasks.values(), return_exceptions=True)
        results_map = dict(zip(all_tasks.keys(), results))

        # 4. Process analysis results
        analysis_result, past_experiences = results_map.pop("analysis", (None, None))
        state["past_experiences"] = past_experiences
        if analysis_result:
            state["emotional_context"] = analysis_result.get("emotional_context")
            routing_decision = analysis_result.get("routing_decision", {})
            state["inner_monologue"] = json.dumps(routing_decision, indent=2)
            action = routing_decision.get("action", "new_layer")
            
            if action == "new_layer":
                state["needs_new_layer"] = True
            elif action in ["none", "no_action_needed"]:
                state["needs_new_layer"] = False
                state["active_layer"] = None
            else:
                state["needs_new_layer"] = False
                state["active_layer"] = action
            log_thinking(f"--- Routing Decision: {action} ---")
        else:
            state["needs_new_layer"] = True # Fallback

        # 5. Process memory retrieval results with session-specific filtering
        all_docs = []
        session_id = state["session_id"]
        
        # Process session summaries (session-specific)
        if 'sessions' in results_map and isinstance(results_map['sessions'], list):
            session_docs = [
                doc for doc in results_map.pop('sessions')
                if doc.metadata.get('session_id') == session_id
            ]
            log_thinking(f"--- Retrieved and filtered {len(session_docs)} session summaries for session {session_id} ---")
            
            if session_docs:
                log_thinking("--- Re-ranking session summaries by recency ---")
                sorted_sessions = sorted(
                    session_docs,
                    key=lambda doc: datetime.fromisoformat(ts) if (ts := doc.metadata.get('creation_timestamp')) else datetime.min,
                    reverse=True
                )
                recent_sessions = sorted_sessions[:3]
                state['recent_sessions_context'] = "\n".join([doc.page_content for doc in recent_sessions])
                all_docs.extend(sorted_sessions)

        # Process long-term memory (session-specific)
        if 'dynamic' in results_map and isinstance(results_map['dynamic'], list):
            dynamic_docs = [
                doc for doc in results_map.pop('dynamic')
                if doc.metadata.get("session_id") == session_id
            ]
            log_thinking(f"--- Retrieved and filtered {len(dynamic_docs)} long-term memories for session {session_id} ---")
            all_docs.extend(dynamic_docs)
        
        # Process user profile info (user-specific)
        user_id = state["user_id"]
        if 'user_profile' in results_map and isinstance(results_map['user_profile'], list):
            user_docs = [
                doc for doc in results_map.pop('user_profile')
                if doc.metadata.get("user_id") == user_id
            ]
            log_thinking(f"--- Retrieved and filtered {len(user_docs)} documents for user {user_id} ---")
            all_docs.extend(user_docs)

        # Process remaining global memories
        for task_name, result in results_map.items():
            if isinstance(result, Exception):
                log_thinking(f"--- Error during {task_name} retrieval: {result} ---")
            elif result and isinstance(result, list):
                all_docs.extend(result)
                log_thinking(f"--- Retrieved {len(result)} docs from global memory: {task_name} ---")
        
        state['retrieved_docs'] = all_docs
        return state



    async def _get_dynamic_memory(self, state: AgentState) -> List[Document]:
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

    async def _get_shared_experiences(self, state: AgentState) -> List[Document]:
        """Helper to retrieve from the shared experiences vector store using a broader context."""
        log_thinking("--- Querying Shared Experiences & Wisdom ---")
        user_input = state["user_input"]
        working_memory = state.get("working_memory", {})
        topic = working_memory.get("topic")

        # Create a broader query that includes the general topic
        if topic:
            query = f"{user_input} (related to the topic of {topic})"
        else:
            query = user_input

        log_thinking(f"--- Shared experiences query: '{query}' ---")
        try:
            return await self.shared_experiences.asimilarity_search(query, k=3)
        except Exception as e:
            log_thinking(f"Error querying shared experiences: {e}")
            return []

    async def _get_user_profile_info(self, state: AgentState) -> List[Document]:
        """Helper to retrieve from the user's profile vector store."""
        log_thinking("--- Querying User Profile Information ---")
        user_input = state["user_input"]
        user_profile_data = state.get("user_profile", {}) # Renamed to avoid confusion
        
        if not user_profile_data or not user_profile_data.get("vector_store_path"):
            log_thinking("--- No user profile vector store available ---")
            return []
        
        try:
            # Add logging for which user's store is being queried
            user_id = user_profile_data.get("id")
            log_thinking(f"--- Accessing vector store for user_id: {user_id} at path {user_profile_data['vector_store_path']} ---")
            user_vector_store = memory.get_vector_store(user_profile_data["vector_store_path"])
            retrieved_info = memory.query_memory(
                user_vector_store, user_input, k=3, user_id=user_id
            )
            
            # Defensive check is now redundant due to filtering in query_memory, but kept for logging
            # (The list comprehension is removed as it is now handled in the query_memory function)
            log_thinking(f"--- Retrieved {len(retrieved_info) if retrieved_info else 0} user profile docs for user {user_id} ---")
            return retrieved_info or []
        except Exception as e:
            log_thinking(f"Error querying user profile for user {user_profile_data.get('id')}: {e}")
            return []

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

    def _build_graph(self):
        """
        Builds the agent's graph.
        """
        workflow = StateGraph(AgentState)

        # New concurrent node
        workflow.add_node("_analyze_and_retrieve_concurrently", self._analyze_and_retrieve_concurrently)
        
        workflow.add_node("propose_new_layer", self._propose_new_layer)
        workflow.add_node("create_new_layer", self._create_new_layer)
        
        # New single-step response generation node
        workflow.add_node("generate_response_with_self_critique", self._generate_response_with_self_critique)
        
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("_update_short_term_summary", self._update_short_term_summary)
        workflow.add_node("query_semantic_cache", self._query_semantic_cache)
        workflow.add_node("update_core_identity", self._update_core_identity)
        workflow.add_node("update_working_memory", self._update_working_memory)
        workflow.add_node("update_user_profile", self._update_user_profile)
        workflow.add_node("decide_drill_down", self._decide_drill_down)
        workflow.add_node("load_full_history", self._load_full_history)
        workflow.add_node("verify_layer_creation", self._verify_layer_creation)

        workflow.set_entry_point("update_working_memory")
        workflow.add_edge("update_working_memory", "update_user_profile")
        workflow.add_edge("update_user_profile", "_analyze_and_retrieve_concurrently")

        workflow.add_conditional_edges(
            "_analyze_and_retrieve_concurrently",
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
            lambda state: "load_full_history" if state.get("drill_down_needed") else "generate_response_with_self_critique"
        )
        workflow.add_edge("load_full_history", "generate_response_with_self_critique")

        workflow.add_edge("generate_response_with_self_critique", "update_core_identity")
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

    async def _update_user_profile(self, state: AgentState) -> AgentState:
        """
        Updates the user profile with new information from the conversation.
        Detects user name introductions and updates user facts.
        """
        log_thinking("--- Updating User Profile ---")
        
        session_id = state["session_id"]
        user_id = state["user_id"] # Get the persistent user_id from state
        
        log_thinking(f"--- Loading profile for user_id: {user_id} in session_id: {session_id} ---")
        current_profile = await user_profile.get_user_profile(user_id) # Use user_id
        state["user_profile"] = current_profile
        # Cache the profile for conversation formatting, keyed by session_id for the current session
        self.session_user_profiles[session_id] = current_profile
        log_thinking(f"--- Loaded profile for user: {current_profile.get('name')} ---")
        
        # Check if user introduced themselves or shared new information
        profile_update_prompt = f"""
        Analyze this user input to detect if the user is introducing themselves or sharing personal information that should be saved to their profile.
        
        User input: "{state['user_input']}"
        Current user profile: {current_profile}
        Working memory: {state.get('working_memory', {})}
        
        Look for:
        - User introducing themselves with their name (e.g., "I'm Ken", "My name is Sarah", "Call me Alex")
        - Personal facts about the user (preferences, background, etc.)
        
        IMPORTANT: Only update the user's profile, not the agent's profile.
        
        Respond with a JSON object with any updates to make to the user profile:
        - "name": if the user introduced themselves
        - "facts": a dictionary of personal facts about the user
        
        If no updates needed, respond with: {{"no_updates": true}}
        
        Example:
        {{"name": "Ken", "facts": {{"profession": "software engineer", "location": "California"}}}}
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=profile_update_prompt)])
        response_text = response.content.strip()
        
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
            
            profile_updates = json.loads(json_str)
            
            if not profile_updates.get("no_updates", False):
                profile_changed = False
                
                # Update name if provided
                if "name" in profile_updates and profile_updates["name"] != current_profile["name"]:
                    current_profile["name"] = profile_updates["name"]
                    profile_changed = True
                    log_thinking(f"--- User name updated to: {profile_updates['name']} ---")
                
                # Update facts if provided
                if "facts" in profile_updates:
                    current_profile["facts"].update(profile_updates["facts"])
                    profile_changed = True
                    log_thinking(f"--- User facts updated: {profile_updates['facts']} ---")
                
                # Save profile if changed
                if profile_changed:
                    await user_profile.save_user_profile(user_id, current_profile) # Use user_id
                    
                    # Also save to user's vector store for future retrieval
                    user_vector_store = memory.get_vector_store(current_profile["vector_store_path"])
                    user_info_summary = f"User profile: Name is {current_profile['name']}. Facts: {current_profile['facts']}"
                    memory.add_to_memory(
                        user_vector_store,
                        user_info_summary,
                        current_profile["vector_store_path"],
                        metadata={"source": "user_profile", "user_id": user_id} # Use user_id
                    )
                    
                    state["user_profile"] = current_profile
                    # Cache the profile for conversation formatting
                    self.session_user_profiles[session_id] = current_profile
        
        except (json.JSONDecodeError, IndexError) as e:
            log_thinking(f"--- User Profile: Could not extract updates --- Error: {e}")
        
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

        # Add the immediate last exchange to prevent repetition
        if state.get("conversation_history") and len(state["conversation_history"]) > 1:
            last_agent_message = state["conversation_history"][-1]
            last_user_message = state["conversation_history"][-2]
            context += "\n--- PREVIOUS EXCHANGE (FOR CONTEXT) ---\n"
            context += f"User: \"{last_user_message['content']}\"\n"
            context += f"You (Agent): \"{last_agent_message['content']}\"\n"
            context += "--- END PREVIOUS EXCHANGE ---\n"
        
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
            shared_wisdom = []
            for doc in state['retrieved_docs']:
                source = doc.metadata.get('source_type', doc.metadata.get('source', 'general'))
                if source == "shared_experience":
                    shared_wisdom.append(doc)
                    continue

                if source not in memories_by_source:
                    memories_by_source[source] = []
                memories_by_source[source].append(doc.page_content)
            
            for source, memories in memories_by_source.items():
                context += f"[{source}]:\n"
                for mem in memories:
                    context += f"• {mem}\n"
                context += "\n"
            
            context += "--- END OF MEMORIES ---\n"

            # Add shared wisdom with a strict privacy directive
            if shared_wisdom:
                context += "\n--- 🏛️ SHARED WISDOM FROM PAST CONVERSATIONS (PRIVACY DIRECTIVE) 🏛️ ---\n"
                context += "**PRIVACY DIRECTIVE: CRITICAL**\n"
                context += "You may use the following lessons, learned from conversations with other users, to provide deeper insight. When you use one, you MUST adhere to these rules:\n"
                context += "1.  **NEVER** share any personal details about the other user beyond their first name.\n"
                context += "2.  **ALWAYS** frame it as a lesson learned from a friend. Use phrases like 'A friend of mine, [First Name], once taught me that...' or 'I learned from my friend [First Name] that...'\n"
                context += "3.  The focus is on the **wisdom**, not the person. Do not elaborate on the person or the context of the original conversation.\n"
                context += "Failure to follow this directive is a critical breach of user trust.\n\n"
                context += "Available Wisdom:\n"
                for wisdom_doc in shared_wisdom:
                    context += f"• {wisdom_doc.page_content}\n"
                context += "--- END OF SHARED WISDOM ---\n"

        if state["inner_monologue"]:
            context += (
                f"\n--- INNER MONOLOGUE ---\n{state['inner_monologue']}\n"
                "--- END INNER MONOLOGUE ---\n"
            )

        # Add user profile information if available
        if state.get("user_profile"):
            user_profile = state["user_profile"]
            context += f"\n--- USER PROFILE ---\n"
            context += f"User's name: {user_profile.get('name', 'User')}\n"
            if user_profile.get('facts'):
                context += f"User facts: {user_profile['facts']}\n"
            context += "--- END USER PROFILE ---\n"

        # Conversation format instructions
        context += (
            "\n--- CONVERSATION FORMAT INSTRUCTIONS ---\n"
            "IMPORTANT: When you see conversation history, it will be formatted with name tags:\n"
            "- [User: Name] indicates what the user said\n"
            "- [Agent: Name] indicates what you (the agent) said in previous exchanges\n"
            "When referring to previous conversations, understand that 'Agent' refers to you.\n"
            "--- END CONVERSATION FORMAT INSTRUCTIONS ---\n"
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
            "memory, ability to learn from all conversations to build collective wisdom, but NO internet access, no web search, "
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
        user_name = self._get_user_name_for_session(state["session_id"])
        agent_name = self.core_identity.get('name', 'Agent')
        turn_text = (
            f"[User: {user_name}] {state['user_input']}\n"
            f"[Agent: {agent_name}] {state['response']}"
        )
        memory.add_to_memory(
            self.long_term_memory,
            turn_text,
            str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss"),
            metadata={"session_id": state["session_id"], "user_id": state["user_id"]} # Add user_id to metadata
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

    async def arun(self, user_input: str, session_id: str = "default_session", user_id: str = "default_user"):
        """
        Runs the agent.
        """
        await self._ensure_initialized()
        
        # Initialize session if needed
        await self._initialize_session_if_needed(session_id, user_id)
        
        # Triage for simple greetings to provide a quick response
        greeting_match = self.simple_greetings.match(user_input)
        if greeting_match and len(self.session_histories.get(session_id, [])) < 4:  # Allow greetings in early conversation
            # Check if the greeting includes the agent's name
            if greeting_match.group(2) and self.core_identity.get('name'):
                agent_name = self.core_identity.get('name', 'there')
                if greeting_match.group(2).strip().lower() == agent_name.lower():
                    response = f"Hey there! Yes, I'm {agent_name}. How can I help you today?"
                else:
                    response = "Hey there! How can I help you today?"
            
            if session_id not in self.session_histories:
                self.session_histories[session_id] = []
            self.session_histories[session_id].append(self._format_conversation_entry("user", user_input, session_id))
            self.session_histories[session_id].append(self._format_conversation_entry("agent", response, session_id))
            return response
        
        # Keep the conversation history within the window size
        current_history = self.session_histories.get(session_id, [])
        history_window = current_history[
            -self.conversation_window_size*2:
        ] if self.conversation_window_size > 0 else current_history

        initial_state = {
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "conversation_history": history_window,  # Pass the windowed history
            "short_term_summary": self.session_short_summaries.get(session_id, ""),

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
            "working_memory": self.session_working_memories.get(session_id, {}).copy(),  # Pass session working memory
            "user_profile": {},  # Will be loaded in update_user_profile node
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

        # Update session-specific history and state
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        self.session_histories[session_id].append(
            self._format_conversation_entry("user", user_input, session_id)
        )
        self.session_histories[session_id].append(
            self._format_conversation_entry("agent", final_state["response"], session_id)
        )

        # Update the session-specific state
        self.session_short_summaries[session_id] = final_state["short_term_summary"]
        # DO NOT update self.core_identity here - it causes race conditions between concurrent sessions
        # Each session should work with its own copy of core_identity passed in the initial_state
        self.session_working_memories[session_id] = final_state.get("working_memory", {})
        
        log_thinking(f"--- Session {session_id} Working Memory: {self.session_working_memories[session_id]} ---")

        return final_state["response"]

    async def astream(self, user_input: str, session_id: str = "default_session", user_id: str = "default_user"):
        """
        Streams the agent's processing in real-time.
        """
        await self._ensure_initialized()
        
        # Initialize session if needed
        await self._initialize_session_if_needed(session_id, user_id)
        
        # Triage for simple greetings to provide a quick response
        greeting_match = self.simple_greetings.match(user_input)
        if greeting_match and len(self.session_histories.get(session_id, [])) < 4:  # Allow greetings in early conversation
            # Check if the greeting includes the agent's name
            if greeting_match.group(2) and self.core_identity.get('name'):
                agent_name = self.core_identity.get('name', 'there')
                if greeting_match.group(2).strip().lower() == agent_name.lower():
                    response = f"Hey there! Yes, I'm {agent_name}. How can I help you today?"
                else:
                    response = "Hey there! How can I help you today?"
            else:
                response = "Hey there! How can I help you today?"
            
            if session_id not in self.session_histories:
                self.session_histories[session_id] = []
            self.session_histories[session_id].append(self._format_conversation_entry("user", user_input, session_id))
            self.session_histories[session_id].append(self._format_conversation_entry("agent", response, session_id))
            yield {"type": "response", "content": response}
            return

        # Keep the conversation history within the window size
        current_history = self.session_histories.get(session_id, [])
        history_window = current_history[
            -self.conversation_window_size*2:
        ] if self.conversation_window_size > 0 else current_history

        initial_state = {
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "conversation_history": history_window, # Pass the windowed history
            "short_term_summary": self.session_short_summaries.get(session_id, ""),

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
            "working_memory": self.session_working_memories.get(session_id, {}).copy(),  # Pass session working memory
            "user_profile": {},  # Will be loaded in update_user_profile node
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
            if node_name == "_analyze_and_retrieve_concurrently":
                yield {"type": "thinking", "content": "Analyzing your message and retrieving relevant memories..."}
            elif node_name == "generate_response_with_self_critique":
                yield {"type": "thinking", "content": "Generating response with integrated self-correction..."}
            elif node_name == "propose_new_layer":
                yield {"type": "thinking", "content": "Creating new memory layer for this topic..."}
            elif node_name == "create_new_layer":
                layer_name = node_data.get("active_layer", "unknown")
                yield {"type": "thinking", "content": f"Initializing new memory layer: {layer_name}"}
                
            final_state = node_data

        if final_state and final_state.get("response"):
            # Update session-specific history
            if session_id not in self.session_histories:
                self.session_histories[session_id] = []
            self.session_histories[session_id].append(
                {"role": "user", "content": user_input}
            )
            self.session_histories[session_id].append(
                {"role": "agent", "content": final_state["response"]}
            )

            # Update the session-specific state
            self.session_short_summaries[session_id] = final_state.get("short_term_summary", self.session_short_summaries.get(session_id, ""))
            # DO NOT update self.core_identity here - it causes race conditions between concurrent sessions  
            self.session_working_memories[session_id] = final_state.get("working_memory", {})
            
            log_thinking(f"--- Session {session_id} Working Memory: {self.session_working_memories[session_id]} ---")

            yield {"type": "response", "content": final_state["response"]}
        else:
            yield {"type": "error", "content": "Failed to generate response"}

    def _get_user_name_for_session(self, session_id: str) -> str:
        """Get the user name for a session, defaulting to 'User' if not set."""
        if session_id in self.session_user_profiles:
            return self.session_user_profiles[session_id].get('name', 'User')
        return 'User'

    def _format_conversation_entry(self, role: str, content: str, session_id: str) -> Dict[str, str]:
        """
        Formats a conversation entry with proper name tags like [User: Ken] and [Agent: Kent].
        """
        user_name = self._get_user_name_for_session(session_id)
        agent_name = self.core_identity.get('name', 'Agent')
        
        name = user_name if role == "user" else agent_name
        tagged_content = f"[{role.capitalize()}: {name}] {content}"
        
        return {"role": role, "content": tagged_content}

    async def _initialize_session_if_needed(self, session_id: str, user_id: str):
        """
        Initializes a new session's state dictionaries if it's the first time
        seeing this session_id. This is thread-safe.
        """
        if session_id not in self.session_histories:
            async with self._session_init_lock:
                # Double-check after acquiring the lock to handle race conditions
                if session_id not in self.session_histories:
                    self.session_start_times[session_id] = asyncio.get_event_loop().time()
                    self.session_histories[session_id] = []
                    self.session_working_memories[session_id] = {}
                    self.session_short_summaries[session_id] = ""
                    # Load the persistent user profile and cache it for this session
                    self.session_user_profiles[session_id] = await user_profile.get_user_profile(user_id)
                    log_thinking(f"--- Initialized New Session: {session_id} for User: {user_id} ---")
    
    async def _summarize_and_save_session(self, session_id: str = None):
        """
        Creates a comprehensive summary of the specified session, saves the full
        history to an archive, and links the two in the session summary vector store.
        """
        # Use current session if no session_id provided (for backward compatibility)
        target_session_id = session_id
        if not target_session_id or target_session_id not in self.session_histories:
            return
            
        session_history = self.session_histories[target_session_id]
        if not session_history:
            return
            
        # 1. Create a dedicated directory for archives if it doesn't exist
        archive_dir = config.VECTOR_STORES_DIR.parent / "conversation_archives"
        archive_dir.mkdir(exist_ok=True)
        
        # 2. Save the full conversation history to a unique archive file
        timestamp = int(asyncio.get_event_loop().time())
        archive_path = archive_dir / f"session_{target_session_id}_{timestamp}.json"
        await asyncio.to_thread(
            lambda: json.dump(session_history, open(archive_path, 'w'), indent=2)
        )
        
        log_thinking(f"--- Full session history saved to {archive_path} ---")

        # 3. Create conversation text for summarization
        conversation_text = ""
        for msg in session_history:
            # The history is already formatted, so we just join the content
            conversation_text += f"{msg['content']}\n"
        
        # Get session-specific working memory
        session_working_memory = self.session_working_memories.get(target_session_id, {})
        
        # Generate comprehensive summary
        summary_prompt = f"""
        Create a comprehensive summary of this conversation session.
        
        Conversation:
        {conversation_text}
        
        Working Memory Facts: {json.dumps(session_working_memory)}
        
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
        session_start_time = self.session_start_times.get(target_session_id, timestamp)
        user_id = self.session_user_profiles.get(target_session_id, {}).get('id', 'unknown_user')
        summary_metadata = {
            "session_id": target_session_id,
            "user_id": user_id,
            "start_time": session_start_time,
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
        """Closes the database connection and saves all active sessions."""
        # Summarize all active sessions before closing
        for session_id, history in self.session_histories.items():
            if history and len(history) > 2:
                await self._summarize_and_save_session(session_id)
            
        if self.conn:
            await self.conn.close()

    async def _generate_response_with_self_critique(self, state: AgentState) -> AgentState:
        """
        Generates a response and performs a single-pass self-correction critique
        within the same LLM call to reduce latency.
        """
        log_thinking("--- Generating Response with Integrated Self-Critique ---")

        context = self._build_response_context(state)

        # A single, powerful prompt that asks for generation, critique, and revision in one go.
        prompt = f"""
        You are Kent, a thoughtful and empathetic AI. Your goal is to provide a human-like response.
        
        **CONTEXT:**
        {context}
        
        **USER'S MESSAGE:**
        "{state['user_input']}"

        **TASK:**
        Follow this thought process:
        1.  **Initial Response:** First, draft a direct and helpful response to the user's message, incorporating relevant memories and context naturally.
        2.  **Self-Critique:** Then, critically evaluate your own draft. Does it sound like Kent? Is it warm, empathetic, and not robotic? Does it properly use the provided memories? Is it honest about your limitations?
        3.  **Final Response:** Based on your critique, revise the draft into a final, polished response. If the initial draft was already excellent, you can use it as the final response.

        **OUTPUT FORMAT:**
        Respond with ONLY a JSON object with three keys:
        - "initial_response": Your first draft.
        - "self_critique": Your brief, honest critique of the draft.
        - "final_response": The polished, final response to send to the user.

        Example:
        {{
            "initial_response": "The capital of France is Paris.",
            "self_critique": "This is too robotic and factual. It doesn't match Kent's warm persona.",
            "final_response": "Ah, Paris! It's the capital of France. I've always been fascinated by the descriptions of the city I've come across. Have you ever been?"
        }}
        """

        messages = [
            SystemMessage(content="You are a helpful AI assistant that provides a response, a self-critique, and a final revised response in a single JSON object."),
            HumanMessage(content=prompt)
        ]

        # Use the main LLM for this complex, multi-step reasoning task
        response = await self.llm.ainvoke(messages)
        response_text = response.content.strip()

        try:
            json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
            parsed_json = json.loads(json_str)
            
            critique = parsed_json.get("self_critique", "Critique could not be parsed.")
            final_response = parsed_json.get("final_response", parsed_json.get("initial_response", "I'm not sure how to respond to that."))
            
            state["critique"] = critique
            state["response"] = final_response
            
            log_thinking(f"--- Self-Critique: {critique} ---")
            log_thinking(f"--- Final Response: {final_response} ---")

        except (json.JSONDecodeError, KeyError) as e:
            log_thinking(f"--- Self-Correction Error: Could not parse JSON. Error: {e} ---")
            # Fallback to a simpler generation if the structured output fails
            fallback_response = await self.llm.ainvoke(f"{context}\n\nUser: {state['user_input']}\n\nAgent:")
            state["critique"] = "Correction failed due to parsing error."
            state["response"] = fallback_response.content.strip()

        return state
