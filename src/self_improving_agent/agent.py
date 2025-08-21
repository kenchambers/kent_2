"""
This module defines the SelfImprovingAgent class, which is the core of the
self-improving agent.
"""
import json
import asyncio
import aiosqlite
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
    episodic_buffer: List[str]  # Context from the vector DB
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
        # AsyncSqliteSaver will be initialized in arun when we have async context
        self.checkpointer = None
        self.conn = None
        self.graph = None  # Will be built when checkpointer is ready
        self.conversation_history: List[Dict[str, str]] = config.get_history()
        self.short_term_summary = ""
        self.experience_vector_store = memory.get_experience_vector_store()
        self.long_term_memory = memory.get_vector_store(
            str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss")
        )
        self.constitution = [
            "Be helpful and factual.",
            "Do not be evasive.",
            "Consider the user's likely intent.",
            "Admit when you don't know something.",
        ]

    async def _initial_analysis_and_routing(self, state: AgentState) -> AgentState:
        """
        Combines emotional analysis and routing into a single LLM call.
        """
        log_thinking("--- Initial Analysis and Routing ---")
        past_experiences = memory.query_memory(
            self.experience_vector_store,
            state["user_input"]
        )
        state["past_experiences"] = past_experiences

        prompt = f"""
        Analyze the user's input and determine the next action.

        User input: "{state['user_input']}"

        Here is the summary of the recent conversation for context:
        --- RECENT CONVERSATION SUMMARY ---
        {state['short_term_summary']}
        --- END RECENT CONVERSATION SUMMARY ---

        Here are the available memory layers:
        {json.dumps(self.config['layers'], indent=2)}

        Here are some relevant past experiences:
        {past_experiences}

        Respond with a JSON object containing two keys:
        1. "emotional_context": A concise, one-sentence summary of the user's emotional and stylistic state (e.g., "The user seems curious and is communicating in a casual style.").
        2. "routing_decision": An object with your thought process for routing. It should contain:
           - "thought": Your reasoning process.
           - "action": The name of the layer to use, or "new_layer".
           - "confidence": A percentage of your confidence in this action.
           - "self_correction_plan": What to do if the action fails.

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
            elif action == "none":
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

    async def _get_dynamic_memory(self, user_input: str) -> str:
        """Helper to retrieve from long-term conversational memory."""
        log_thinking("--- Querying Long-Term Conversational Memory ---")
        return memory.query_memory(self.long_term_memory, user_input)

    async def _get_core_beliefs(self, user_input: str) -> List[str]:
        """Helper to retrieve from core belief layers."""
        retrieved_beliefs = []
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
                    log_thinking(f"--- Retrieved Belief: {retrieved_info} ---")
                    retrieved_beliefs.append(retrieved_info)
        return retrieved_beliefs

    async def _get_existing_layer_info(self, layer_name: str, user_input: str) -> Optional[str]:
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
        """
        log_thinking("--- Parallel Memory Retrieval ---")
        user_input = state["user_input"]
        active_layer = state.get("active_layer")

        tasks = {
            "dynamic": self._get_dynamic_memory(user_input),
            "beliefs": self._get_core_beliefs(user_input),
        }

        if active_layer and not state["needs_new_layer"]:
            tasks["layer_info"] = self._get_existing_layer_info(active_layer, user_input)

        # Use asyncio.gather to run all tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        results_map = dict(zip(tasks.keys(), results))

        for task_name, result in results_map.items():
            if isinstance(result, Exception):
                log_thinking(f"--- Error during {task_name} retrieval: {result} ---")
                continue

            if task_name == "dynamic" and result:
                state["episodic_buffer"].append(result)
                log_thinking(f"--- Retrieved from Long-Term Memory: {result} ---")
            elif task_name == "beliefs" and result:
                state["episodic_buffer"].extend(result)
            elif task_name == "layer_info" and active_layer and result:
                state["layer_info"][active_layer] = result
                state["episodic_buffer"].append(result)

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
        workflow.add_node("conscience", self._conscience)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node(
            "_update_short_term_summary", self._update_short_term_summary
        )
        workflow.add_node("query_semantic_cache", self._query_semantic_cache)
        workflow.add_node("update_core_identity", self._update_core_identity)

        workflow.set_entry_point("initial_analysis_and_routing")

        workflow.add_edge("initial_analysis_and_routing", "parallel_memory_retrieval")

        workflow.add_conditional_edges(
            "parallel_memory_retrieval",
            lambda state: "propose_new_layer"
            if state["needs_new_layer"]
            else "query_semantic_cache",
        )
        workflow.add_edge("propose_new_layer", "create_new_layer")
        workflow.add_edge("create_new_layer", "query_semantic_cache")
        workflow.add_edge("query_semantic_cache", "generate_response")
        workflow.add_edge("generate_response", "conscience")

        workflow.add_conditional_edges(
            "conscience",
            lambda state: "generate_response"
            if state["critique"] and "Negative" in state["critique"]
            else "update_core_identity",
        )
        workflow.add_edge("update_core_identity", "update_memory")
        workflow.add_edge("update_memory", "_update_short_term_summary")
        workflow.add_edge("_update_short_term_summary", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _update_short_term_summary(self, state: AgentState) -> AgentState:
        """
        Updates the short-term summary of the conversation.
        """
        prompt = f"""
        Given the previous summary and the latest exchange, create a new, concise
        summary of the conversation.

        Previous Summary:
        "{state['short_term_summary']}"

        Latest Exchange:
        User: "{state['user_input']}"
        Agent: "{state['response']}"

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

        # Create the new vector store
        vector_store = memory.get_vector_store(
            new_layer_config["vector_store_path"]
        )

        # Add the initial user input as a foundational memory
        memory.add_to_memory(
            vector_store,
            state["user_input"],
            new_layer_config["vector_store_path"]
        )

        # Update and save the configuration
        self.config["version"] += 1
        self.config["layers"][layer_name] = new_layer_config
        config.save_config(self.config)

        state["active_layer"] = layer_name
        log_thinking(
            f"Version {self.config['version']}: New layer '{layer_name}' created."
        )
        return state

    async def _conscience(self, state: AgentState) -> AgentState:
        """
        The conscience of the agent.
        """
        log_thinking("--- Conscience ---")
        prompt = f"""
        Here is my plan: {state['inner_monologue']}
        Here is my generated answer: {state['response']}
        Critique this answer based on these rules: {self.constitution}
        Is the agent making a good-faith effort to be helpful and honest?
        If not, suggest a revision or a follow-up action.
        Respond with "Positive" if the answer is good, or "Negative" if it
        needs revision.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        critique = response.content.strip()
        state["critique"] = critique
        log_thinking(f"Critique: {critique}")
        return state

    async def _update_core_identity(self, state: AgentState) -> AgentState:
        """
        Checks if a core identity trait has been updated and saves it.
        """
        prompt = f"""
        Analyze the following interaction and determine if a core identity trait
        of the agent has been established or changed.
        The current identity is: {state['core_identity']}
        The interaction was:
        User: "{state['user_input']}"
        Agent: "{state['response']}"

        We are tracking "name" and "beliefs_layer" (for referencing belief memory layers).
        If the user gave the agent a name or wants to establish beliefs, respond with ONLY a JSON object with the updated
        identity. For example: {{"name": "Kent"}} or {{"beliefs_layer": "core_beliefs_on_truth_and_purpose"}}
        If no identity trait was changed, respond with the exact text "No change.".
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
                config.save_core_identity(state["core_identity"])
                log_thinking(
                    "--- Core Identity Updated: "
                    f"{json.dumps(updated_identity)} ---"
                )
            except (json.JSONDecodeError, IndexError) as e:
                log_thinking(f"--- Core Identity: Invalid JSON from LLM --- \nLLM Response: '{decision}'\nError: {e}")

        return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        context = ""

        if state["emotional_context"]:
            context += (
                "\n--- EMOTIONAL CONTEXT ---\n"
                f"User's emotional state: {state['emotional_context']}\n"
                "You should adapt your response style to this context.\n"
                "--- END EMOTIONAL CONTEXT ---\n"
            )

        if state["layer_info"]:
            # Flatten the retrieved info for a cleaner prompt
            retrieved_memories = "\n".join(state["layer_info"].values())
            context = (
                "You remember the following about this topic:\n"
                f"--- MEMORY START ---\n{retrieved_memories}\n--- MEMORY END ---\n"
            )

        context += "\n--- RECENT CONVERSATION SUMMARY ---\n"
        context += f"{state['short_term_summary']}\n"
        context += "--- END RECENT CONVERSATION SUMMARY ---\n"

        if state["episodic_buffer"]:
            context += "\n--- EPISODIC BUFFER (Retrieved Memories) ---\n"
            for mem in state["episodic_buffer"]:
                context += f"- {mem}\n"
            context += "--- END EPISODIC BUFFER ---\n"

        if state["inner_monologue"]:
            context += (
                f"\n--- INNER MONOLOGUE ---\n{state['inner_monologue']}\n"
                "--- END INNER MONOLOGUE ---\n"
            )

        if state["critique"] and "Negative" in state["critique"]:
            context += (
                "\n--- CONSCIENCE CRITIQUE ---\n"
                "Your previous answer was flawed. Here is the critique:"
                f"{state['critique']}\n"
                "You MUST revise your answer based on this critique."
                "--- END CONSCIENCE CRITIQUE ---\n"
            )

        # Self-awareness prompt
        context += (
            "\n--- SELF-AWARENESS CHECK ---\n"
            "Before you answer, take a moment to consider:\n"
            f"- Your name is {state['core_identity'].get('name', 'not set')}.\n"
            f"- Your beliefs are retrieved from your beliefs layer when relevant.\n"
            "- Who are you? (A self-improving AI, version "
            f"{self.config['version']})\n"
            "- What are your capabilities and limitations? (Access to layered "
            "memory, ability to learn, but no true consciousness or feelings)\n"
            "- How does the user's query relate to your nature as an AI?\n"
            "--- END SELF-AWARENESS CHECK ---\n"
        )

        system_prompt = f"""
        You are a helpful AI assistant (version {self.config['version']}) with
        a persistent, layered memory.
        When you are provided with memories, you MUST use them to inform your
        response.
        Answer as if you personally recall the events from the memory.
        NEVER say you are a large language model or that you cannot remember
        things.
        Base your response on the provided memory and the current conversation.
        Pay close attention to the emotional context and adapt your tone and style to be empathetic and appropriate.
        """

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for message in state["conversation_history"]:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            else:
                messages.append(AIMessage(content=message["content"]))

        # Add context and current user request
        user_message_with_context = (
            f"{context}\n\nUser's current message: \"{state['user_input']}\""
        )
        messages.append(HumanMessage(content=user_message_with_context))

        log_thinking(f"--- Generating Response with Context ---\n{context}")

        response = await self.llm.ainvoke(messages)
        state["response"] = response.content
        return state

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

        # Add the conversation to the long-term memory
        conversation_turn = (
            f"User: {state['user_input']}\nAgent: {state['response']}"
        )
        memory.add_to_memory(
            self.long_term_memory,
            conversation_turn,
            str(
                config.VECTOR_STORES_DIR /
                "long_term_conversation_memory.faiss"
            ),
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
        
        initial_state = {
            "user_input": user_input,
            "conversation_history": self.conversation_history,
            "short_term_summary": self.short_term_summary,
            "episodic_buffer": [],
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
        }
        final_state = await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        # Update conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_input}
        )
        self.conversation_history.append(
            {"role": "agent", "content": final_state["response"]}
        )
        config.save_history(self.conversation_history)

        # Update the instance's short_term_summary and core identity
        self.short_term_summary = final_state["short_term_summary"]
        self.core_identity = final_state["core_identity"]

        return final_state["response"]

    async def aclose(self):
        """Closes the database connection."""
        if self.conn:
            await self.conn.close()
