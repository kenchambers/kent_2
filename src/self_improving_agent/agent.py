"""
This module defines the SelfImprovingAgent class, which is the core of the
self-improving agent.
"""
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
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
    long_term_memory: Optional[Any]  # The LTM vector store
    active_layer: Optional[str]
    layer_info: Dict[str, Any]
    new_layer_description: Optional[str]
    needs_new_layer: bool
    response: str
    inner_monologue: Optional[str]
    critique: Optional[str]
    past_experiences: Optional[str]
    core_identity: Dict[str, Any]


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
        self.graph = self._build_graph()
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

    def _build_graph(self):
        """
        Builds the agent's graph.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("inner_monologue", self._inner_monologue)
        workflow.add_node("query_existing_layer", self._query_existing_layer)
        workflow.add_node("propose_new_layer", self._propose_new_layer)
        workflow.add_node("create_new_layer", self._create_new_layer)
        workflow.add_node("conscience", self._conscience)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)

        workflow.add_node(
            "_update_short_term_summary", self._update_short_term_summary
        )
        workflow.add_node(
            "_dynamic_memory_retrieval", self._dynamic_memory_retrieval
        )
        workflow.add_node("update_episodic_buffer", self._update_episodic_buffer)
        workflow.add_node("query_semantic_cache", self._query_semantic_cache)
        workflow.add_node("update_core_identity", self._update_core_identity)

        workflow.set_entry_point("inner_monologue")

        workflow.add_edge("inner_monologue", "_dynamic_memory_retrieval")

        workflow.add_conditional_edges(
            "_dynamic_memory_retrieval",
            lambda state: "propose_new_layer"
            if state["needs_new_layer"]
            else "query_existing_layer",
        )
        workflow.add_edge("query_existing_layer", "update_episodic_buffer")
        workflow.add_edge("propose_new_layer", "create_new_layer")
        workflow.add_edge("create_new_layer", "query_existing_layer")
        workflow.add_edge("update_episodic_buffer", "query_semantic_cache")
        workflow.add_edge("query_semantic_cache", "generate_response")
        workflow.add_conditional_edges(
            "conscience",
            lambda state: "generate_response"
            if state["critique"] and "Negative" in state["critique"]
            else "update_core_identity",
        )
        workflow.add_edge("update_core_identity", "update_memory")
        workflow.add_edge("update_memory", "_update_short_term_summary")
        workflow.add_edge("_update_short_term_summary", END)

        return workflow.compile()

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

    async def _dynamic_memory_retrieval(self, state: AgentState) -> AgentState:
        """
        Retrieves relevant memories from the long-term conversational memory.
        """
        retrieved_memories = memory.query_memory(
            state["long_term_memory"],
            state["user_input"]
        )
        state["episodic_buffer"].append(retrieved_memories)
        log_thinking(
            "--- Retrieved from Long-Term Memory: "
            f"{retrieved_memories} ---"
        )
        return state

    async def _inner_monologue(self, state: AgentState) -> AgentState:
        """
        The inner monologue of the agent.
        """
        log_thinking("--- Inner Monologue ---")
        past_experiences = memory.query_memory(
            self.experience_vector_store,
            state["user_input"]
        )
        state["past_experiences"] = past_experiences

        prompt = f"""
        You are a routing agent. Your job is to determine if the user's request
        can be answered by one of the existing memory layers, or if a new
        layer is needed.

        Here is the summary of the recent conversation for context:
        --- RECENT CONVERSATION SUMMARY ---
        {state['short_term_summary']}
        --- END RECENT CONVERSATION SUMMARY ---

        The user's latest request is: "{state['user_input']}"

        Here are the available layers:
        {json.dumps(self.config['layers'], indent=2)}

        Here are some relevant past experiences:
        {past_experiences}

        Based on the user's request, recent conversation summary, and past experiences, generate a structured
        "thought" string.
        Your thought process should include the following:
        - **Thought:** Your reasoning process.
        - **Action:** The name of the layer to use, or "new_layer".
        - **Confidence:** A percentage of your confidence in this action.
        - **Self-Correction Plan:** What to do if the action fails.

        Example:
        Thought: The user is asking about the history of the internet. The
        "history_of_technology" layer seems relevant.
        Action: history_of_technology
        Confidence: 90%
        Self-Correction Plan: If the "history_of_technology" layer does not
        have the answer, I will create a new layer called
        "history_of_the_internet".
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        monologue = response.content.strip()
        state["inner_monologue"] = monologue
        log_thinking(monologue)

        # Extract the action from the monologue
        action = "new_layer"
        for line in monologue.split("\n"):
            if line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()
                break

        if action == "new_layer":
            state["needs_new_layer"] = True
        else:
            state["needs_new_layer"] = False
            state["active_layer"] = action

        return state

    async def _route_request(self, state: AgentState) -> AgentState:
        prompt = f"""
        You are a routing agent. Your job is to determine if the user's request
        can be answered by one of the existing memory layers, or if a new
        layer is needed.

        The user's request is: "{state['user_input']}"

        Here are the available layers:
        {json.dumps(self.config['layers'], indent=2)}

        Does the user's request fit into one of the existing layers?
        If yes, respond with ONLY the name of the layer (e.g.,
        'general_knowledge').
        If no, respond with ONLY the word "new_layer".
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision = response.content.strip()

        if decision == "new_layer":
            state["needs_new_layer"] = True
        else:
            state["needs_new_layer"] = False
            state["active_layer"] = decision

        return state

    async def _query_existing_layer(self, state: AgentState) -> AgentState:
        layer_name = state["active_layer"]
        if layer_name and layer_name in self.config["layers"]:
            log_thinking(f"--- Checking Memory Layer: {layer_name} ---")
            layer_config = self.config["layers"][layer_name]
            vector_store = memory.get_vector_store(
                layer_config["vector_store_path"]
            )
            retrieved_info = memory.query_memory(
                vector_store,
                state["user_input"]
            )
            log_thinking(f"--- Retrieved Info: {retrieved_info} ---")
            state["layer_info"][layer_name] = retrieved_info
        return state



    async def _update_episodic_buffer(self, state: AgentState) -> AgentState:
        # Placeholder for more complex logic
        if state["active_layer"] and state["layer_info"]:
            retrieved_info = state["layer_info"][state["active_layer"]]
            state["episodic_buffer"].append(retrieved_info)
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
        Does this answer *truly* address what the user was likely wanting to
        know? If not, suggest a revision or a follow-up action.
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

        Currently, we are only tracking the "name" trait.
        If the user gave the agent a name, respond in JSON format with the updated
        identity. For example: {{"name": "Kent"}}
        If no identity trait was changed, respond with "No change."
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision = response.content.strip()

        if "No change." not in decision:
            try:
                updated_identity = json.loads(decision)
                state["core_identity"].update(updated_identity)
                config.save_core_identity(state["core_identity"])
                log_thinking(
                    "--- Core Identity Updated: "
                    f"{json.dumps(updated_identity)} ---"
                )
            except json.JSONDecodeError:
                log_thinking("--- Core Identity: Invalid JSON from LLM ---")

        return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        context = ""
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
            state["long_term_memory"],
            conversation_turn,
            str(
                config.VECTOR_STORES_DIR /
                "long_term_conversation_memory.faiss"
            ),
        )

        # Update semantic cache
        state["semantic_cache"][state["user_input"]] = state["response"]

        return state

    async def arun(self, user_input: str):
        """
        Runs the agent.
        """
        initial_state = {
            "user_input": user_input,
            "conversation_history": self.conversation_history,
            "short_term_summary": self.short_term_summary,
            "episodic_buffer": [],
            "semantic_cache": {},
            "long_term_memory": self.long_term_memory,
            "layer_info": {},
            "active_layer": None,
            "new_layer_description": None,
            "needs_new_layer": False,
            "response": "",
            "inner_monologue": None,
            "critique": None,
            "past_experiences": None,
            "core_identity": self.core_identity,
        }
        final_state = await self.graph.ainvoke(initial_state)

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
