from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from . import config
from . import memory
import uuid
from .utils import create_layer_name, log_thinking
import json
from collections import deque

class AgentState(TypedDict):
    user_input: str
    conversation_history: List[Dict[str, str]]
    working_memory: deque # For short-term recall
    episodic_buffer: List[str] # Context from the vector DB
    semantic_cache: Dict[str, Any] # Caching factual lookups
    active_layer: Optional[str]
    layer_info: Dict[str, Any]
    new_layer_description: Optional[str]
    needs_new_layer: bool
    response: str

class SelfImprovingAgent:
    def __init__(self):
        self.config = config.get_config()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=config.load_api_key())
        self.graph = self._build_graph()
        self.conversation_history: List[Dict[str, str]] = config.get_history()
        self.working_memory = deque(maxlen=10)  # Keep the last 5 turns (user + agent)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("route_request", self._route_request)
        workflow.add_node("query_existing_layer", self._query_existing_layer)
        workflow.add_node("propose_new_layer", self._propose_new_layer)
        workflow.add_node("create_new_layer", self._create_new_layer)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("manage_working_memory", self._manage_working_memory)
        workflow.add_node("update_episodic_buffer", self._update_episodic_buffer)
        workflow.add_node("query_semantic_cache", self._query_semantic_cache)


        workflow.set_entry_point("route_request")

        workflow.add_conditional_edges(
            "route_request",
            lambda state: "propose_new_layer" if state["needs_new_layer"] else "query_existing_layer",
        )
        workflow.add_edge("query_existing_layer", "manage_working_memory")
        workflow.add_edge("propose_new_layer", "create_new_layer")
        workflow.add_edge("create_new_layer", "query_existing_layer") 
        workflow.add_edge("manage_working_memory", "update_episodic_buffer")
        workflow.add_edge("update_episodic_buffer", "query_semantic_cache")
        workflow.add_edge("query_semantic_cache", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()

    async def _route_request(self, state: AgentState) -> AgentState:
        prompt = f"""
        You are a routing agent. Your job is to determine if the user's request can be answered by one of the existing memory layers, or if a new layer is needed.

        The user's request is: "{state['user_input']}"

        Here are the available layers:
        {json.dumps(self.config['layers'], indent=2)}

        Does the user's request fit into one of the existing layers?
        If yes, respond with ONLY the name of the layer (e.g., 'general_knowledge').
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
            vector_store = memory.get_vector_store(layer_config["vector_store_path"])
            retrieved_info = memory.query_memory(vector_store, state["user_input"])
            log_thinking(f"--- Retrieved Info: {retrieved_info} ---")
            state["layer_info"][layer_name] = retrieved_info
        return state

    async def _manage_working_memory(self, state: AgentState) -> AgentState:
        # For now, this is handled in _update_memory and arun
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
            log_thinking(f"--- Semantic Cache Hit for: {state['user_input']} ---")
        return state

    async def _propose_new_layer(self, state: AgentState) -> AgentState:
        prompt = f"""
        You are a memory architect. Your job is to create a new memory layer for the agent.
        The user's request is: "{state['user_input']}"
        Based on this request, create a concise, one-sentence description for a new memory layer.
        For example, if the user asks about the history of Rome, the description could be: "For questions about the history of ancient Rome."
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["new_layer_description"] = response.content.strip()
        return state

    async def _create_new_layer(self, state: AgentState) -> AgentState:
        description = state["new_layer_description"]
        layer_name = create_layer_name(description)
        
        new_layer_config = {
            "description": description,
            "vector_store_path": str(config.VECTOR_STORES_DIR / f"{layer_name}.faiss"),
        }
        
        # Create the new vector store
        vector_store = memory.get_vector_store(new_layer_config["vector_store_path"])
        
        # Add the initial user input as a foundational memory
        memory.add_to_memory(vector_store, state["user_input"], new_layer_config["vector_store_path"])

        # Update and save the configuration
        self.config["version"] += 1
        self.config["layers"][layer_name] = new_layer_config
        config.save_config(self.config)
        
        state["active_layer"] = layer_name
        log_thinking(f"Version {self.config['version']}: New layer '{layer_name}' created.")
        return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        context = ""
        if state["layer_info"]:
            # Flatten the retrieved info for a cleaner prompt
            retrieved_memories = "\n".join(state["layer_info"].values())
            context = f"You remember the following about this topic:\n--- MEMORY START ---\n{retrieved_memories}\n--- MEMORY END ---\n"
        
        context += f"\n--- WORKING MEMORY (Recent Conversation) ---\n"
        for turn in state["working_memory"]:
            context += f"{turn['role'].capitalize()}: {turn['content']}\n"
        context += "--- END WORKING MEMORY ---\n"

        if state["episodic_buffer"]:
            context += f"\n--- EPISODIC BUFFER (Retrieved Memories) ---\n"
            for mem in state["episodic_buffer"]:
                context += f"- {mem}\n"
            context += "--- END EPISODIC BUFFER ---\n"


        system_prompt = f"""
        You are a helpful AI assistant (version {self.config['version']}) with a persistent, layered memory.
        When you are provided with memories, you MUST use them to inform your response.
        Answer as if you personally recall the events from the memory.
        NEVER say you are a large language model or that you cannot remember things.
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
        user_message_with_context = f"{context}\n\nUser's current message: \"{state['user_input']}\""
        messages.append(HumanMessage(content=user_message_with_context))
        
        response = await self.llm.ainvoke(messages)
        state["response"] = response.content
        return state

    async def _update_memory(self, state: AgentState) -> AgentState:
        layer_name = state["active_layer"]
        if layer_name:
            layer_config = self.config["layers"][layer_name]
            vector_store = memory.get_vector_store(layer_config["vector_store_path"])
            
            # Create a summary of the interaction to store in memory
            summary_prompt = f"""
            Based on the following interaction, create a concise summary of the key information discussed.
            User: "{state['user_input']}"
            Agent: "{state['response']}"
            """
            summary_response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
            summary = summary_response.content
            
            memory.add_to_memory(vector_store, summary, layer_config["vector_store_path"])
        
        # Update working memory
        state["working_memory"].append({"role": "user", "content": state["user_input"]})
        state["working_memory"].append({"role": "agent", "content": state["response"]})
        
        # Update semantic cache
        state["semantic_cache"][state["user_input"]] = state["response"]

        return state

    async def arun(self, user_input: str):
        initial_state = {
            "user_input": user_input,
            "conversation_history": self.conversation_history,
            "working_memory": self.working_memory,
            "episodic_buffer": [],
            "semantic_cache": {},
            "layer_info": {},
            "active_layer": None,
            "new_layer_description": None,
            "needs_new_layer": False,
            "response": "",
        }
        final_state = await self.graph.ainvoke(initial_state)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "agent", "content": final_state["response"]})
        config.save_history(self.conversation_history)
        
        # Update the instance's working memory
        self.working_memory = final_state["working_memory"]
        
        return final_state["response"]
