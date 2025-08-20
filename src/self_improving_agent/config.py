import json
import os
from pathlib import Path
from typing import Dict, Any, List

CONFIG_FILE = Path("agent_config.json")
HISTORY_FILE = Path("conversation_history.json")
VECTOR_STORES_DIR = Path("vector_stores")

def get_config() -> Dict[str, Any]:
    """
    Loads the agent's configuration from the config file.
    """
    if not CONFIG_FILE.exists():
        return {
            "version": 1,
            "layers": {
                "general_knowledge": {
                    "description": "For general questions and knowledge.",
                    "vector_store_path": str(VECTOR_STORES_DIR / "general_knowledge.faiss"),
                }
            },
        }
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config: Dict[str, Any]):
    """
    Saves the agent's configuration to the config file.
    """
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def get_history() -> List[Dict[str, str]]:
    """
    Loads the conversation history from the history file.
    """
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history: List[Dict[str, str]]):
    """
    Saves the conversation history to the history file.
    """
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_api_key() -> str:
    """
    Loads the Google API key from the .env file.
    """
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    return api_key

CORE_IDENTITY_FILE = Path("src/self_improving_agent/core_identity.json")

def get_core_identity() -> Dict[str, Any]:
    """
    Loads the agent's core identity from the identity file.
    """
    if not CORE_IDENTITY_FILE.exists():
        return {"name": None}
    with open(CORE_IDENTITY_FILE, "r") as f:
        return json.load(f)

def save_core_identity(identity: Dict[str, Any]):
    """
    Saves the agent's core identity to the identity file.
    """
    with open(CORE_IDENTITY_FILE, "w") as f:
        json.dump(identity, f, indent=4)
