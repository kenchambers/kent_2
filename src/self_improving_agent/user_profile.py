import json
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, Any
import asyncio

from . import memory, config

USER_PROFILES_DIR = Path("user_profiles")
USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

USER_VECTOR_STORES_DIR = config.VECTOR_STORES_DIR / "user_specific"
USER_VECTOR_STORES_DIR.mkdir(parents=True, exist_ok=True)


async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Loads a user profile, or creates a default one if it doesn't exist.
    Also handles creation of the user-specific vector store.
    """
    profile_path = USER_PROFILES_DIR / f"{user_id}.json"
    if profile_path.exists():
        try:
            with open(profile_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or gone, create a new one
            pass

    # Create new profile
    vector_store_path = str(USER_VECTOR_STORES_DIR / f"{user_id}.faiss")

    # Initialize an empty vector store
    await asyncio.to_thread(
        memory.get_vector_store, vector_store_path, initial_documents=[]
    )

    profile = {
        "id": user_id,
        "name": "user",
        "created_at": datetime.now().isoformat(),
        "vector_store_path": vector_store_path,
        "facts": {},
    }
    await save_user_profile(user_id, profile)
    return profile


async def save_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """Saves the user profile to a JSON file."""
    profile_path = USER_PROFILES_DIR / f"{user_id}.json"

    def _save():
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=4)

    await asyncio.to_thread(_save)
