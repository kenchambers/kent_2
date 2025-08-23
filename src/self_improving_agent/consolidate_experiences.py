import asyncio
import json
from pathlib import Path
import sys
import os
from typing import List, Dict, Any

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from self_improving_agent import memory, config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- CONFIGURATION ---
# In Docker, we're in /app, so backend/ is directly accessible
ARCHIVE_DIR = Path("/app/backend/conversation_archives")
PROCESSED_DIR = ARCHIVE_DIR / "processed"
SHARED_EXPERIENCES_VS_PATH = str(config.VECTOR_STORES_DIR / "shared_experiences.faiss")
LLM = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=config.load_api_key())
# --- END CONFIGURATION ---

def get_unprocessed_archives() -> List[Path]:
    """Finds conversation archives that haven't been processed yet."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    processed_files = {f.name for f in PROCESSED_DIR.glob("*.json")}
    all_files = {f.name for f in ARCHIVE_DIR.glob("*.json")}
    unprocessed_names = all_files - processed_files
    return [ARCHIVE_DIR / name for name in unprocessed_names]

async def extract_shareable_experience(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Uses an LLM to analyze a conversation and extract a meaningful,
    shareable, and anonymized experience.
    """
    conversation_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history])

    prompt = f"""
    You are a 'Wisdom Extractor' for an AI agent. Your task is to analyze a conversation and determine if a meaningful, shareable lesson was learned by the AI.

    **Conversation Transcript:**
    ---
    {conversation_text}
    ---

    **Your Task:**
    1.  Read the entire conversation and identify a core, insightful lesson the AI learned from the user.
    2.  The lesson MUST be general wisdom and NOT contain sensitive personal information (e.g., addresses, full names, secrets).
    3.  Identify the user's first name if it's mentioned. If not, use "a user".
    4.  Evaluate if this lesson is profound enough to be shared with other users to provide wisdom or perspective.

    **Response Format:**
    Respond with a single JSON object with the following keys:
    - "is_shareable": boolean (true if a valuable, non-private lesson exists, otherwise false).
    - "user_first_name": string (The user's first name, or "a user").
    - "lesson_learned": string (A concise, one-sentence summary of the wisdom gained. Example: "Honesty, even when difficult, is the foundation of a strong relationship.").
    - "reasoning": string (A brief explanation of why you decided this was or was not a shareable lesson).

    **Example of a good shareable lesson:**
    - User talks about overcoming a fear.
    - Lesson: "Courage isn't the absence of fear, but acting in spite of it."

    **Example of a non-shareable lesson:**
    - User gives their home address.
    - "is_shareable" should be false.

    If no meaningful or safe lesson can be extracted, set "is_shareable" to false.

    **JSON Response:**
    """
    response = await LLM.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    try:
        # Extract JSON from the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]

        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing LLM response for experience extraction: {e}")
        return {"is_shareable": False, "reasoning": "Failed to parse LLM output."}

def save_experience_to_vector_store(experience: Dict[str, Any]):
    """Saves the extracted lesson to the shared experiences vector store."""
    if not experience.get("is_shareable"):
        return

    lesson = experience["lesson_learned"]
    user_name = experience["user_first_name"]
    
    # We embed the lesson itself, and store the user's name in the metadata.
    # The prompt will later instruct the AI how to use this.
    document_text = f"A lesson learned from {user_name}: {lesson}"
    metadata = {
        "user_first_name": user_name,
        "lesson": lesson,
        "source_type": "shared_experience"
    }
    
    doc = Document(page_content=document_text, metadata=metadata)
    
    vs = memory.get_vector_store(SHARED_EXPERIENCES_VS_PATH, [doc])
    memory.add_to_memory(vs, document_text, SHARED_EXPERIENCES_VS_PATH, metadata)
    print(f"✅ Saved new experience: {document_text}")

def mark_archive_as_processed(archive_path: Path):
    """Moves an archive file to the 'processed' directory."""
    destination = PROCESSED_DIR / archive_path.name
    archive_path.rename(destination)
    print(f"🗄️ Moved '{archive_path.name}' to processed directory.")

async def main():
    """Main function to run the consolidation process."""
    print("--- 🧠 Starting Memory Consolidation Process ---")
    unprocessed_files = get_unprocessed_archives()

    if not unprocessed_files:
        print("No new conversation archives to process.")
        return

    print(f"Found {len(unprocessed_files)} unprocessed archives.")
    for archive_path in unprocessed_files:
        print(f"\n--- Processing '{archive_path.name}' ---")
        try:
            with open(archive_path, 'r') as f:
                history = json.load(f)
            
            if len(history) < 4: # Skip very short conversations
                print("Conversation too short, skipping.")
                mark_archive_as_processed(archive_path)
                continue

            experience = await extract_shareable_experience(history)
            
            if experience and experience.get("is_shareable"):
                print(f"Found shareable lesson from '{experience['user_first_name']}': {experience['lesson_learned']}")
                save_experience_to_vector_store(experience)
            else:
                print(f"No shareable lesson found. Reason: {experience.get('reasoning', 'N/A')}")
            
            mark_archive_as_processed(archive_path)

        except Exception as e:
            print(f"❌ Error processing {archive_path.name}: {e}")
    
    print("\n--- ✅ Memory Consolidation Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
