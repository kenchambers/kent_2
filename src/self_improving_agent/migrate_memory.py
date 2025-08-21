import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from . import memory
from . import config
import json

async def migrate_legacy_memory():
    """
    Loads raw conversation turns from the legacy vector store,
    summarizes them, and adds them to the session_summaries store.
    """
    print("--- Starting Legacy Memory Migration ---")

    # 1. Load vector stores
    legacy_memory_path = str(config.VECTOR_STORES_DIR / "long_term_conversation_memory.faiss")
    try:
        legacy_store = memory.get_vector_store(legacy_memory_path)
        print("✅ Loaded legacy memory store.")
    except Exception as e:
        print(f"❌ Could not load legacy memory store at {legacy_memory_path}. It might not exist. Aborting. Error: {e}")
        return

    session_summaries_path = str(config.VECTOR_STORES_DIR / "session_summaries.faiss")
    session_store = memory.get_vector_store(session_summaries_path)
    print("✅ Loaded session summaries store.")

    # 2. Extract all documents from the legacy store
    if not hasattr(legacy_store, 'docstore') or not hasattr(legacy_store.docstore, '_dict'):
         print("❌ Legacy store format is incompatible. Cannot retrieve all documents.")
         return
         
    all_docs = legacy_store.docstore._dict.values()
    if not all_docs:
        print("Legacy memory store is empty. No migration needed.")
        return

    print(f"Found {len(all_docs)} conversation turns in legacy memory.")
    
    # 3. Combine into a single text block
    conversation_text = "\n".join([doc.page_content for doc in all_docs if doc.page_content != "initial document"])
    
    if not conversation_text.strip():
        print("Legacy memory contains no actual conversation content. No migration needed.")
        return

    print("Combining legacy conversations into a single text for summarization...")

    # 4. Summarize using the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=config.load_api_key()
    )

    summary_prompt = f"""
    You are a data migration assistant. Your task is to create a comprehensive summary of a large, unstructured block of past conversation history.

    Here is the conversation history:
    ---
    {conversation_text}
    ---

    Create a detailed summary that identifies:
    1. Key individuals mentioned (like "Ken" and "Kent").
    2. Main topics of discussion (e.g., the AI's memory, creativity, the user's LLM project).
    3. Recurring themes and patterns (e.g., the user's frustration with memory, the AI's explanation of its nature).
    4. Any key facts or decisions made.

    Format this as a coherent narrative under the title "Summary of Historical Conversations".
    """
    
    print("Generating summary of legacy conversations... (This may take a moment)")
    response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
    historical_summary = response.content.strip()

    # 5. Add the summary to the session_summaries store
    summary_with_metadata = f"Historical Conversation Summary (Migrated):\n{historical_summary}"
    memory.add_to_memory(
        session_store,
        summary_with_metadata,
        session_summaries_path
    )
    
    print("✅ Successfully added historical summary to the session summaries store.")
    print("\n--- Migration Complete ---")
    print(f"You can now safely delete the old memory directory: '{legacy_memory_path}'")


def main():
    # This allows the async function to be run from the command line
    # In a production environment, you might use a more robust CLI tool like typer or click
    try:
        asyncio.run(migrate_legacy_memory())
    except KeyboardInterrupt:
        print("\nMigration cancelled by user.")

if __name__ == "__main__":
    main()
