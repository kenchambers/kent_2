import os
import shutil

def main():
    """
    Clears all agent memory, including vector stores, conversation history,
    and database caches, resetting the agent to a clean slate.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"Project root identified as: {project_root}")

    # Directories to remove
    paths_to_remove = [
        # Root directories and files
        os.path.join(project_root, 'vector_stores'),
        os.path.join(project_root, 'conversation_archives'),
        os.path.join(project_root, 'user_profiles'),
        os.path.join(project_root, 'conversations.db'),
        os.path.join(project_root, 'langgraph_cache.db'),
        os.path.join(project_root, 'langgraph_cache.db-shm'),
        os.path.join(project_root, 'langgraph_cache.db-wal'),
        # Backend directories and files
        os.path.join(project_root, 'backend', 'vector_stores'),
        os.path.join(project_root, 'backend', 'conversations.db'),
        os.path.join(project_root, 'backend', 'langgraph_cache.db'),
        os.path.join(project_root, 'backend', 'langgraph_cache.db-shm'),
        os.path.join(project_root, 'backend', 'langgraph_cache.db-wal'),
        os.path.join(project_root, 'backend', 'conversation_archives'),
        os.path.join(project_root, 'backend', 'user_profiles'),
    ]

    # Config files to reset from templates
    configs_to_reset = [
        # Root configs
        (os.path.join(project_root, 'agent_config.json'), os.path.join(project_root, 'agent_config.json.template')),
        (os.path.join(project_root, 'conversation_history.json'), os.path.join(project_root, 'conversation_history.json.template')),
        (os.path.join(project_root, 'src', 'self_improving_agent', 'core_identity.json'), os.path.join(project_root, 'src', 'self_improving_agent', 'core_identity.json.template')),
        # Backend configs
        (os.path.join(project_root, 'backend', 'agent_config.json'), os.path.join(project_root, 'agent_config.json.template')),
        (os.path.join(project_root, 'backend', 'conversation_history.json'), os.path.join(project_root, 'conversation_history.json.template')),
    ]

    # --- Deletion and Resetting ---

    print("\n--- Clearing Memory Files and Directories ---")
    for path in paths_to_remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"Removed file: {path}")
        else:
            print(f"Path not found, skipping: {path}")

    print("\n--- Resetting Configuration Files from Templates ---")
    for target_file, template_file in configs_to_reset:
        try:
            shutil.copy(template_file, target_file)
            print(f"Reset '{target_file}' from template.")
        except FileNotFoundError:
            print(f"Template not found for '{target_file}', skipping reset.")
        except Exception as e:
            print(f"Error resetting '{target_file}': {e}")

    print("\n✅ Agent memory cleared successfully.")

if __name__ == "__main__":
    main()
