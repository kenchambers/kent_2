import re

def create_layer_name(description: str) -> str:
    """
    Creates a file-safe layer name from a description.
    """
    # Remove special characters and replace spaces with underscores
    s = re.sub(r"[^\w\s]", "", description)
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def log_thinking(message: str):
    """
    Logs a message in a muted dark green color.
    """
    print(f"\033[32m{message}\033[0m")

def log_response(message: str):
    """
    Logs a message in a bold bright purple color.
    """
    print(f"\033[1m\033[95m{message}\033[0m")
