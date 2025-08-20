import re
import asyncio
from typing import Optional, Callable
import threading

# Global callback for streaming thinking logs
_thinking_callback: Optional[Callable[[str], None]] = None
_thinking_callback_lock = threading.Lock()

def create_layer_name(description: str) -> str:
    """
    Creates a file-safe layer name from a description.
    """
    # Remove special characters and replace spaces with underscores
    s = re.sub(r"[^\w\s]", "", description)
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def set_thinking_callback(callback: Optional[Callable[[str], None]]):
    """
    Set a callback function to receive thinking logs for streaming.
    """
    global _thinking_callback
    with _thinking_callback_lock:
        _thinking_callback = callback

def log_thinking(message: str):
    """
    Logs a message in a muted dark green color and optionally streams it via callback.
    """
    # Always print to console for backward compatibility
    print(f"\033[32m{message}\033[0m")
    
    # Also call the streaming callback if set
    with _thinking_callback_lock:
        if _thinking_callback:
            try:
                _thinking_callback(message)
            except Exception as e:
                print(f"\033[31mError in thinking callback: {e}\033[0m")

def log_response(message: str):
    """
    Logs a message in a bold bright purple color.
    """
    print(f"\033[1m\033[95m{message}\033[0m")
