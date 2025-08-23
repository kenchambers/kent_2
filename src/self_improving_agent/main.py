import asyncio
import sys
from self_improving_agent.agent import SelfImprovingAgent
from self_improving_agent.utils import log_response

async def amain():
    """
    The main asynchronous function for the agent.
    """
    agent = SelfImprovingAgent()
    print("Welcome to the Self-Improving Agent! (Streaming Mode)")
    print("Type 'exit' to quit.")
    print()

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            
            print("Agent: ", end="", flush=True)
            response_content = ""
            
            # Stream the response
            async for chunk in agent.astream(user_input):
                if chunk["type"] == "thinking":
                    print(f"\r\033[90m[Thinking: {chunk['content']}]\033[0m", end="", flush=True)
                elif chunk["type"] == "response":
                    print(f"\r{chunk['content']}")
                    response_content = chunk["content"]
                    break
                elif chunk["type"] == "error":
                    print(f"\r\033[91mError: {chunk['content']}\033[0m")
                    break
            
            if not response_content:
                print("\rNo response generated.")
                
    finally:
        await agent.aclose()
        print("\nConnection closed. Goodbye!")


def main():
    """
    Synchronous entry point for the agent.
    """
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
