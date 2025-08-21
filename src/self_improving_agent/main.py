import asyncio
from self_improving_agent.agent import SelfImprovingAgent
from self_improving_agent.utils import log_response

async def amain():
    """
    The main asynchronous function for the agent.
    """
    agent = SelfImprovingAgent()
    print("Welcome to the Self-Improving Agent!")
    print("Type 'exit' to quit.")
    print()

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = await agent.arun(user_input)
            print(f"Agent: {response}")
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
