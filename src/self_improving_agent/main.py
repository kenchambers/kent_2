import asyncio
from self_improving_agent.agent import SelfImprovingAgent
from self_improving_agent.utils import log_response

async def amain():
    """
    Async main function to run the self-improving agent.
    """
    agent = SelfImprovingAgent()
    print("Welcome to the Self-Improving Agent!")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break

            response = await agent.arun(user_input)
            log_response(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

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
