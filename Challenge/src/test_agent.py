from agents import NetworkAgent
import time
import subprocess
import sys


def check_ollama_installed():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            return True
        else:
            print("❌ Ollama is installed but not running properly")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed. Please install Ollama first:")
        print("Visit: https://ollama.ai/download")
        return False


def list_available_models():
    """List available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        print("Available models:")
        print(result.stdout)
        return True
    except:
        return False


def test_agent_scenarios():
    """Test the intelligent agent with various scenarios"""

    if not check_ollama_installed():
        return

    # Model selection
    print("\n🤖 Select a model:")
    print("1. deepseek-r1:7b")
    print("2. deepseek-r1:14b")
    print("3. deepseek-r1:32b")

    choice = input("Enter choice (1-3, default 1): ").strip()
    model_map = {
        "1": "deepseek-r1:7b",
        "2": "deepseek-r1:14b",
        "3": "deepseek-r1:32b",

    }
    model_name = model_map.get(choice, "deepseek-r1:7b")

    # Initialize the agent
    print(f"🚀 Initializing Agent with {model_name}...")
    agent = NetworkAgent(model_name=model_name)
    print("✅ Agent initialized successfully!\n")

    # Test scenarios from the challenge PDF
    test_scenarios = [
        {
            "name": "Basic Network Check",
            "command": "Something seems wrong with the network, can you check it out?",
            "description": "General troubleshooting request"
        },
        {
            "name": "Device Connectivity Issue",
            "command": "Why isn't John's phone connecting? His device ID is 90400",
            "description": "Specific device connectivity problem"
        },
        {
            "name": "Comprehensive Health Check",
            "command": "Perform a full system health check and fix anything that's broken",
            "description": "Complete system assessment and repair"
        },
        {
            "name": "Network Startup",
            "command": "Start the network and check its status",
            "description": "Basic network operation sequence"
        },
        {
            "name": "Device Inventory Check",
            "command": "What devices are currently connected? Show me the network status",
            "description": "Inventory and status inquiry"
        }
    ]

    print("🧪 Starting Test Scenarios (Direct Ollama API)")
    print("=" * 70)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print(f"💬 Command: '{scenario['command']}'")
        print("-" * 70)

        # Execute the scenario
        start_time = time.time()
        response = agent.process_command(scenario["command"])
        execution_time = time.time() - start_time

        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📄 Response: {response}")
        print(f"✅ Scenario {i} completed!")
        print("=" * 70)

        # Pause between scenarios
        if i < len(test_scenarios):
            time.sleep(2)

    print(f"\n🎉 All {len(test_scenarios)} scenarios completed!")


def run_interactive_mode():
    """Run the agent in interactive mode"""
    model_name = input("Enter model name (default: deepseek-r1:7b): ").strip() or "deepseek-r1:7b"

    print(f'model name choosen :{model_name}')
    agent = NetworkAgent(model_name=model_name)
    agent.interactive_mode()


if __name__ == "__main__":
    print("🤖 LabLabee Technical Challenge - Intelligent Network Agent")
    print("📚 Using Direct Ollama API")
    print("=" * 60)

    while True:
        print("\nSelect mode:")
        print("1. Run all test scenarios")
        print("2. Interactive mode")
        print("3. Check available models")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            test_agent_scenarios()
        elif choice == "2":
            run_interactive_mode()
        elif choice == "3":
            list_available_models()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")