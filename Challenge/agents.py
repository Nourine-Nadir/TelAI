import ollama
from engine import NetworkSandbox
import json
import re
import time


class NetworkAgent:
    def __init__(self, model_name: str = "deepseek-coder:14b"):
        """Initialize the intelligent network agent with direct Ollama API"""
        self.sandbox = NetworkSandbox(base_url="http://localhost:8080/api/v1")
        self.model_name = model_name

        # Test the model availability
        self._check_model_available()

        # Available tools mapping
        self.tools = {
            "start_network": {
                "function": self.sandbox.start_network,
                "description": "Start or restart the 5G network service"
            },
            "stop_network": {
                "function": self.sandbox.stop_network,
                "description": "Stop or shutdown the 5G network service"
            },
            "get_network_status": {
                "function": self.sandbox.get_network_status,
                "description": "Get current network health status and metrics"
            },
            "connect_device": {
                "function": self.sandbox.connect_device,
                "description": "Connect a specific device to the network",
                "requires_device_id": True
            },
            "disconnect_device": {
                "function": self.sandbox.disconnect_device,
                "description": "Disconnect a specific device from the network",
                "requires_device_id": True
            },
            "get_system_logs": {
                "function": self.sandbox.get_system_logs,
                "description": "Get system logs and diagnostics"
            }

        }


    def _check_model_available(self):
        """Check if the specified model is available"""
        try:
            models = ollama.list()
            available_models = [model['model'] for model in models['models']]
            if self.model_name not in available_models:
                print(f"⚠️  Model {self.model_name} not found. Available models: {available_models}")
                # Try to pull the model
                print(f"📥 Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
                print(f"✅ Model {self.model_name} is now available")
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            raise

    def _extract_device_id(self, query: str) -> str:
        """Extract device ID from natural language query"""
        patterns = [
            r'device\s+([a-zA-Z0-9-]+)',
            r'ID\s+([a-zA-Z0-9-]+)',
            r'phone\s+([a-zA-Z0-9-]+)',
            r'(\d{4,})',  # Sequence of 4+ digits
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return None  # Default fallback


    def _generate_plan(self, user_query: str) -> dict:
        """Use Ollama to generate an action plan based on the user query"""

        tools_description = "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()])

        prompt = f"""You are an intelligent 5G network operations agent. Analyze the engineer's request and create a step-by-step action plan.

Available Tools:
{tools_description}

Engineer's Request: "{user_query}"

Based on the request, determine which actions to take and in what order. Consider:
1. What is the main goal of the request?
2. What information do we need to gather first?
3. What actions might solve the problem?
4. What verification steps are needed?

Respond with a JSON format only:
{{
    "reasoning": "Brief explanation of your approach",
    "actions": ["action1", "action2", "action3"],
    "device_id": "extracted_id_or_null",
    "expected_outcome": "What we expect to achieve"
}}

Example:
{{
    "reasoning": "The engineer reports a network issue. I'll first check the status, then look for anomalies, and finally check logs for details.",
    "actions": ["get_network_status", "check_anomalies", "get_system_logs"],
    "device_id": null,
    "expected_outcome": "Identify the root cause of the network issue"
}}

Now analyze this request:"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent planning
                    "top_p": 0.9,
                    "num_predict": 100000
                }
            )

            # Extract JSON from response
            response_text = response['response']

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                return plan
            else:
                # Fallback plan if JSON parsing fails
                return {
                    "reasoning": "Default plan for network troubleshooting",
                    "actions": ["get_network_status", "check_anomalies", "get_system_logs"],
                    "device_id": None,
                    "expected_outcome": "Basic network assessment"
                }

        except Exception as e:
            print(f"❌ Error generating plan: {e}")
            # Fallback plan
            return {
                "reasoning": "Error in planning, using default troubleshooting approach",
                "actions": ["get_network_status"],
                "device_id": None,
                "expected_outcome": "Basic network status check"
            }

    def _execute_actions(self, actions: list, device_id: str = None) -> dict:
        """Execute the planned actions and collect results"""
        results = {}

        for action in actions:
            if action in self.tools:
                try:
                    print(f"🔄 Executing: {action}")

                    # Handle device-specific actions
                    if self.tools[action].get("requires_device_id") and device_id:
                        result = self.tools[action]["function"](device_id)
                    else:
                        result = self.tools[action]["function"]()

                    results[action] = {
                        "success": True,
                        "data": result
                    }
                    print(f"✅ {action} completed successfully")

                except Exception as e:
                    results[action] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"❌ {action} failed: {e}")
            else:
                results[action] = {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                print(f"❌ Unknown action: {action}")

            # Brief pause between actions
            time.sleep(0.5)
        print(f'results {results}')
        return results

    def _generate_response(self, user_query: str, plan: dict, results: dict) -> str:
        """Use Ollama to generate a natural language response based on the results"""

        # Format results for the prompt
        results_summary = ""
        for action, result in results.items():
            if result["success"]:
                results_summary += f"✓ {action}: Success\n"
                # Include key data points if available
                if "data" in result and isinstance(result["data"], dict):
                    if "success" in result["data"]:
                        results_summary += f"  Status: {result['data']['success']}\n"
            else:
                results_summary += f"✗ {action}: Failed - {result['error']}\n"

        prompt = f"""You are a helpful 5G network operations assistant. You've just executed a series of actions based on an engineer's request.

Original Request: "{user_query}"

Plan Executed:
- Reasoning: {plan['reasoning']}
- Actions: {', '.join(plan['actions'])}
- Expected Outcome: {plan['expected_outcome']}

Execution Results:
{results_summary}

Based on these results, provide a concise, helpful response to the engineer. Include:
1. What you did to address their request
2. Key findings from the execution
3. Any issues encountered
4. Recommendations or next steps if applicable

Keep the response professional but friendly, focused on being helpful to the network engineer."""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,  # Slightly higher for more natural responses
                    "top_p": 0.9,
                    "num_predict": 10000
                }
            )

            return response['response']

        except Exception as e:
            return f"I've completed the requested actions. Here's a summary of what was done:\n\n{results_summary}\n\nError generating detailed response: {e}"

    def process_command(self, engineer_command: str) -> str:
        """Process a natural language command from a network engineer"""
        print(f"\n🎯 Processing command: '{engineer_command}'")
        print("=" * 60)

        try:
            # Step 1: Generate action plan using Ollama
            print("🧠 Generating action plan...")
            plan = self._generate_plan(engineer_command)

            print(f"📋 Plan: {plan['reasoning']}")
            print(f"🔧 Actions: {', '.join(plan['actions'])}")

            # Step 2: Extract device ID if needed
            device_id = plan.get('device_id')
            print(f'device ID {device_id}')
            if not device_id:
                device_id = self._extract_device_id(engineer_command)
                if device_id:
                    print(f"📱 Extracted device ID: {device_id}")

            # Step 3: Execute the planned actions
            print("⚡ Executing actions...")
            results = self._execute_actions(plan['actions'], device_id)

            # Step 4: Generate natural language response
            print("💬 Generating response...")
            response = self._generate_response(engineer_command, plan, results)

            print("=" * 60)
            return response

        except Exception as e:
            error_msg = f"❌ Error processing command: {str(e)}"
            print(error_msg)
            return error_msg

    def interactive_mode(self):
        """Start an interactive session with the agent"""
        print(f"\n🤖 Intelligent Network Agent - Interactive Mode")
        print(f"📚 Model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the session")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nEngineer: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Ending session. Goodbye!")
                    break

                if not user_input:
                    continue

                start_time = time.time()
                response = self.process_command(user_input)
                execution_time = time.time() - start_time

                print(f"\nAgent: {response}")
                print(f"⏱️  Response time: {execution_time:.2f} seconds")

            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

