# discover_parameters.py
from engine import NetworkSandbox
import requests


def discover_required_parameters():
    """Try to discover what parameters the API expects"""
    base_url = "http://localhost:8080/api/v1"

    # Test different payloads for start endpoint
    test_payloads = [
        {},  # Empty object
        {"action": "start"},
        {"command": "start", "parameters": {}},
        {"operation": "start_network"},
        {"region": "all", "mode": "normal"}
    ]

    print("🔍 Discovering required parameters for /network/start...")

    for i, payload in enumerate(test_payloads):
        try:
            response = requests.post(f"{base_url}/network/start", json=payload)
            print(f"Payload {i + 1}: {payload}")
            print(f"Status: {response.status_code}")
            if response.status_code != 400:  # If it's not a validation error
                print(f"Response: {response.text[:200]}")
            print("-" * 50)
        except Exception as e:
            print(f"Payload {i + 1}: ERROR - {e}")
            print("-" * 50)


if __name__ == "__main__":
    discover_required_parameters()