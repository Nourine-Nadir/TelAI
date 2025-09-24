import json
from engine import NetworkSandbox

def test_sandbox():
    sandbox = NetworkSandbox()
    device_id = "5432111"

    print("=== Testing 5G Network Sandbox API with Full Retry Logic ===")
    print(f"Using base URL: {sandbox.base_url}")

    # Test starting the network with retry
    print("\n1. Starting network (with retry)...")
    result = sandbox.start_network(max_retries=5, retry_delay=2.0)
    print(f"Final result: {json.dumps(result, indent=2)}")

    if not result['success']:
        print("Network failed to start after retries. Aborting test.")
        return

    # Wait for network to stabilize
    print("\n2. Waiting for network to stabilize...")
    if sandbox.wait_for_operation(timeout=30, check_interval=2):
        print("Network is stable")
    else:
        print("Network failed to stabilize within timeout")

    # Connect a device with retry
    print(f"\n3. Connecting device {device_id} (with retry)...")
    connect_result = sandbox.connect_device(device_id, max_retries=5, retry_delay=2.0)
    print(f"Final connect result: {json.dumps(connect_result, indent=2)}")

    # Check status after connection
    print("\n4. Status after connection...")
    status_after_connect = sandbox.get_network_status()
    print(f"Status: {json.dumps(status_after_connect, indent=2)}")

    # Disconnect the device with retry
    print(f"\n5. Disconnecting device {device_id} (with retry)...")
    disconnect_result = sandbox.disconnect_device(device_id, max_retries=5, retry_delay=2.0)
    print(f"Final disconnect result: {json.dumps(disconnect_result, indent=2)}")

    # Check status after disconnection
    print("\n6. Status after disconnection...")
    status_after_disconnect = sandbox.get_network_status()
    print(f"Status: {json.dumps(status_after_disconnect, indent=2)}")

    # Get system logs
    print("\n7. Retrieving system logs...")
    logs = sandbox.get_system_logs()
    print(f"Logs: {json.dumps(logs, indent=2)}")

    # Stop the network with retry
    print("\n8. Stopping network (with retry)...")
    stop_result = sandbox.stop_network(max_retries=5, retry_delay=2.0)
    print(f"Final stop result: {json.dumps(stop_result, indent=2)}")


    print("\n=== Test completed ===")


if __name__ == "__main__":
    test_sandbox()