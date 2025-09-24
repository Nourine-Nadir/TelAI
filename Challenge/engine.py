import requests
import time
from typing import Dict, Any, Optional
from functools import wraps


class NetworkSandbox:
    def __init__(self, base_url: str = "http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        # Common headers for API requests
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Helper method to handle all API requests with error handling"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Try to parse JSON, fall back to text if not JSON
            try:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'data': response.json(),
                    'message': 'Request successful'
                }
            except ValueError:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'data': response.text,
                    'message': 'Request successful (non-JSON response)'
                }

        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'status_code': getattr(e.response, 'status_code', None),
                'error': str(e),
                'message': f'Request failed: {e}'
            }

    def _retry_operation(self, operation_func, max_retries: int = 3, delay: float = 2.0,
                         operation_name: str = "operation") -> Dict[str, Any]:
        """Retry an operation until success or max retries reached"""
        for attempt in range(max_retries):
            result = operation_func()

            if result.get('success'):
                print(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result

            print(f"{operation_name} failed on attempt {attempt + 1}: {result.get('message', 'Unknown error')}")

            # Don't wait after the last attempt
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        print(f"{operation_name} failed after {max_retries} attempts")
        return result

    # Network Control Methods with retry capability
    def start_network(self, reason: str = "No reason", mode: str = "normal",
                      max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """Start the network with retry logic"""

        def _start_operation():
            payload = {"action": 'start'}
            return self._make_request('POST', '/network/start', json=payload)

        return self._retry_operation(_start_operation, max_retries, retry_delay, "Starting network")

    def connect_device(self, device_id: str, device_type: str = "phone", priority: str = "normal",
                       max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """Connect device with retry logic"""

        def _connect_operation():
            payload = {
                "device_id": device_id,
                "device_type": device_type,
                "connection_parameters": {
                    "priority": priority,
                    "qos_profile": "default"
                }
            }
            return self._make_request('POST', '/device/connect', json=payload)

        return self._retry_operation(_connect_operation, max_retries, retry_delay, f"Connecting device {device_id}")

    def disconnect_device(self, device_id: str, reason: str = "admin_action",
                         max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """Disconnect device with retry logic"""

        def _disconnect_operation():
            payload = {
                "device_id": device_id,
                "reason": reason
            }
            return self._make_request('POST', '/device/disconnect', json=payload)

        return self._retry_operation(_disconnect_operation, max_retries, retry_delay, f"Disconnecting device {device_id}")

    def stop_network(self, reason: str = "maintenance", graceful: bool = True,
                    max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """Stop the network with retry logic"""

        def _stop_operation():
            payload = {
                "action": 'stop'
            }
            return self._make_request('POST', '/network/stop', json=payload)

        return self._retry_operation(_stop_operation, max_retries, retry_delay, "Stopping network")

    # Methods without retry logic (keep original versions for backward compatibility)
    def disconnect_device_no_retry(self, device_id: str, reason: str = "user_request") -> Dict[str, Any]:
        """Disconnect device without retry logic (original version)"""
        payload = {
            "device_id": device_id,
            "reason": reason
        }
        return self._make_request('POST', '/device/disconnect', json=payload)

    def stop_network_no_retry(self, reason: str = "maintenance", graceful: bool = True) -> Dict[str, Any]:
        """Stop the network without retry logic (original version)"""
        payload = {
            "action": 'stop'
        }
        return self._make_request('POST', '/network/stop', json=payload)

    # Other existing methods
    def get_network_status(self) -> Dict[str, Any]:
        return self._make_request('GET', '/network/status')

    def get_system_logs(self) -> Dict[str, Any]:
        return self._make_request('GET', '/logs')

    def is_network_running(self) -> bool:
        """Check if the network service is currently running"""
        status = self.get_network_status()
        return status['success'] and status.get('success', {}) == True

    def wait_for_operation(self, timeout: int = 30, check_interval: int = 2) -> bool:
        """Wait for network operations to stabilize"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_network_running():
                return True
            time.sleep(check_interval)
        return False
