#!/usr/bin/env python3
"""
Shared utilities for KOTOR Agents
"""

import requests
import time

def check_server(port, timeout=5):
    """Check if llama.cpp server is running on given port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def wait_for_servers(ports, max_wait=30):
    """Wait for all servers to be ready."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if all(check_server(port) for port in ports):
            return True
        time.sleep(1)
    return False