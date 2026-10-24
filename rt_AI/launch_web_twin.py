#!/usr/bin/env python3
"""launch_web_twin.py

Convenient launcher for the rt_AI GPU-Accelerated 3D Web Digital Twin.
Usage:
    python3 launch_web_twin.py [--port 8080] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import socket
import subprocess
import time

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src", "robot_control")

# Ensure package is discoverable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from web_twin.server import create_app, web


def free_port_if_in_use(port: int, host: str = "127.0.0.1"):
    """Check if port is already bound, and terminate the stale process cleanly."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                print(f"[WebTwin] Port {port} is currently in use. Freeing port...")
                subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.6)
    except Exception as e:
        print(f"[WebTwin] Note on checking port: {e}")


def main():
    parser = argparse.ArgumentParser(description="Launch rt_AI 3D Web Digital Twin")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    free_port_if_in_use(args.port)

    print("\n" + "=" * 72)
    print("  🤖 rt_AI GPU-ACCELERATED 3D WEB DIGITAL TWIN")
    print("  Kinematics: Stage 3 Test 2 Production Multi-Robot Stack")
    print(f"  Web Dashboard: http://localhost:{args.port}")
    print(f"  WebSocket:     ws://localhost:{args.port}/ws")
    print("=" * 72 + "\n")

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

