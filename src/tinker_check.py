"""Check if Tinker can be connected to. If so, list available models.

Usage:
    pdm run tinkercheck
"""

from __future__ import annotations

from tinker import ServiceClient


def main() -> None:
    service_client = ServiceClient()
    print("Connected to Tinker!\nAvailable models:")
    for item in service_client.get_server_capabilities().supported_models:
        if item.model_name:
            print("- " + item.model_name)


if __name__ == "__main__":
    main()
