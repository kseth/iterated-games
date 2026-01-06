import tinker


def main() -> None:
    service_client = tinker.ServiceClient()
    print("Connected to Tinker!\nAvailable models:")
    for item in service_client.get_server_capabilities().supported_models:
        if item.model_name:
            print("- " + item.model_name)


if __name__ == "__main__":
    main()
