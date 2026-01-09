"""Interactive REPL for chatting with Qwen3 models via Tinker."""

from __future__ import annotations

from dataclasses import dataclass

import chz
from tinker import SamplingClient, SamplingParams, ServiceClient

from qwen3_utils import Qwen3Message, Qwen3Renderer, Qwen3Role, get_qwen3_tokenizer


def format_response(message: Qwen3Message) -> str:
    """Format a Message for display, showing reasoning if present."""
    if message.reasoning:
        return f"<think> {message.reasoning} </think>\n\n{message.content}"
    return message.content


# Supported models
SUPPORTED_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

# Max context length for all models
MAX_CONTEXT_LENGTH = 32_768


@chz.chz
@dataclass
class ReplConfig:
    model_name: str | None = None
    """Base model name (e.g., 'Qwen/Qwen3-8B'). If omitted, you'll be prompted."""

    checkpoint: str | None = None
    """Optional checkpoint path/name for sampling (e.g., 'tinker://...')."""


def select_model(service_client: ServiceClient) -> str:
    """Display available models and let user select one."""
    capabilities = service_client.get_server_capabilities()
    available = {m.model_name for m in capabilities.supported_models if m.model_name}

    # Filter to supported models that are available on the server
    models = [m for m in SUPPORTED_MODELS if m in available]

    if not models:
        raise RuntimeError(
            f"No supported models available. Expected one of: {SUPPORTED_MODELS}"
        )

    print("Supported models:")
    for i, model in enumerate(models):
        print(f"  [{i}] {model}")

    while True:
        choice = input(f"\nSelect model [0-{len(models) - 1}] (default: 0): ").strip()
        if choice == "":
            return models[0]
        try:
            idx = int(choice)
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def validate_supported_base_model(model_name: str) -> None:
    """Raise ValueError if model_name is not in SUPPORTED_MODELS."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Expected one of: {SUPPORTED_MODELS}"
        )


def infer_base_model_from_checkpoint(
    service_client: ServiceClient, checkpoint: str
) -> str | None:
    """Try to infer base model from checkpoint via REST API. Returns None on failure."""
    try:
        rest_client = service_client.create_rest_client()
        weights_info = rest_client.get_weights_info_by_tinker_path(checkpoint).result()
        base_model = weights_info.base_model
        if base_model in SUPPORTED_MODELS:
            return base_model
        print(f"Checkpoint base model '{base_model}' not in supported models.")
        return None
    except Exception as e:
        print(f"Could not infer base model from checkpoint: {e}")
        return None


def resolve_base_model(service_client: ServiceClient, config: ReplConfig) -> str:
    """Resolve the base model for tokenizer/renderer.

    Priority:
    1. Explicit --model_name (validated)
    2. If checkpoint provided, try REST inference
    3. Fall back to interactive selection
    """
    # Explicit model_name always wins
    if config.model_name is not None:
        validate_supported_base_model(config.model_name)
        return config.model_name

    # Try to infer from checkpoint if provided
    if config.checkpoint:
        inferred = infer_base_model_from_checkpoint(service_client, config.checkpoint)
        if inferred:
            print(f"Inferred base model from checkpoint: {inferred}")
            return inferred
        print("Falling back to model selection...")

    # Interactive selection
    return select_model(service_client)


def create_sampling_client(
    service_client: ServiceClient, config: ReplConfig, *, base_model: str
) -> SamplingClient:
    """Create a sampling client from checkpoint or base model."""
    if config.checkpoint:
        return service_client.create_sampling_client(model_path=config.checkpoint)
    return service_client.create_sampling_client(base_model=base_model)


def main() -> None:
    config = chz.entrypoint(ReplConfig)
    print("Connecting to Tinker...")
    service_client = ServiceClient()

    base_model = resolve_base_model(service_client, config)
    if config.checkpoint:
        print(f"\nLoading checkpoint: {config.checkpoint}")
        print(f"Tokenizer base model: {base_model}")
    else:
        print(f"\nLoading {base_model}...")

    # Get tokenizer and renderer
    renderer = Qwen3Renderer(get_qwen3_tokenizer(base_model))

    # Create sampling client
    sampling_client = create_sampling_client(
        service_client, config, base_model=base_model
    )

    # Sampling parameters (Qwen3 recommended: temp=0.6, top_p=0.95, top_k=20)
    # Using larger max_tokens to allow for extended thinking
    sampling_params = SamplingParams(
        max_tokens=4096,
        stop=renderer.get_stop_sequences(),
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    # Chat history and debug mode
    messages: list[Qwen3Message] = []
    debug_mode = False

    print(f"\n{'=' * 60}")
    header = f"Tinker REPL — {base_model}"
    if config.checkpoint:
        header += f" ({config.checkpoint})"
    print(header)
    print("Commands: /clear, /debug, /exit")
    print(f"{'=' * 60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            print("(history cleared)\n")
            continue

        if user_input.lower() == "/debug":
            debug_mode = not debug_mode
            print(f"(debug mode: {'on' if debug_mode else 'off'})\n")
            continue

        # Add user message to history
        messages.append(Qwen3Message(role=Qwen3Role.USER, content=user_input))

        # Build prompt and sample
        prompt = renderer.build_generation_prompt(messages)

        if debug_mode:
            # Display the prompt being sent
            prompt_text = renderer.build_generation_prompt_text(messages)
            prompt_context_used = (prompt.length / MAX_CONTEXT_LENGTH) * 100
            print(f"\n{'─' * 60}")
            print(
                f"DEBUG: Context ({prompt.length} tokens, {prompt_context_used:.1f}%)"
            )
            print(f"{'─' * 60}")
            print(prompt_text)
            print(f"{'─' * 60}")

        output = sampling_client.sample(
            prompt,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        # Parse response (extracts reasoning into separate field)
        response_tokens = output.sequences[0].tokens
        assistant_message, _parse_success = renderer.parse_response(response_tokens)

        # Display formatted response
        print(f"\nAssistant: {format_response(assistant_message)}")

        # Check context usage and warn if high
        total_tokens = prompt.length + len(response_tokens)
        total_context_used = (total_tokens / MAX_CONTEXT_LENGTH) * 100

        if debug_mode:
            print(f"\n{'─' * 60}")
            print(f"DEBUG: Context ({total_tokens} tokens, {total_context_used:.1f}%)")
            print(f"{'─' * 60}")

        if total_context_used > 80:
            print(
                f"\n⚠️  Warning: Context {total_context_used:.0f}% full. Use /clear to reset."
            )

        print()

        # Store message in history (reasoning will be stripped on next render)
        messages.append(assistant_message)


if __name__ == "__main__":
    main()
