"""Prompt templates and datum construction for poetry training."""

from __future__ import annotations

from tinker import Datum, ModelInput, TensorData

from qwen3_utils import Qwen3Message, Qwen3Renderer, Qwen3Role

# Consistent system prompt across all phases
SYSTEM_PROMPT = "You are a poet."

DESCRIPTION_PROMPT_TEMPLATE = """Read this poem titled "{title}":

{poem}

Write a description (under 150 words) that would allow a poet to write this poem without seeing it.

Your description should capture:
- The core emotion or insight
- The imagery and sensory details
- The tone and voice
- The arc or movement of the poem

Important: Describe the poem's essence, not its exact words. Do not quote or closely paraphrase any lines."""

ECHO_PROMPT_TEMPLATE = """Write a poem with this title and description.

Title: {title}
Description: {description}

Respond with the title and poem in this format:
Title: <title>
Poem: <poem>"""

CREATIVE_PROMPT_TEMPLATE = """Write a poem based on this description. Choose an appropriate title.

Description: {description}

Respond with the title and poem in this format:
Title: <title>
Poem: <poem>"""


def build_description_request(
    renderer: Qwen3Renderer,
    title: str,
    poem: str,
) -> ModelInput:
    """Build a prompt for generating descriptions.

    Args:
        renderer: The Qwen3 renderer for tokenization.
        title: The poem title.
        poem: The poem content.

    Returns:
        ModelInput ready for sampling.
    """
    user_content = DESCRIPTION_PROMPT_TEMPLATE.format(title=title, poem=poem)

    messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]

    return renderer.build_generation_prompt(messages)


def build_poem_request(
    renderer: Qwen3Renderer,
    title: str,
    description: str,
) -> ModelInput:
    """Build a prompt for generating poems (for evaluation).

    Args:
        renderer: The Qwen3 renderer for tokenization.
        title: The poem title (used in Echo variant prompt).
        description: The description to generate from.

    Returns:
        ModelInput ready for sampling.
    """
    user_content = ECHO_PROMPT_TEMPLATE.format(title=title, description=description)

    messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]

    return renderer.build_generation_prompt(messages)


def build_training_datum(
    renderer: Qwen3Renderer,
    title: str,
    description: str,
    poem: str,
    include_title: bool,
) -> Datum:
    """Create a training datum with proper loss masking.

    Args:
        renderer: The Qwen3 renderer for tokenization.
        title: The poem title.
        description: The candidate description.
        poem: The poem content.
        include_title: If True, use Echo format (title in prompt).
                      If False, use Creative format (no title in prompt).

    Returns:
        Datum with context tokens (weight=0) and target tokens (weight=1).
    """
    if include_title:
        user_content = ECHO_PROMPT_TEMPLATE.format(title=title, description=description)
    else:
        user_content = CREATIVE_PROMPT_TEMPLATE.format(description=description)

    # Target format is the same for both variants
    target_content = f"Title: {title}\nPoem: {poem}"

    messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
        Qwen3Message(role=Qwen3Role.ASSISTANT, content=target_content),
    ]

    context_messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]
    context_tokens = renderer.build_generation_prompt(context_messages).to_ints()
    full_tokens = renderer.build_training_sequence(messages)

    # Target = everything after context; weights mask context (0) vs target (1)
    target_tokens = full_tokens[len(context_tokens) :]
    weights = [0.0] * len(context_tokens) + [1.0] * len(target_tokens)

    return Datum(
        model_input=ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=full_tokens, dtype="int64", shape=[len(full_tokens)]
            ),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )
