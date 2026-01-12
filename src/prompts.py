"""Prompt templates and datum construction for poetry training."""

from __future__ import annotations

from tinker import Datum, ModelInput, TensorData

from qwen3_utils import Qwen3Message, Qwen3Renderer, Qwen3Role

# Consistent system prompt across all phases
SYSTEM_PROMPT = "You are a poet."

DESCRIPTION_GENERATION_TEMPLATE = """Read this poem titled "{title}":

{poem}

In about 100 words, describe what this poem is really about. 

Write the description like you would write notes or a brief sketch. It should capture:
- Core emotions or insights
- Any imagery or sensory details
- The mood, tone, or voice
- Structure or progression in the poem

Important: Describe the poem's essence in a way that would allow a poet to create it without seeing it. However, do not quote or copy any lines."""

POEM_PROMPT_TEMPLATE = """Write a poem based on this description. Choose an appropriate title.

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
    user_content = DESCRIPTION_GENERATION_TEMPLATE.format(title=title, poem=poem)

    messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]

    return renderer.build_generation_prompt(messages)


def build_poem_request(
    renderer: Qwen3Renderer,
    description: str,
) -> ModelInput:
    """Build a prompt for generating poems (for evaluation).

    Args:
        renderer: The Qwen3 renderer for tokenization.
        description: The description to generate from.

    Returns:
        ModelInput ready for sampling.
    """
    user_content = POEM_PROMPT_TEMPLATE.format(description=description)

    messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]

    return renderer.build_generation_prompt(messages)


def build_training_datum(
    renderer: Qwen3Renderer,
    description: str,
    title: str,
    poem: str,
) -> Datum:
    """Create a training datum with proper loss masking.

    Includes static reasoning tokens for reasoning models, but masks them from loss.

    Args:
        renderer: The Qwen3 renderer for tokenization.
        description: The candidate description (input).
        title: The poem title (target).
        poem: The poem content (target).

    Returns:
        Datum with context tokens (weight=0), reasoning tokens (weight=0),
        and target content tokens (weight=1).
    """
    user_content = POEM_PROMPT_TEMPLATE.format(description=description)

    # Target format: Title + Poem
    target_content = f"Title: {title}\nPoem: {poem}"

    context_messages = [
        Qwen3Message(role=Qwen3Role.SYSTEM, content=SYSTEM_PROMPT),
        Qwen3Message(role=Qwen3Role.USER, content=user_content),
    ]
    context_tokens = renderer.build_generation_prompt(context_messages).to_ints()

    # Build assistant turn tokens (context_tokens already includes <|im_start|>assistant\n)
    # This is just sample reasoning content -- in practice, the model will generate its own reasoning.
    reasoning_content = "I am thinking and writing a poem."
    assistant_reasoning = f"<think>\n{reasoning_content}\n</think>\n"
    assistant_content = target_content + "<|im_end|>"

    assistant_reasoning_tokens = renderer.tokenizer.encode(
        assistant_reasoning, add_special_tokens=False
    )
    assistant_content_tokens = renderer.tokenizer.encode(assistant_content, add_special_tokens=False)

    # Build full sequence: context (includes assistant header) + reasoning + content
    full_tokens = context_tokens + assistant_reasoning_tokens + assistant_content_tokens

    # Compute weights: context (0), reasoning (0), content (1)
    weights = (
        [0.0] * len(context_tokens)
        + [0.0] * len(assistant_reasoning_tokens)
        + [1.0] * len(assistant_content_tokens)
    )

    return Datum(
        model_input=ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=full_tokens, dtype="int64", shape=[len(full_tokens)]
            ),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )
