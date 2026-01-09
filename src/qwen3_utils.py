"""Qwen3 tokenizer and renderer for Tinker.

Simplified from tinker-cookbook, focused on Qwen3-8B and Qwen3-32B text models.
Uses HuggingFace transformers for tokenization.

Reference: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers/qwen3.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING, Any

from tinker import ModelInput

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    Tokenizer = PreTrainedTokenizer
else:
    Tokenizer = Any


class Qwen3Role(StrEnum):
    """Chat message roles for Qwen3."""

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


@dataclass
class Qwen3Message:
    """Qwen3 chat message with role, content, and optional reasoning."""

    role: Qwen3Role
    content: str = ""
    reasoning: str | None = None  # Thinking/reasoning for assistant messages

    def __post_init__(self) -> None:
        """Validate that reasoning is only set for assistant messages."""
        if self.reasoning is not None and self.role != Qwen3Role.ASSISTANT:
            raise ValueError(
                f"reasoning can only be set for assistant messages, got role={self.role}"
            )


@cache
def get_qwen3_tokenizer(model_name: str) -> Tokenizer:
    """Load tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _parse_thinking(raw_content: str) -> tuple[str | None, str]:
    """Parse raw model output into (reasoning, content) tuple."""
    if "<think>" in raw_content and "</think>" in raw_content:
        think_start = raw_content.index("<think>") + len("<think>")
        think_end = raw_content.rindex("</think>")
        reasoning = raw_content[think_start:think_end].strip()
        content = raw_content[think_end + len("</think>") :].strip()
        return reasoning, content
    return None, raw_content.strip()


class Qwen3Renderer:
    """
    Renderer for Qwen3 models with thinking enabled.

    Matches HuggingFace's Qwen3 chat template behavior (enable_thinking=True).

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        <think>
        ...thinking...
        </think>
        Hi there!<|im_end|>
    """

    def __init__(
        self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True
    ) -> None:
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips reasoning
                from assistant messages in history. Matches HuggingFace behavior.
        """
        self.tokenizer = tokenizer
        self.strip_thinking_from_history = strip_thinking_from_history

    def _render_message(self, idx: int, message: Qwen3Message) -> list[int]:
        """Render a single message to tokens.

        Args:
            idx: Message index (0-based)
            message: The message to render
        """
        maybe_newline = "\n" if idx > 0 else ""
        header = f"{maybe_newline}<|im_start|>{message.role}\n"

        # Build content string
        if message.role == Qwen3Role.ASSISTANT and message.reasoning:
            if self.strip_thinking_from_history:
                # Strip reasoning from history
                output_content = message.content
            else:
                # Include reasoning wrapped in <think> tags
                output_content = (
                    f"<think>\n{message.reasoning}\n</think>\n{message.content}"
                )
        else:
            output_content = message.content

        # Concatenate and encode together to ensure correct tokenization
        full_str = header + output_content + "<|im_end|>"
        return self.tokenizer.encode(full_str, add_special_tokens=False)

    def _build_prompt_tokens(self, messages: list[Qwen3Message]) -> list[int]:
        """Build prompt tokens from a list of messages."""
        tokens: list[int] = []

        # Render all existing messages
        for idx, message in enumerate(messages):
            tokens.extend(self._render_message(idx, message))

        # Add prompt for new assistant response
        # Model will naturally generate <think>...</think> when thinking is enabled
        new_turn_idx = len(messages)
        maybe_newline = "\n" if new_turn_idx > 0 else ""
        new_turn = f"{maybe_newline}<|im_start|>assistant\n"
        tokens.extend(self.tokenizer.encode(new_turn, add_special_tokens=False))

        return tokens

    def build_generation_prompt(self, messages: list[Qwen3Message]) -> ModelInput:
        """Build a prompt for generation from a list of messages.

        Matches HuggingFace's Qwen3 chat template (enable_thinking=True).
        The model naturally generates <think>...</think> blocks.
        """
        return ModelInput.from_ints(self._build_prompt_tokens(messages))

    def build_generation_prompt_text(self, messages: list[Qwen3Message]) -> str:
        """Build prompt as text (for debugging)."""
        return self.tokenizer.decode(self._build_prompt_tokens(messages))

    def build_training_sequence(self, messages: list[Qwen3Message]) -> list[int]:
        """Build a complete conversation for training (all messages fully rendered).

        Unlike build_generation_prompt which leaves the assistant turn open,
        this renders all messages including a complete assistant response with
        the closing <|im_end|> token.

        Args:
            messages: List of messages forming the complete conversation.
                      Should include the assistant response as the final message.

        Returns:
            List of token IDs for the full conversation.
        """
        tokens: list[int] = []

        for idx, message in enumerate(messages):
            maybe_newline = "\n" if idx > 0 else ""
            header = f"{maybe_newline}<|im_start|>{message.role}\n"

            # For training, use content directly (no reasoning handling)
            content = message.content

            full_str = header + content + "<|im_end|>"
            tokens.extend(self.tokenizer.encode(full_str, add_special_tokens=False))

        return tokens

    @property
    def _end_token(self) -> int:
        """Get the <|im_end|> token ID."""
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, (
            f"Expected single token for <|im_end|>, got {len(tokens)}"
        )
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        """Return stop token(s) for sampling."""
        return [self._end_token]

    def parse_response(self, response: list[int]) -> tuple[Qwen3Message, bool]:
        """Parse sampled tokens into a Qwen3Message.

        Extracts reasoning into the `reasoning` field and final response into `content`.
        Returns (message, success) tuple.
        """
        end_token = self._end_token
        count = response.count(end_token)

        if count == 0:
            # No stop token found (e.g., ran out of tokens)
            raw_content = self.tokenizer.decode(response)
            reasoning, content = _parse_thinking(raw_content)
            return Qwen3Message(
                role=Qwen3Role.ASSISTANT, content=content, reasoning=reasoning
            ), False
        elif count == 1:
            # Normal case: decode up to stop token
            raw_content = self.tokenizer.decode(response[: response.index(end_token)])
            reasoning, content = _parse_thinking(raw_content)
            return Qwen3Message(
                role=Qwen3Role.ASSISTANT, content=content, reasoning=reasoning
            ), True
        else:
            raise ValueError(
                f"Expected at most 1 stop token, got {count}. "
                "Check your stop sequences in sampling params."
            )
