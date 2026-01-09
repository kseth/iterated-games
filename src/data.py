"""Data loading utilities for poetry training.

Loads poems from JSONL format with title and content fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class Poem:
    """A poem with title and content."""

    title: str
    content: str


def load_poems(path: str | Path) -> list[Poem]:
    """Load poems from a JSONL file.

    Expected format: {"title": "...", "content": "..."}

    Args:
        path: Path to the JSONL file.

    Returns:
        List of Poem objects.
    """
    df = pd.read_json(path, lines=True)
    return [
        Poem(title=row["title"], content=row["content"]) for _, row in df.iterrows()
    ]


def filter_poems_by_length(
    poems: list[Poem],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
) -> list[Poem]:
    """Filter out poems that exceed the token limit.

    Args:
        poems: List of poems to filter.
        tokenizer: Tokenizer to use for counting tokens.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        Filtered list of poems under the token limit.
    """
    filtered = []
    for poem in poems:
        tokens = tokenizer.encode(poem.content, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            filtered.append(poem)
    return filtered
