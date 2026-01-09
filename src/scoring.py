"""Scoring utilities for poetry training.

Implements IDF-weighted overlap calculation and loss computation
for selecting the best candidate descriptions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import cache

from snowballstemmer import stemmer as _create_stemmer  # type: ignore[import-untyped]

from tinker.types import LossFnOutput
from tinker import Datum

_english_stemmer = _create_stemmer("english")


@cache
def stem(word: str) -> str:
    """Stem a word using English Porter stemmer (cached for performance)."""
    result: str = _english_stemmer.stemWord(word)
    return result


def tokenize(text: str) -> list[str]:
    """Tokenize and stem text for IDF/overlap calculation.

    Args:
        text: Input text to tokenize.

    Returns:
        List of stemmed tokens.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return [stem(w) for w in words]


def build_idf_table(poems: list[str]) -> dict[str, float]:
    """Build IDF table from a corpus of poems.

    IDF = log(N / doc_freq) - rare stems get higher weight.

    Args:
        poems: List of poem texts (corpus).

    Returns:
        Dictionary mapping stems to IDF scores.
    """
    n = len(poems)
    doc_freq: Counter[str] = Counter()
    for poem in poems:
        doc_freq.update(set(tokenize(poem)))

    return {stem_word: math.log(n / freq) for stem_word, freq in doc_freq.items()}


def compute_overlaps(
    descriptions: list[str],
    poem: str,
    idf_table: dict[str, float],
) -> list[float]:
    """Compute IDF-weighted overlap for multiple descriptions against one poem.

    What fraction of each description's "information weight" comes from poem words?

    Args:
        descriptions: List of candidate descriptions.
        poem: The original poem text.
        idf_table: Pre-computed IDF table from corpus.

    Returns:
        List of overlap fractions, each from 0.0 (no overlap) to 1.0 (complete overlap).
    """
    poem_stems = set(tokenize(poem))

    results = []
    for desc in descriptions:
        desc_stems = set(tokenize(desc))
        total_weight = sum(idf_table.get(s, 1.0) for s in desc_stems)
        overlap_weight = sum(idf_table.get(s, 1.0) for s in desc_stems & poem_stems)
        results.append(overlap_weight / total_weight if total_weight > 0 else 0.0)

    return results


def compute_mean_nll(
    loss_fn_output: LossFnOutput,
    datum: Datum,
) -> float:
    """Compute mean negative log-likelihood from forward output.

    Args:
        loss_fn_output: Output from training_client.forward() containing logprobs.
        datum: The datum that was scored, containing loss_fn_inputs with weights.

    Returns:
        Mean NLL over weighted tokens (lower is better).
    """
    logprobs_data = loss_fn_output["logprobs"].tolist()
    weights_data = datum.loss_fn_inputs["weights"].tolist()

    assert len(logprobs_data) == len(weights_data), (
        f"Length mismatch: logprobs={len(logprobs_data)}, weights={len(weights_data)}"
    )

    total_loss = 0.0
    total_weight = 0.0
    for logprob, weight in zip(logprobs_data, weights_data, strict=True):
        total_loss += -logprob * weight
        total_weight += weight

    return total_loss / total_weight if total_weight > 0 else 0.0
