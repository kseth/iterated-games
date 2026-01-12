"""Scoring utilities for poetry training.

Implements loss computation for selecting the best candidate descriptions.
"""

from __future__ import annotations

from tinker.types import LossFnOutput
from tinker import Datum


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
