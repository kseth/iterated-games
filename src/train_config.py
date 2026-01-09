"""Training configuration for poetry training."""

from __future__ import annotations

import chz


@chz.chz
class Config:
    """Training configuration."""

    dataset_path: str = "example_data/tiny_poems.jsonl"
    """Path to poems JSONL file."""

    model_name: str = "Qwen/Qwen3-8B"
    """Base model name."""

    log_path: str = "logs/poetry-train"
    """Base directory for checkpoints and logs."""

    run_name: str | None = None
    """Run name for this training run. Auto-generated if not provided."""

    resume_from: str | None = None
    """Tinker checkpoint name to resume from (e.g., 'whimsical-whitman-0042.step-000050')."""

    max_iterations: int = 3
    """Number of full passes through dataset."""

    num_candidates: int = 4
    """N: candidates to generate per poem."""

    top_k: int = 2
    """K: winners to keep per poem."""

    overlap_weight: float = 0.1
    """Weight for overlap penalty in scoring."""

    score_batch_size: int = 32
    """Poems per batch during generate/score."""

    train_batch_size: int = 64
    """Training examples per batch."""

    saves_per_iteration: float = 2.0
    """How many checkpoints to save per iteration (0=disabled, 0.5=every 2 iters, 2=twice per iter)."""

    evals_per_iteration: float = 5.0
    """How often to resample per iteration (0=disabled).

    Resampling refreshes the sampling client to use the latest model weights; when we
    resample, we also generate an eval sample with the fresh weights.
    """
