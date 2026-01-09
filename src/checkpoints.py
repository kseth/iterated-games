"""Checkpoint utilities for poetry training.

Handles saving and loading model state, optimizer state, and loop metadata.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from chz import asdict
from train_config import Config

if TYPE_CHECKING:
    from tinker import TrainingClient

logger = logging.getLogger(__name__)

_ADJECTIVES = [
    "whimsical",
    "melodic",
    "lyrical",
    "dreamy",
    "wistful",
    "golden",
    "silver",
    "dancing",
    "wandering",
    "twilight",
    "radiant",
    "serene",
    "wild",
    "gentle",
    "fierce",
    "velvet",
    "crimson",
    "azure",
    "emerald",
    "starlit",
]

_POETS = [
    "whitman",
    "dickinson",
    "frost",
    "plath",
    "neruda",
    "rumi",
    "keats",
    "shelley",
    "byron",
    "blake",
    "yeats",
    "auden",
    "bishop",
    "hughes",
    "cummings",
    "rilke",
    "basho",
    "sappho",
    "hafiz",
    "tagore",
]


# Poet-themed name generator (like GitHub's funny release names)
def generate_run_name() -> str:
    """Generate a random poet-themed run name like 'whimsical-whitman-0042'."""
    adj = random.choice(_ADJECTIVES)
    poet = random.choice(_POETS)
    num = random.randint(0, 9999)
    return f"{adj}-{poet}-{num:04d}"


async def save_checkpoint_async(
    training_client: TrainingClient,
    run_name: str,
    log_path: str,
    step: int,
    config: Config,
    final: bool = False,
) -> None:
    """Save model state + optimizer state + loop metadata.

    Args:
        training_client: The training client to save state from.
        run_name: Name of this training run.
        log_path: Local directory for metadata files.
        step: Current global step number.
        config: Config object for reproducibility.
        final: Whether this is the final checkpoint.
    """
    # Simple name for tinker storage (no slashes - only alphanumeric, hyphens, underscores, dots)
    ckpt_suffix = "final" if final else f"step-{step:06d}"
    tinker_name = f"{run_name}.{ckpt_suffix}"

    # Save training weights (for resuming training)
    save_future = training_client.save_state(tinker_name)
    await save_future

    # Save sampler weights (for inference/sampling - faster, less storage)
    sampler_future = training_client.save_weights_for_sampler(tinker_name)
    sampler_result = await sampler_future
    sampler_path = sampler_result.path

    # Save metadata locally for resume
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "tinker_name": tinker_name,
        "sampler_path": sampler_path,
        "step": step,
        "final": final,
        "config": asdict(config),
    }

    metadata_path = log_dir / f"{ckpt_suffix}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved checkpoint to tinker: {tinker_name} (training), "
        f"{sampler_path} (sampling), metadata: {metadata_path}"
    )
