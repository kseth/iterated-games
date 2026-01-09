"""Self-improving poetry training loop.

Implements the 4-phase training loop:
1. Generate N candidate descriptions per poem
2. Score candidates and select top-K winners
3. Build K * 2 training examples (Echo + Creative variants)
4. Train with LoRA using masked loss

Usage:
    pdm run train --dataset_path example_data/poems.jsonl --model_name Qwen/Qwen3-8B
"""

from __future__ import annotations

import asyncio
import logging
import random
from itertools import batched
from pathlib import Path
from statistics import mean

import chz
from tinker import AdamParams, Datum, SamplingParams, ServiceClient
from tqdm import tqdm

from checkpoints import generate_run_name, save_checkpoint
from train_config import Config
from data import Poem, filter_poems_by_length, load_poems
from prompts import build_description_request, build_poem_request, build_training_datum
from qwen3_utils import Qwen3Renderer, get_qwen3_tokenizer
from scoring import build_idf_table, compute_mean_nll, compute_overlaps

logger = logging.getLogger(__name__)


def setup_logging(log_path: str) -> None:
    """Configure file-only logging. Console output handled by tqdm."""
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def log_and_print(msg: str, level: str = "info") -> None:
    """Log to file and print to console (via tqdm.write for clean output)."""
    getattr(logger, level)(msg)
    tqdm.write(msg)


# ============================================================================
# Fixed Parameters
# ============================================================================

LORA_RANK = 32  # Works well for both 8B and 32B models
START_LR = 1e-4  # Initial learning rate
END_LR = 5e-5  # Final learning rate (linear decay)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

MAX_OUTPUT_TOKENS = 8192  # Max tokens for training sequences
MAX_POEM_TOKENS = 6144  # Filter poems to ~75% of max_output_tokens

# Generation sampling parameters (higher diversity for candidate generation)
GENERATION_TEMPERATURE = 0.9  # Higher than chat default (0.6)
GENERATION_TOP_P = 0.95
GENERATION_TOP_K = 40  # Higher than chat default (20)
GENERATION_MAX_TOKENS = 4096  # Allow space for <think> reasoning


async def run_training(config: Config) -> None:
    """Main training loop for self-improving poetry generation."""
    run_name = config.run_name or generate_run_name()
    run_path = f"{config.log_path}/{run_name}"

    setup_logging(run_path)
    log_and_print(f"Run: {run_name}")
    log_and_print(f"Starting training with config: {config}")

    service_client = ServiceClient()
    tokenizer = get_qwen3_tokenizer(config.model_name)
    renderer = Qwen3Renderer(tokenizer)

    # Load and filter poems by length
    all_poems = load_poems(config.dataset_path)
    poems = filter_poems_by_length(all_poems, tokenizer, MAX_POEM_TOKENS)
    log_and_print(
        f"Filtered to {len(poems)}/{len(all_poems)} poems under {MAX_POEM_TOKENS} tokens"
    )

    if not poems:
        log_and_print(
            "No poems found after filtering. Check dataset and token limit.", "error"
        )
        return

    idf_table = build_idf_table([p.content for p in poems])

    # Calculate total training batches for LR schedule
    winners_per_poem = min(config.top_k, config.num_candidates)
    examples_per_iteration = len(poems) * winners_per_poem * 2
    batches_per_iteration = (
        examples_per_iteration + config.train_batch_size - 1
    ) // config.train_batch_size
    total_batches = config.max_iterations * batches_per_iteration
    log_and_print(
        f"LR schedule: {START_LR} -> {END_LR} over {total_batches} batches "
        f"({batches_per_iteration}/iteration)"
    )

    # Compute save/eval intervals from per-iteration config
    # saves_per_iteration=2 -> save every batches_per_iteration/2 batches
    # saves_per_iteration=0.5 -> save every batches_per_iteration*2 batches
    save_every = (
        max(1, round(batches_per_iteration / config.saves_per_iteration))
        if config.saves_per_iteration > 0
        else 0
    )
    # resample_every refreshes the sampling_client, so subsequent description
    # generation within the iteration will use the updated model weights.
    # When we resample, we also generate an eval sample with the fresh weights.
    resample_every = (
        max(1, round(batches_per_iteration / config.evals_per_iteration))
        if config.evals_per_iteration > 0
        else 0
    )

    if config.resume_from:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                config.resume_from
            )
        )
        log_and_print(f"Loaded weights from: {config.resume_from}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=LORA_RANK
        )
        log_and_print(f"Starting fresh training with LoRA rank {LORA_RANK}")

    sampling_client = training_client.save_weights_and_get_sampling_client()
    gen_params = SamplingParams(
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        top_k=GENERATION_TOP_K,
        max_tokens=GENERATION_MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
    )

    global_batch = 0
    for iteration in range(config.max_iterations):
        log_and_print(f"=== Iteration {iteration + 1}/{config.max_iterations} ===")
        training_examples: list[Datum] = []
        sample_example: tuple[str, str, str] | None = None  # (title, description, poem)

        # Shuffle poems each iteration for better training dynamics
        shuffled_poems = list(poems)
        random.shuffle(shuffled_poems)

        # Single progress bar for entire iteration:
        # - Generate/score/select: 1 step per poem processed
        # - Train: 1 step per training batch
        num_train_batches = batches_per_iteration
        total_steps = len(poems) + num_train_batches

        pbar = tqdm(
            total=total_steps, desc=f"Iter {iteration + 1} [generate]", leave=False
        )

        # Phases 1-3: Generate, score, select
        for poem_batch_tuple in batched(shuffled_poems, config.score_batch_size):
            poem_batch = list(poem_batch_tuple)

            # Phase 1: Generate N candidates per poem (parallel async)
            prompts = [
                build_description_request(renderer, p.title, p.content)
                for p in poem_batch
            ]
            sample_tasks = [
                sampling_client.sample_async(prompt, config.num_candidates, gen_params)
                for prompt in prompts
            ]
            all_responses = await asyncio.gather(*sample_tasks)

            # Extract description text (message.content only, no reasoning)
            poem_candidates: list[tuple[Poem, list[str]]] = []
            for poem, resp in zip(poem_batch, all_responses, strict=True):
                descriptions = []
                for seq in resp.sequences:
                    message, _success = renderer.parse_response(seq.tokens)
                    descriptions.append(message.content)
                poem_candidates.append((poem, descriptions))

            # Phase 2: Score and select
            for poem, descriptions in poem_candidates:
                overlaps = compute_overlaps(descriptions, poem.content, idf_table)

                # Use same datum structure as training (include_title=True for scoring)
                scoring_datums = [
                    build_training_datum(
                        renderer, poem.title, desc, poem.content, include_title=True
                    )
                    for desc in descriptions
                ]

                fwd_future = await training_client.forward_async(
                    scoring_datums, "cross_entropy"
                )
                result = await fwd_future.result_async()
                losses = [
                    compute_mean_nll(out, datum)
                    for out, datum in zip(
                        result.loss_fn_outputs, scoring_datums, strict=True
                    )
                ]

                # Combine scores (lower is better): loss + overlap_weight * overlap
                scores = [
                    loss + config.overlap_weight * overlap
                    for loss, overlap in zip(losses, overlaps, strict=True)
                ]

                winner_indices = sorted(range(len(scores)), key=lambda i: scores[i])[
                    : config.top_k
                ]

                # Phase 3: Build K * 2 training examples (Echo + Creative variants)
                for idx in winner_indices:
                    desc = descriptions[idx]
                    if sample_example is None:
                        sample_example = (poem.title, desc, poem.content)
                    training_examples.append(
                        build_training_datum(
                            renderer, poem.title, desc, poem.content, include_title=True
                        )
                    )
                    training_examples.append(
                        build_training_datum(
                            renderer,
                            poem.title,
                            desc,
                            poem.content,
                            include_title=False,
                        )
                    )

                pbar.update(1)

        # Phase 4: Train on collected examples
        random.shuffle(training_examples)
        pbar.set_description(f"Iter {iteration + 1} [train]")

        iteration_losses: list[float] = []

        for batch_tuple in batched(training_examples, config.train_batch_size):
            batch = list(batch_tuple)

            # Linear LR decay from START_LR to END_LR
            progress = global_batch / total_batches if total_batches > 0 else 0.0
            current_lr = START_LR + (END_LR - START_LR) * progress
            adam_params = AdamParams(
                learning_rate=current_lr,
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                eps=ADAM_EPS,
            )

            fwd_bwd_future = await training_client.forward_backward_async(
                batch, "cross_entropy"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()

            batch_losses = [
                compute_mean_nll(out, datum)
                for out, datum in zip(
                    fwd_bwd_result.loss_fn_outputs, batch, strict=True
                )
            ]
            batch_loss = mean(batch_losses)
            iteration_losses.append(batch_loss)

            pbar.set_postfix(loss=f"{batch_loss:.3f}", lr=f"{current_lr:.1e}")
            pbar.update(1)
            global_batch += 1

            if save_every > 0 and global_batch % save_every == 0:
                save_checkpoint(
                    training_client=training_client,
                    run_name=run_name,
                    log_path=run_path,
                    batch=global_batch,
                    config=config,
                )

            # Resample: refresh weights mid-iteration, generate eval sample
            if (
                resample_every > 0
                and global_batch % resample_every == 0
                and sample_example
            ):
                sampling_client = training_client.save_weights_and_get_sampling_client()
                title, desc, ground_truth_poem = sample_example
                log_and_print(f"[Resample @ batch {global_batch}] Title: {title}")
                log_and_print(f"[Resample] Description: {desc[:150]}...")

                eval_prompt = build_poem_request(renderer, title, desc)
                eval_response = await sampling_client.sample_async(
                    eval_prompt, 1, gen_params
                )
                generated_msg, _ = renderer.parse_response(
                    eval_response.sequences[0].tokens
                )

                log_and_print(f"[Resample] Generated: {generated_msg.content[:200]}...")
                log_and_print(f"[Resample] Ground truth: {ground_truth_poem[:200]}...")

        pbar.close()

        iter_mean_loss = mean(iteration_losses) if iteration_losses else 0.0
        log_and_print(
            f"Iteration {iteration + 1} complete: "
            f"loss={iter_mean_loss:.4f}, examples={len(training_examples)}"
        )

        # Refresh weights for next iteration
        sampling_client = training_client.save_weights_and_get_sampling_client()

    save_checkpoint(
        training_client=training_client,
        run_name=run_name,
        log_path=run_path,
        batch=global_batch,
        config=config,
        final=True,
    )
    log_and_print("Training complete!")


def main() -> None:
    """CLI entrypoint."""
    config = chz.entrypoint(Config)
    asyncio.run(run_training(config))


if __name__ == "__main__":
    main()
