"""Self-improving poetry training loop.

Implements an interleaved training loop where generation and training happen
in mini-batches rather than all-generation-then-all-training:

For each mini-batch of poems:
1. Generate N candidate descriptions per poem
2. Score candidates and select top-K winners
3. Build K * 2 training examples (Echo + Creative variants)
4. Train immediately on these examples

Usage:
    pdm run train dataset_path=example_data/poems.jsonl model_name=Qwen/Qwen3-8B
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from itertools import batched
from pathlib import Path
from statistics import mean

import chz
from tinker import AdamParams, Datum, SamplingParams, ServiceClient
from tqdm import tqdm

from checkpoints import generate_run_name, save_checkpoint_async
from data import Poem, filter_poems_by_length, load_poems
from prompts import build_description_request, build_poem_request, build_training_datum
from qwen3_utils import Qwen3Renderer, get_qwen3_tokenizer
from scoring import build_idf_table, compute_mean_nll, compute_overlaps
from train_config import Config

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

MAX_POEM_TOKENS = 6144  # ~75% of max context to leave room for description

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
    log_and_print(f"Config: {config}")

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
        log_and_print("No poems found after filtering.", "error")
        return

    idf_table = build_idf_table([p.content for p in poems])

    # Calculate batch sizes
    # Each poem produces top_k * 2 training examples (Echo + Creative)
    winners_per_poem = min(config.top_k, config.num_candidates)
    examples_per_poem = winners_per_poem * 2
    poems_per_train_batch = max(1, config.train_batch_size // examples_per_poem)

    # Total steps for LR schedule (one step = one mini-batch of poems trained)
    steps_per_iteration = math.ceil(len(poems) / poems_per_train_batch)
    total_steps = config.max_iterations * steps_per_iteration
    log_and_print(
        f"Interleaved training: {poems_per_train_batch} poems -> ~{poems_per_train_batch * examples_per_poem} examples per train step"
    )
    log_and_print(
        f"LR schedule: {START_LR} -> {END_LR} over {total_steps} steps "
        f"({steps_per_iteration}/iteration)"
    )

    # Compute checkpoint/resample intervals from per-iteration config
    # saves_per_iteration=2 -> save every steps_per_iteration/2 steps
    # saves_per_iteration=0.5 -> save every steps_per_iteration*2 steps
    save_every = (
        max(1, int(math.floor(steps_per_iteration / config.saves_per_iteration)))
        if config.saves_per_iteration > 0
        else 0
    )
    # resample_every refreshes the sampling_client, so subsequent description
    # generation within the iteration will use the updated model weights.
    # When we resample, we also generate an eval sample with the fresh weights.
    resample_every = (
        max(1, int(math.floor(steps_per_iteration / config.evals_per_iteration)))
        if config.evals_per_iteration > 0
        else 0
    )

    # Initialize training client
    if config.resume_from:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                config.resume_from
            )
        )
        log_and_print(f"Resumed from: {config.resume_from}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=LORA_RANK
        )
        log_and_print(f"Fresh training with LoRA rank {LORA_RANK}")

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    gen_params = SamplingParams(
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        top_k=GENERATION_TOP_K,
        max_tokens=GENERATION_MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
    )

    global_step = 0
    for iteration in range(config.max_iterations):
        log_and_print(f"=== Iteration {iteration + 1}/{config.max_iterations} ===")

        # Shuffle poems each iteration for better training dynamics
        shuffled_poems = list(poems)
        random.shuffle(shuffled_poems)

        iteration_losses: list[float] = []
        total_examples_this_iter = 0
        sample_example: tuple[str, str, str] | None = None  # (title, desc, poem)
        current_lr = START_LR  # Will be updated in training loop
        last_resample_step = 0  # Track to avoid double-resample at iteration end

        # Progress bar for the iteration
        pbar = tqdm(
            total=steps_per_iteration,
            desc=f"Iter {iteration + 1}",
            leave=False,
        )

        # Process poems in training-batch-sized chunks
        for train_batch_poems_tuple in batched(shuffled_poems, poems_per_train_batch):
            train_batch_poems = list(train_batch_poems_tuple)
            training_examples: list[Datum] = []
            first_poem_sampled = False  # Track first poem in this mini-batch

            # Phase 1: Generate N candidates per poem (in score_batch_size chunks)
            pbar.set_description(f"Iter {iteration + 1} [generate]")
            pbar.refresh()
            poem_candidates: list[tuple[Poem, list[str]]] = []
            for score_batch_tuple in batched(
                train_batch_poems, config.score_batch_size
            ):
                score_batch = list(score_batch_tuple)
                prompts = [
                    build_description_request(renderer, p.title, p.content)
                    for p in score_batch
                ]
                sample_tasks = [
                    sampling_client.sample_async(
                        prompt, config.num_candidates, gen_params
                    )
                    for prompt in prompts
                ]
                all_responses = await asyncio.gather(*sample_tasks)

                # Extract descriptions (strip <think> reasoning)
                for poem, resp in zip(score_batch, all_responses, strict=True):
                    descriptions = []
                    for seq in resp.sequences:
                        message, _ = renderer.parse_response(seq.tokens)
                        descriptions.append(message.content)
                    poem_candidates.append((poem, descriptions))

            # Update progress after generation phase (generation takes the most time)
            pbar.update(0.5)
            log_and_print(
                f"Generated {len(poem_candidates)} poems * {config.num_candidates} candidates"
            )

            # Phase 2: Score all candidates and select winners
            pbar.set_description(f"Iter {iteration + 1} [score]")
            pbar.refresh()

            # Build all scoring datums
            scoring_datums: list[Datum] = []
            overlaps: list[float] = []

            for poem, descriptions in poem_candidates:
                # Compute overlap penalty (penalizes descriptions that copy the poem)
                poem_overlaps = compute_overlaps(descriptions, poem.content, idf_table)

                # Build datums for all candidates of this poem
                # Use include_title=True (Echo format) for scoring to match training
                for desc in descriptions:
                    scoring_datums.append(
                        build_training_datum(
                            renderer, poem.title, desc, poem.content, include_title=True
                        )
                    )
                    overlaps.append(poem_overlaps[descriptions.index(desc)])

            # Single forward pass for all scoring datums (batched for efficiency)
            if scoring_datums:
                fwd_future = await training_client.forward_async(
                    scoring_datums, "cross_entropy"
                )
                result = await fwd_future
                losses = [
                    compute_mean_nll(loss_fn_output, datum)
                    for loss_fn_output, datum in zip(
                        result.loss_fn_outputs, scoring_datums, strict=True
                    )
                ]

                # Process each poem's candidates and select winners
                datum_idx = 0
                for poem, descriptions in poem_candidates:
                    next_datum_idx = datum_idx + config.num_candidates
                    poem_losses = losses[datum_idx:next_datum_idx]
                    poem_overlaps = overlaps[datum_idx:next_datum_idx]
                    datum_idx = next_datum_idx

                    # Combined score: loss + overlap_weight * overlap (lower is better)
                    scores = [
                        loss + config.overlap_weight * overlap
                        for loss, overlap in zip(
                            poem_losses, poem_overlaps, strict=True
                        )
                    ]

                    # Select top_k winners per poem (lowest combined score)
                    winner_indices = sorted(
                        range(len(scores)), key=lambda i: scores[i]
                    )[: config.top_k]

                    # Phase 3: Build K * 2 training examples (Echo + Creative variants)
                    for idx in winner_indices:
                        desc = descriptions[idx]
                        # Update sample_example to first example from current mini-batch
                        if not first_poem_sampled:
                            sample_example = (poem.title, desc, poem.content)
                            first_poem_sampled = True
                        # Echo variant (title in prompt)
                        training_examples.append(
                            build_training_datum(
                                renderer,
                                poem.title,
                                desc,
                                poem.content,
                                include_title=True,
                            )
                        )
                        # Creative variant (no title in prompt)
                        training_examples.append(
                            build_training_datum(
                                renderer,
                                poem.title,
                                desc,
                                poem.content,
                                include_title=False,
                            )
                        )

            # Update progress after scoring/selection phase
            pbar.update(0.25)
            log_and_print(
                f"Scored and selected top-{config.top_k} winners per poem "
                f"from {config.num_candidates} candidates"
            )

            # Phase 4: Train immediately on this batch's examples
            pbar.set_description(f"Iter {iteration + 1} [train]")
            pbar.refresh()
            total_examples_this_iter += len(training_examples)
            random.shuffle(training_examples)

            for train_batch_tuple in batched(
                training_examples, config.train_batch_size
            ):
                train_batch = list(train_batch_tuple)

                # Linear LR decay from START_LR to END_LR
                progress = global_step / total_steps if total_steps > 0 else 0.0
                current_lr = START_LR + (END_LR - START_LR) * progress
                adam_params = AdamParams(
                    learning_rate=current_lr,
                    beta1=ADAM_BETA1,
                    beta2=ADAM_BETA2,
                    eps=ADAM_EPS,
                )

                # Submit both requests together (they'll run in the same clock cycle)
                # Tinker handles the dependency: optim_step waits for forward_backward internally
                fwd_bwd_future = await training_client.forward_backward_async(
                    train_batch, "cross_entropy"
                )
                optim_future = await training_client.optim_step_async(adam_params)
                # Wait for both results
                fwd_bwd_result = await fwd_bwd_future
                await optim_future

                step_losses = [
                    compute_mean_nll(out, datum)
                    for out, datum in zip(
                        fwd_bwd_result.loss_fn_outputs, train_batch, strict=True
                    )
                ]
                step_loss = mean(step_losses)
                iteration_losses.append(step_loss)

                global_step += 1

            # Update progress after training phase
            pbar.set_postfix(
                loss=f"{iteration_losses[-1]:.3f}" if iteration_losses else "N/A",
                lr=f"{current_lr:.1e}",
                poems=len(train_batch_poems),
            )
            pbar.update(0.25)

            # Log step details to file
            if iteration_losses:
                log_and_print(
                    f"Step {global_step}: loss={iteration_losses[-1]:.4f}, "
                    f"lr={current_lr:.2e}, poems={len(train_batch_poems)}, "
                    f"examples={len(training_examples)}"
                )

            # Checkpoint saving (based on global_step)
            if save_every > 0 and global_step % save_every == 0:
                await save_checkpoint_async(
                    training_client=training_client,
                    run_name=run_name,
                    log_path=run_path,
                    step=global_step,
                    config=config,
                )

            # Resample: refresh sampling client for tighter feedback loop
            if resample_every > 0 and global_step % resample_every == 0:
                sampling_client = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )
                last_resample_step = global_step

                # Generate eval sample if we have one
                if sample_example:
                    title, desc, ground_truth_poem = sample_example
                    log_and_print(f"[Resample @ step {global_step}] Title: {title}")
                    log_and_print(f"[Resample] Description: {desc[:150]}...")

                    eval_prompt = build_poem_request(renderer, title, desc)
                    eval_response = await sampling_client.sample_async(
                        eval_prompt, 1, gen_params
                    )
                    generated_msg, _ = renderer.parse_response(
                        eval_response.sequences[0].tokens
                    )

                    log_and_print(
                        f"[Resample] Generated: {generated_msg.content[:200]}..."
                    )
                    log_and_print(
                        f"[Resample] Ground truth: {ground_truth_poem[:200]}..."
                    )

        pbar.close()

        iter_mean_loss = mean(iteration_losses) if iteration_losses else 0.0
        log_and_print(
            f"Iteration {iteration + 1} complete: "
            f"loss={iter_mean_loss:.4f}, examples={total_examples_this_iter}"
        )

        # Refresh sampling client for next iteration (skip if we just resampled)
        if last_resample_step != global_step:
            sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )

    # Final checkpoint
    await save_checkpoint_async(
        training_client=training_client,
        run_name=run_name,
        log_path=run_path,
        step=global_step,
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
