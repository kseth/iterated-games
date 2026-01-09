# Iterative Poetry

A [Tinker](https://tinker-docs.thinkingmachines.ai/) workspace for experimenting with iterative writing.

## Status

🚧 **Early development**

Done:
- Tinker healthcheck
- REPL to access Qwen3 base models and checkpoints
- Training loop with checkpoint save/resume

## Data

Example datasets in `example_data/` (JSONL format, each line has `title` and `content`):
- `tiny_poems.jsonl` — small dataset for prototyping
- `more_poems.jsonl` — larger public domain collection

## Setup

### Requirements

- Python 3.13
- [PDM](https://pdm-project.org/)

### Installation

```bash
pdm install
```

### Configuration

Create a `.env` file at the project root with your Tinker API key:

```
TINKER_API_KEY=your_api_key_here
```

## Usage

Verify your Tinker connection:

```bash
pdm run tinkercheck
```

Start the interactive REPL:

```bash
pdm run repl                            # interactive model selection
pdm run repl checkpoint=tinker://...    # load from checkpoint
```

Supports Qwen3-8B and Qwen3-32B with thinking mode. Commands: `/clear`, `/debug`, `/exit`.

Run the training loop:

```bash
pdm run train                           # uses defaults from train_config.py
pdm run train --help                    # see all options
```

### Other Commands

```bash
pdm run format     # Format code with ruff
pdm run lint       # Lint and fix with ruff
pdm run typecheck  # Type check with mypy
```

## Generation

Use the REPL (`pdm run repl`) to chat with base models or fine-tuned checkpoints. The model uses Qwen3's native thinking mode—you'll see `<think>` reasoning blocks before responses.

Key flags:
- `checkpoint=tinker://...` — load a trained checkpoint (base model is inferred automatically)
- `model_name=Qwen/Qwen3-8B` — override model selection

The REPL is useful for spot-checking checkpoint quality, testing prompts, and interactive experimentation.

## Training

Use `pdm run train` to run the self-improving training loop. Each iteration generates candidate descriptions for poems, scores them by predictive utility, and trains on the winners.

Key flags:
- `dataset_path=...` — path to poems JSONL
- `resume_from=tinker://...` — resume from a checkpoint (includes optimizer state)
- `max_iterations=N` — number of passes through the dataset

Checkpoints are saved to `logs/poetry-train/<run-name>/` with auto-generated run names. See `docs/training.md` for the full algorithm description.

## Docs

The `docs/` directory contains design notes and training writeups.

- `docs/training.md`: The current architecture for *self-improving poetry generation*. This is where we track the end-to-end training loop (candidate description generation, scoring/selection, data construction, and how we train the final `(title?, description) -> (title, poem)` behavior).

## License

MIT
