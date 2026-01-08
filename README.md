# Iterative Poetry

A [Tinker](https://tinker-docs.thinkingmachines.ai/) workspace for experimenting with iterative poetry.

## Status

🚧 **Early development**

Done:
- Tinker healthcheck
- REPL to access Qwen3 base models and checkpoints

Next:
- Training and publishing checkpoints

## Data

### `example_data/poems.jsonl`

An example public domain dataset of poems in JSONL format. Each line contains a JSON object with:
- `title`: The poem's title
- `content`: The full text of the poem

This is an example dataset for iterative poetry models.

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
pdm run repl
```

Supports Qwen3-8B and Qwen3-32B with thinking mode. Commands: `/clear`, `/debug`, `/exit`.

### Other Commands

```bash
pdm run format     # Format code with ruff
pdm run lint       # Lint and fix with ruff
pdm run typecheck  # Type check with mypy
```

## Docs

The `docs/` directory contains design notes and training writeups.

- `docs/training.md`: The current architecture for *self-improving poetry generation*. This is where we track the end-to-end training loop (candidate description generation, scoring/selection, data construction, and how we train the final `(title?, description) -> (title, poem)` behavior).

## License

MIT
