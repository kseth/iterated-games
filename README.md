# Iterated Games

A [Tinker](https://tinker-docs.thinkingmachines.ai/) workspace for experimenting with reinforcement learning over iterated games and transfer learning.

## Status

🚧 **Early development**

Done:
- Tinker healthcheck
- REPL to access Qwen3 base models and checkpoints

Next:
- Training and publishing checkpoints

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

## License

MIT

