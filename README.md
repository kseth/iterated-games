# Iterated Games

A [Tinker](https://tinker-docs.thinkingmachines.ai/) workspace for experimenting with reinforcement learning over iterated games and transfer learning.

## Status

🚧 **Early development** — only the scratchpad (`tinker_check.py`) has been implemented.

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

### Other Commands

```bash
pdm run format     # Format code with ruff
pdm run lint       # Lint and fix with ruff
pdm run typecheck  # Type check with mypy
```

## License

MIT

