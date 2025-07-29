# lbscratch

Collection of utilities for radio astronomy data processing and interferometry analysis.
Use at your own peril!

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for modern Python package management.

### Prerequisites

Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Development Installation

Clone the repository and install in development mode:
```bash
git clone https://github.com/landmanbester/lbscratch.git
cd lbscratch
uv sync --dev
```

This will:
- Create a virtual environment with Python 3.10+
- Install the package in editable mode
- Install all development dependencies

### Regular Installation

For users who just want to install the package:
```bash
uv add lbscratch
```

### Optional Dependencies

Install with JAX support for enhanced scientific computing:
```bash
uv sync --extra jax
```

Install with documentation tools:
```bash
uv sync --extra docs
```

Install everything:
```bash
uv sync --all-extras
```

## Development Workflow

### Running Tests
```bash
uv run pytest
```

### Code Formatting and Linting
```bash
# Format code
uv run ruff format lbscratch tests

# Check and fix linting issues
uv run ruff check lbscratch tests --fix

# Check only (no fixes)
uv run ruff check lbscratch tests
```

### Type Checking
```bash
uv run mypy lbscratch
```

### Using the CLI
After installation, the `lbs` command is available:
```bash
uv run lbs --help
uv run lbs fledges --help
```

### Legacy pip Installation
If you prefer pip, you can still install in editable mode:
```bash
pip install -e .
```

Requires Python 3.10+ and a fresh virtual environment.
