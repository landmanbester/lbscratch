# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

lbscratch is a collection of utilities for radio astronomy data processing and interferometry analysis. It's built as a Python package with CLI tools that process measurement sets, perform calibration, and apply various imaging/flagging algorithms.

## Installation and Setup

Install in editable mode:
```bash
pip install -e .
```

Requires Python 3.10+ and dependencies include:
- finufft (for non-uniform FFTs)
- pfb-imaging (from git+https://github.com/ratt-ru/pfb-imaging.git@main)
- smoove (from git+https://github.com/landmanbester/smoove.git@test_ci)

## CLI Usage

The main CLI entry point is `lbs` which provides various worker commands:

```bash
lbs fledges --help    # Frequency flagging
lbs bsmooth --help    # Baseline smoothing
lbs restimator --help # Residual estimation
lbs gsmooth --help    # Gain smoothing
lbs hess_psf --help   # Hessian PSF operations
```

## Development Commands

### Testing
```bash
uv run pytest tests/
```

### Code Quality
```bash
# Format code with ruff
uv run ruff format lbscratch tests

# Lint and fix issues
uv run ruff check lbscratch tests --fix

# Type checking
uv run mypy lbscratch
```

### Test Files
- `test_delay_init.py` - delay calibration tests (currently commented out)
- `test_1D_claude.py` - 1D processing tests
- `test_grid1D.py` - 1D gridding tests

## Architecture

### Core Components

- **lbscratch/workers/**: CLI command implementations for data processing pipelines
- **lbscratch/parser/**: YAML schema definitions and parameter validation using scabha
- **lbscratch/utils.py**: Mathematical utilities (smoothing, thresholding, numba-accelerated functions)
- **lbscratch/pygridder.py**: Gridding and degridding operations using ducc0.wgridder

### Configuration System

The project uses a schema-driven configuration system:
- YAML files in `lbscratch/parser/` define input/output parameters for each worker
- `schemas.py` loads all YAML configurations using scabha/OmegaConf
- Parameters are automatically converted to Click CLI options
- Stimela integration via `stimela_cabs.yaml`

### Worker Pattern

Each worker follows this pattern:
1. Schema-defined parameters loaded from YAML
2. Click CLI interface auto-generated from schema
3. Logging setup with pyscilog
4. OmegaConf configuration management

Example workers:
- `fledges`: Frequency domain flagging/masking
- `bsmooth`: Baseline-dependent smoothing
- `restimator`: Residual image estimation
- `gsmooth`: Gain solution smoothing
- `hess_psf`: Hessian-based PSF operations

### Key Dependencies

- **ducc0**: Fast FFTs and gridding operations
- **scabha**: Configuration and schema validation
- **numba**: JIT compilation for performance-critical code
- **smoove**: Smoothing algorithms (kanterp interpolation)
- **pfb**: Imaging utilities and measurement set operations

## Development Notes

- Use existing mathematical utilities in `utils.py` before implementing new ones
- Follow the worker pattern when adding new CLI commands
- YAML schemas must be updated when adding new parameters
- The codebase focuses on radio astronomy data processing, particularly interferometry
- Most algorithms are optimized for large-scale data processing with dask/numba
