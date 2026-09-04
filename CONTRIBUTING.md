# Contributing

Thanks for your interest in contributing to pyseekdb. This file covers development and testing
instructions.

## Development

- Use Python 3.11+.
- Use uv 0.12.9 or newer within the 0.12 release line.

This project uses [uv](https://docs.astral.sh/uv/) as the package manager with
[hatchling](https://hatch.pypa.io/) as the build backend. Common development
commands are run directly through `uv`.

### Prerequisites

Install uv:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/oceanbase/pyseekdb.git
cd pyseekdb

# Install development dependencies
uv sync --group dev
```

## Common Commands

```bash
uv lock --locked               # Verify the lock file is up to date
uv run ruff check .            # Run lint checks
uv run ruff format --check .   # Verify formatting
uv run pytest tests/unit_tests/ -v --log-cli-level=INFO
uv sync --group dev --group docs # Install documentation dependencies too
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv build                       # Build the package
```

The documentation toolchain requires Python 3.12 or newer. The package itself
continues to support Python 3.11.

## Build Artifacts

After running `uv build`, the distribution files will be in the `dist/` directory:
- `pyseekdb-<version>.tar.gz` - Source distribution
- `pyseekdb-<version>-py3-none-any.whl` - Wheel distribution

## Testing

```bash
# Run unit tests
uv run pytest tests/unit_tests/ -v --log-cli-level=INFO

# Run specific tests with uv run
uv run pytest tests/integration_tests/ -v -k "server"     # server mode (requires seekdb server)
uv run pytest tests/integration_tests/ -v -k "oceanbase"  # oceanbase mode (requires OceanBase)

# Run specific test file
uv run pytest tests/integration_tests/test_collection_query.py -v

# Run specific test function
uv run pytest tests/integration_tests/test_collection_query.py::TestCollectionQuery::test_collection_query -v
```
