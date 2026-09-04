# Contributing

Thanks for your interest in contributing to pyseekdb. This file covers development and testing
instructions.

## Development

- Use Python 3.11+.
- This repo uses `uv`. You can override the binary via `UV=...`.

This project uses [uv](https://docs.astral.sh/uv/) as the package manager with
[hatchling](https://hatch.pypa.io/) as the build backend. All common development
tasks are unified through the `Makefile`.

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

# Install dependencies (all groups)
make install
```

## Make Targets

```bash
make help                      # Show all available targets
make install                   # Install dependencies
make test                      # Run unit tests
make docs                      # Build documentation
make build                     # Build the package
make clean                     # Clean build artifacts
```

## Build Artifacts

After running `make build`, the distribution files will be in the `dist/` directory:
- `pyseekdb-<version>.tar.gz` - Source distribution
- `pyseekdb-<version>-py3-none-any.whl` - Wheel distribution

## Testing

```bash
# Run unit tests
make test

# Run specific tests with uv run
uv run pytest tests/integration_tests/ -v -k "server"     # server mode (requires seekdb server)
uv run pytest tests/integration_tests/ -v -k "oceanbase"  # oceanbase mode (requires OceanBase)

# Run specific test file
uv run pytest tests/integration_tests/test_collection_query.py -v

# Run specific test function
uv run pytest tests/integration_tests/test_collection_query.py::TestCollectionQuery::test_collection_query -v
```
