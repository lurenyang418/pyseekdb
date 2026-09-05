"""
Pytest configuration and shared fixtures for pyseekdb tests.
Provides parameterized client fixtures for testing across server and oceanbase modes.
"""

import contextlib
import os
import sys
from pathlib import Path

import pytest

# Add project path (repo root + src + this directory for local test helpers)
integration_tests_root = Path(__file__).resolve().parent
repo_root = integration_tests_root.parents[1]
src_root = repo_root / "src"
sys.path.insert(0, str(integration_tests_root))
sys.path.insert(0, str(src_root))

from namespace_test_support import maybe_skip_namespace_integration_test  # noqa: E402

import pyseekdb  # noqa: E402
from pyseekdb.client.embedding_function import EmbeddingFunctionRegistry  # noqa: E402
from tests.stubs import StubEmbeddingFunction  # noqa: E402

EmbeddingFunctionRegistry.register(StubEmbeddingFunction)

# ==================== Environment Variable Configuration ====================
# Server mode (seekdb Server)
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "2881"))
SERVER_DATABASE = os.environ.get("SERVER_DATABASE", "test")
SERVER_USER = os.environ.get("SERVER_USER", "root")
SERVER_PASSWORD = os.environ.get("SERVER_PASSWORD", "")

# OceanBase mode
OB_HOST = os.environ.get("OB_HOST", "localhost")
OB_PORT = int(os.environ.get("OB_PORT", "11202"))
OB_TENANT = os.environ.get("OB_TENANT", "mysql")
OB_DATABASE = os.environ.get("OB_DATABASE", "test")
OB_USER = os.environ.get("OB_USER", "root")
OB_PASSWORD = os.environ.get("OB_PASSWORD", "")


# ==================== Client Factory Functions ====================
def create_server_client():
    """Create a server client instance."""
    client = pyseekdb.Client(
        host=SERVER_HOST,
        port=SERVER_PORT,
        tenant="sys",
        database=SERVER_DATABASE,
        user=SERVER_USER,
        password=SERVER_PASSWORD,
    )

    # Test connection
    try:
        assert client.ping()
    except Exception as exc:
        pytest.fail(f"seekdb server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")

    return client


def create_oceanbase_client():
    """Create an OceanBase client instance."""
    client = pyseekdb.Client(
        host=OB_HOST,
        port=OB_PORT,
        tenant=OB_TENANT,
        database=OB_DATABASE,
        user=OB_USER,
        password=OB_PASSWORD,
    )

    # Test connection
    try:
        assert client.ping()
    except Exception as exc:
        pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")

    return client


# ==================== Parameterized Client Fixtures ====================
@pytest.fixture(params=["server", "oceanbase"])
def db_client(request):
    """
    Parameterized fixture that provides clients for server and oceanbase modes.

    Usage:
        def test_my_feature(db_client):
            collection = db_client.get_or_create_collection(...)
            # test logic here

    This will automatically run 2 times: once for each client mode.
    Generated test names will be:
        - test_my_feature[server]
        - test_my_feature[oceanbase]
    """
    mode = request.param

    if mode == "server":
        client = create_server_client()
    elif mode == "oceanbase":
        client = create_oceanbase_client()
    else:
        raise ValueError(f"Unknown client mode: {mode}")

    yield client

    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def server_client():
    """Fixture for server client only."""
    client = create_server_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def oceanbase_client():
    """Fixture for OceanBase client only."""
    client = create_oceanbase_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


def pytest_runtest_setup(item):
    """Skip namespace integration tests on unsupported backends or OB versions."""
    maybe_skip_namespace_integration_test(item)
