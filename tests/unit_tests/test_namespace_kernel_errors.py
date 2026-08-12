"""
Unit tests for OceanBase kernel error translation in namespace operations.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.kernel_errors import (  # noqa: E402
    _friendly_kernel_error_message,
    _is_missing_collection_physical_error,
    _is_namespace_dropping_error,
    _is_namespace_missing_kernel_error,
    maybe_reraise_friendly_kernel_error,
    namespace_kernel_error_scope,
)


class IntegrityError(Exception):
    """Stand-in for pymysql IntegrityError in unit tests."""


class TestNamespaceKernelErrorDetection:
    """TestNamespaceKernelErrorDetection class."""

    def test_detects_namespace_dropping_4109(self):
        """Test detects namespace dropping 4109."""
        exc = Exception("DBMS_LOGIC_TABLE.PREWARM failed: errcode=-4109")
        assert _is_namespace_dropping_error(exc) is True

    def test_detects_namespace_missing_4018(self):
        """Test detects namespace missing 4018."""
        exc = Exception("namespace not exist, code=4018")
        assert _is_namespace_missing_kernel_error(exc) is True

    def test_detects_missing_logic_data_table(self):
        """Test detects missing logic data table."""
        exc = Exception("Table 'abc123_logic_data_table' doesn't exist")
        assert _is_missing_collection_physical_error(exc, "abc123") is True

    def test_maps_duplicate_namespace_to_friendly_message(self):
        """Duplicate sdk_namespaces insert is handled in create path, not here."""
        exc = IntegrityError("(1062, \"Duplicate entry 'cid-demo_ns' for key 'uk_sdk_ns_coll_name'\")")
        assert (
            _friendly_kernel_error_message(
                exc,
                namespace_name="demo_ns",
                collection_name="coll",
                collection_id="cid",
            )
            is None
        )


class TestNamespaceKernelErrorTranslation:
    """TestNamespaceKernelErrorTranslation class."""

    def test_wraps_namespace_dropping_error(self):
        """Test wraps namespace dropping error."""
        exc = Exception("prewarm failed, errcode=-4109")
        with (
            namespace_kernel_error_scope(
                namespace_name="demo_ns",
                collection_name="coll",
                collection_id="cid",
            ),
            pytest.raises(ValueError, match="being dropped"),
        ):
            maybe_reraise_friendly_kernel_error(exc)

    def test_wraps_namespace_missing_error(self):
        """Test wraps namespace missing error."""
        exc = Exception("namespace does not exist, code=4018")
        with (
            namespace_kernel_error_scope(
                namespace_name="demo_ns",
                collection_name="coll",
                collection_id="cid",
            ),
            pytest.raises(ValueError, match="no longer exists"),
        ):
            maybe_reraise_friendly_kernel_error(exc)

    def test_wraps_missing_collection_physical_error(self):
        """Test wraps missing collection physical error."""
        exc = Exception("Table 'abc123_logic_data_table' doesn't exist")
        with (
            namespace_kernel_error_scope(
                namespace_name="demo_ns",
                collection_name="coll",
                collection_id="abc123",
            ),
            pytest.raises(ValueError, match="Collection 'coll' does not exist"),
        ):
            maybe_reraise_friendly_kernel_error(exc)

    def test_leaves_unrelated_errors_untouched(self):
        """Test leaves unrelated errors untouched."""
        exc = ValueError("syntax error")
        with namespace_kernel_error_scope(
            namespace_name="demo_ns",
            collection_name="coll",
            collection_id="cid",
        ):
            maybe_reraise_friendly_kernel_error(exc)
