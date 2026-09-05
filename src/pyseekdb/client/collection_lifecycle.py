"""Shared lifecycle state helpers for SDK collection catalog rows."""

from __future__ import annotations

import json
from typing import Any

COLLECTION_STATE_CREATING = "creating"
COLLECTION_STATE_READY = "ready"
COLLECTION_STATE_FAILED = "failed"
COLLECTION_CREATION_TOKEN_KEY = "creation_token"  # noqa: S105


def parse_collection_settings(value: Any) -> dict[str, Any]:
    """Decode a catalog ``settings`` value into a mapping.

    Older catalog rows do not contain a lifecycle state and are intentionally
    treated as ready by :func:`collection_state` for backwards compatibility.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Collection settings must be a JSON object, got {type(value).__name__}")
    return value


def collection_state(settings: Any) -> str:
    """Return a collection lifecycle state, defaulting legacy rows to ready."""
    return parse_collection_settings(settings).get("state", COLLECTION_STATE_READY)
