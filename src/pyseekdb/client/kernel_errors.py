"""Translate OceanBase kernel errors into SDK-friendly ValueError messages."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import re
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

_NS_KERNEL_ERROR_CTX: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "ns_kernel_error_ctx", default=None
)

_NS_PHYSICAL_TABLE_SUFFIXES = (
    "_logic_data_table",
    "_kv_data_table",
    "_logic_schema_table",
    "_hot_table",
)


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield the exception and its cause/context chain."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        cause = current.__cause__
        context = current.__context__
        current = cause if cause is not None else context


def _exception_text(exc: BaseException) -> str:
    """Flatten the exception chain into one searchable string."""
    return " ".join(str(item) for item in _iter_exception_chain(exc))


def _has_ob_error_code(exc: BaseException, *codes: int) -> bool:
    """Return whether any chained exception carries one of the OceanBase error codes."""
    text = _exception_text(exc)
    lowered = text.lower()
    for code in codes:
        code_str = str(code)
        if (
            f"code={code_str}" in lowered
            or f"code=-{code_str}" in lowered
            or f"errcode=-{code_str}" in lowered
            or re.search(rf"\({code_str},", text) is not None
            or re.search(rf"(?:^|[\s,])(?:-{code_str}|{code_str})(?:[\s,)]|$)", text) is not None
        ):
            return True
    return False


def _is_namespace_dropping_error(exc: BaseException) -> bool:
    """Whether the kernel reports a namespace is still being dropped."""
    text = _exception_text(exc).lower()
    return (
        _has_ob_error_code(exc, 4109)
        or "namespace is dropping" in text
        or "namespace is in drop" in text
        or ("being dropped" in text and "namespace" in text)
    )


def _is_namespace_missing_kernel_error(exc: BaseException) -> bool:
    """Whether the kernel reports the namespace no longer exists."""
    text = _exception_text(exc).lower()
    return (
        _has_ob_error_code(exc, 4018)
        or "namespace not exist" in text
        or "namespace does not exist" in text
        or "namespace doesn't exist" in text
        or "no such namespace" in text
    )


def _is_missing_collection_physical_error(exc: BaseException, collection_id: str | None = None) -> bool:
    """Whether the error indicates namespace collection physical tables are gone."""
    text = _exception_text(exc).lower()
    if collection_id:
        cid = collection_id.lower()
        for suffix in _NS_PHYSICAL_TABLE_SUFFIXES:
            if f"{cid}{suffix}" in text:
                return True
    if any(suffix in text for suffix in _NS_PHYSICAL_TABLE_SUFFIXES):
        return True
    if "table doesn't exist" in text or "table does not exist" in text or "table not exist" in text:
        return True
    return _has_ob_error_code(exc, 1146, 5019, 942)


def _friendly_kernel_error_message(
    exc: BaseException,
    *,
    namespace_name: str | None,
    collection_name: str | None,
    collection_id: str | None,
) -> str | None:
    """Map a kernel exception to a user-facing message, or None to keep the original."""
    if _is_namespace_dropping_error(exc):
        ns_label = f"'{namespace_name}'" if namespace_name else "The namespace"
        return (
            f"Namespace {ns_label} is being dropped and is not available. "
            "Wait for the drop to finish or use a different namespace."
        )
    if _is_namespace_missing_kernel_error(exc):
        ns_label = f"'{namespace_name}'" if namespace_name else "The namespace"
        return (
            f"Namespace {ns_label} no longer exists (it may have been deleted). "
            "Operations are not allowed on a deleted namespace."
        )
    if _is_missing_collection_physical_error(exc, collection_id):
        coll_label = f"'{collection_name}'" if collection_name else "The collection"
        return (
            f"Collection {coll_label} does not exist (it may have been deleted). "
            "Namespace operations are not allowed on a deleted collection."
        )
    return None


def maybe_reraise_friendly_kernel_error(exc: BaseException) -> None:
    """Re-raise a wrapped ValueError when a namespace-scoped kernel error is recognized."""
    ctx = _NS_KERNEL_ERROR_CTX.get()
    if ctx is None:
        return
    friendly = _friendly_kernel_error_message(
        exc,
        namespace_name=ctx.get("namespace_name"),
        collection_name=ctx.get("collection_name"),
        collection_id=ctx.get("collection_id"),
    )
    if friendly is None:
        return
    raise ValueError(friendly) from exc


@contextlib.contextmanager
def namespace_kernel_error_scope(
    *,
    namespace_name: str | None = None,
    collection_name: str | None = None,
    collection_id: str | None = None,
):
    """Attach namespace/collection context used to translate kernel SQL errors."""
    token = _NS_KERNEL_ERROR_CTX.set({
        "namespace_name": namespace_name or "",
        "collection_name": collection_name or "",
        "collection_id": collection_id or "",
    })
    try:
        yield
    finally:
        _NS_KERNEL_ERROR_CTX.reset(token)


_F = TypeVar("_F", bound=Callable[..., Any])


def namespace_kernel_error_guard(method: _F) -> _F:
    """Decorator for ``BaseClient._namespace_*`` methods with standard leading args."""

    @functools.wraps(method)
    def wrapper(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        *args: Any,
        **kwargs: Any,
    ):
        with namespace_kernel_error_scope(
            namespace_name=namespace_name,
            collection_name=collection_name,
            collection_id=str(collection_id) if collection_id is not None else "",
        ):
            return method(self, collection_id, collection_name, namespace_id, namespace_name, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
