"""Build hybrid-search and SQL document filter expressions from ``where_document``."""

from __future__ import annotations

import re
from typing import Any

_DOCUMENT_FIELD = "document"
_SCORING_LEAF_KEYS = frozenset({"query_string", "match", "multi_match", "match_phrase"})
# Lucene query_string reserved characters (Elasticsearch-style).
_QUERY_STRING_RESERVED_RE = re.compile(r'([+\-=&|><!(){}\[\]^"~*?:\\/])')


def _escape_query_string_term(text: str) -> str:
    """Escape Lucene ``query_string`` syntax characters in a user term."""
    return _QUERY_STRING_RESERVED_RE.sub(r"\\\1", text)


def _is_atomic_query_string_term(text: str) -> bool:
    """Return whether *text* is a single token safe for fast-path ``$and``/``$or`` joining."""
    if not text or re.search(r"\s", text):
        return False
    return _QUERY_STRING_RESERVED_RE.search(text) is None


def _query_string_contains(query: str, *, boost: float | None = None) -> dict[str, Any]:
    """Build a ``query_string`` leaf for document full-text match."""
    body: dict[str, Any] = {
        "fields": [_DOCUMENT_FIELD],
        "query": _escape_query_string_term(query),
    }
    if boost is not None:
        body["boost"] = boost
    return {"query_string": body}


def _regexp_document(pattern: str, *, boost: float | None = None) -> dict[str, Any]:
    """Build a regexp leaf on the document column."""
    body: dict[str, Any] = {"value": pattern}
    if boost is not None:
        body["boost"] = boost
    return {"regexp": {_DOCUMENT_FIELD: body}}


def build_document_hybrid_expression(
    where_document: dict[str, Any] | str,
    *,
    boost: float | None = None,
) -> dict[str, Any] | None:
    """
    Recursively translate ``where_document`` into a hybrid_search document expression.

    Supports ``$contains``, ``$not_contains``, ``$regex``, ``$and``, ``$or`` and nesting.
    """
    if isinstance(where_document, str):
        return _query_string_contains(where_document, boost=boost)

    if not isinstance(where_document, dict) or not where_document:
        return None

    if len(where_document) == 1:
        op, value = next(iter(where_document.items()))
        if op == "$contains" and isinstance(value, str):
            return _query_string_contains(value, boost=boost)
        if op == "$not_contains" and isinstance(value, str):
            return {
                "bool": {
                    "must_not": [_query_string_contains(value, boost=boost)],
                }
            }
        if op == "$regex" and isinstance(value, str):
            return _regexp_document(value, boost=boost)
        if op == "$and" and isinstance(value, list):
            return _combine_document_bool(value, combiner="and", boost=boost)
        if op == "$or" and isinstance(value, list):
            return _combine_document_bool(value, combiner="or", boost=boost)

    # Multiple top-level keys: treat as implicit AND.
    parts = [build_document_hybrid_expression({key: val}, boost=boost) for key, val in where_document.items()]
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"bool": {"must": parts}}


def _contains_operator(where_document: dict[str, Any] | str, operator: str) -> bool:
    """Return whether *operator* appears anywhere in a where_document tree."""
    if isinstance(where_document, str):
        return False
    if not isinstance(where_document, dict):
        return False
    if operator in where_document:
        return True
    for key in ("$and", "$or"):
        children = where_document.get(key)
        if isinstance(children, list):
            for child in children:
                if _contains_operator(child, operator):
                    return True
    return False


def where_document_knn_prefilterable(where_document: dict[str, Any] | str | None) -> bool:
    """
    Whether *where_document* can be enforced via ``knn.filter`` on OceanBase.

    Negative predicates (``$not_contains``) and ``$regex`` are not supported inside
    knn filters and must be applied as SDK post-filters instead.
    """
    if where_document is None:
        return False
    if _contains_operator(where_document, "$not_contains"):
        return False
    return not _contains_operator(where_document, "$regex")


def doc_matches_where_document(document: str, where_document: dict[str, Any] | str) -> bool:
    """Evaluate a where_document predicate against a single document string."""
    text = document.lower()
    if isinstance(where_document, str):
        return where_document.lower() in text
    if "$contains" in where_document:
        return str(where_document["$contains"]).lower() in text
    if "$not_contains" in where_document:
        return str(where_document["$not_contains"]).lower() not in text
    if "$regex" in where_document:
        return re.search(str(where_document["$regex"]), document) is not None
    if "$and" in where_document:
        return all(doc_matches_where_document(document, sub) for sub in where_document["$and"])
    if "$or" in where_document:
        return any(doc_matches_where_document(document, sub) for sub in where_document["$or"])
    raise ValueError(f"Unsupported where_document: {where_document!r}")


def _combine_document_bool(
    conditions: list[Any],
    *,
    combiner: str,
    boost: float | None,
) -> dict[str, Any] | None:
    """Combine child ``where_document`` clauses with AND or OR."""
    if combiner == "and":
        if all(
            isinstance(c, dict)
            and "$contains" in c
            and isinstance(c["$contains"], str)
            and _is_atomic_query_string_term(c["$contains"])
            for c in conditions
        ):
            queries = [c["$contains"] for c in conditions if isinstance(c, dict)]
            body: dict[str, Any] = {
                "fields": [_DOCUMENT_FIELD],
                "query": " ".join(_escape_query_string_term(q) for q in queries),
                "default_operator": "and",
            }
            if boost is not None:
                body["boost"] = boost
            return {"query_string": body}

        if all(
            isinstance(c, dict) and "$not_contains" in c and isinstance(c["$not_contains"], str) for c in conditions
        ):
            return {
                "bool": {
                    "must_not": [
                        _query_string_contains(c["$not_contains"], boost=boost)
                        for c in conditions
                        if isinstance(c, dict)
                    ],
                    "filter": [{"exists": {"field": _DOCUMENT_FIELD}}],
                }
            }

    if combiner == "or" and all(
        isinstance(c, dict)
        and "$contains" in c
        and isinstance(c["$contains"], str)
        and _is_atomic_query_string_term(c["$contains"])
        for c in conditions
    ):
        queries = [c["$contains"] for c in conditions if isinstance(c, dict)]
        body: dict[str, Any] = {
            "fields": [_DOCUMENT_FIELD],
            "query": " ".join(_escape_query_string_term(q) for q in queries),
            "default_operator": "or",
        }
        if boost is not None:
            body["boost"] = boost
        return {"query_string": body}

    child_exprs: list[dict[str, Any]] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        expr = build_document_hybrid_expression(condition, boost=boost)
        if expr is not None:
            child_exprs.append(expr)

    if not child_exprs:
        return None

    if combiner == "and":
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        for expr in child_exprs:
            neg = _pure_must_not_clauses(expr)
            if neg is not None:
                must_not.extend(neg)
            else:
                must.append(expr)
        bool_q: dict[str, Any] = {}
        if must:
            bool_q["must"] = must
        if must_not:
            bool_q["must_not"] = must_not
            if not must:
                bool_q["filter"] = [{"exists": {"field": _DOCUMENT_FIELD}}]
        if not bool_q:
            return None
        return {"bool": bool_q}

    return {
        "bool": {
            "should": child_exprs,
            "minimum_should_match": 1,
        }
    }


def _pure_must_not_clauses(expr: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Return inner ``must_not`` leaves when *expr* is a pure negation bool."""
    if not isinstance(expr, dict) or set(expr.keys()) != {"bool"}:
        return None
    bool_node = expr.get("bool")
    if not isinstance(bool_node, dict) or set(bool_node.keys()) != {"must_not"}:
        return None
    clauses = bool_node.get("must_not")
    if not isinstance(clauses, list):
        return None
    return clauses


def document_expr_as_knn_filter(doc_expr: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Wrap a document expression for use inside ``knn.filter``.

    Scoring leaves (``query_string`` / ``match``) are wrapped in a non-scoring bool.
    Pure ``must_not`` bools gain a permissive positive ``filter`` leaf (OB requirement).
    """
    if doc_expr is None:
        return None

    if "bool" in doc_expr:
        bool_node = dict(doc_expr["bool"])
        must_not = bool_node.get("must_not")
        if must_not and not bool_node.get("must") and not bool_node.get("filter"):
            bool_node.setdefault("filter", [{"exists": {"field": _DOCUMENT_FIELD}}])
        return {"bool": bool_node}

    if _SCORING_LEAF_KEYS & doc_expr.keys():
        return {"bool": {"must": [doc_expr]}}

    return doc_expr
