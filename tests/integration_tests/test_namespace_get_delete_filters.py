"""
Namespace conditional get/delete integration tests (batch A / P0).

Covers:
  - delete(where=...) / delete(where_document=...)
  - get(where=...) / get(where_document=...)
  - multi-namespace isolation for metadata filter delete
  - cross-namespace get(where=...) on a shared connection (regression)
"""

from __future__ import annotations

import pytest
from namespace_dml_helpers import (
    FILTER_MULTI_NS_KEEP,
    FILTER_MULTI_NS_PURGE,
    assert_count_after_document_delete,
    assert_get_where_count,
    assert_get_where_document_count,
    build_filter_corpus,
    cleanup,
    create_ns_collection,
    load_dml_corpus,
    load_filter_corpus,
    precheck_where_document_hits,
)


class TestNamespaceDeleteWhere:
    """P0 #1-#2: conditional delete on namespace."""

    def test_delete_where_metadata_tag(self, db_client):
        """Test delete where metadata tag."""
        collection = create_ns_collection(db_client, suffix="_del_where")
        ns = collection.create_namespace("del_where_ns")
        ns.prewarm()
        corpus, gt = build_filter_corpus()
        try:
            load_dml_corpus(ns, corpus)
            assert ns.count() == gt["total"]

            ns.delete(where={"tag": "purge"})

            assert ns.count() == gt["keep"]
            assert_get_where_count(ns, {"tag": "purge"}, 0)
            assert_get_where_count(
                ns,
                {"tag": "keep"},
                gt["keep"],
                limit=gt["keep"],
                check_meta={"tag": "keep"},
            )
        finally:
            cleanup(db_client, collection)

    def test_delete_where_document_obsolete(self, db_client):
        """Test delete where document obsolete."""
        collection = create_ns_collection(db_client, suffix="_del_wdoc")
        ns = collection.create_namespace("del_wdoc_ns")
        ns.prewarm()
        corpus, gt = build_filter_corpus()
        where_document = {"$contains": "obsolete"}
        try:
            load_filter_corpus(ns, corpus)
            pre = precheck_where_document_hits(
                ns,
                where_document,
                gt["doc_obsolete"],
                context="pre-delete FTS probe",
            )
            ns.delete(where_document=where_document)

            assert_count_after_document_delete(
                ns,
                where_document=where_document,
                expected_count=gt["keep"],
                total_before=gt["total"],
                expected_removed=gt["doc_obsolete"],
                pre_delete_hit_count=len(pre["ids"]),
            )
            assert_get_where_document_count(
                ns,
                where_document,
                0,
                limit=10,
                substring="obsolete",
                context="post-delete",
            )
            assert_get_where_count(ns, {"tag": "keep"}, gt["keep"], limit=gt["keep"])
        finally:
            cleanup(db_client, collection)


class TestNamespaceGetWhere:
    """P0 #3-#4: conditional get on namespace."""

    def test_get_where_metadata_category_ai(self, db_client):
        """Test get where metadata category ai."""
        collection = create_ns_collection(db_client, suffix="_get_where")
        ns = collection.create_namespace("get_where_ns")
        ns.prewarm()
        corpus, gt = build_filter_corpus()
        try:
            load_dml_corpus(ns, corpus)

            assert_get_where_count(
                ns,
                {"category": "AI"},
                gt["category_ai"],
                limit=gt["category_ai"],
                check_meta={"category": "AI"},
            )
            # Capped limit returns fewer rows but all still match.
            capped = assert_get_where_count(
                ns,
                {"category": "AI"},
                10,
                limit=10,
                check_meta={"category": "AI"},
            )
            assert len(capped["ids"]) == 10
        finally:
            cleanup(db_client, collection)

    def test_get_where_document_contains_python(self, db_client):
        """Test get where document contains python."""
        collection = create_ns_collection(db_client, suffix="_get_wdoc")
        ns = collection.create_namespace("get_wdoc_ns")
        ns.prewarm()
        corpus, gt = build_filter_corpus()
        where_document = {"$contains": "python"}
        try:
            load_filter_corpus(ns, corpus)

            assert_get_where_document_count(
                ns,
                where_document,
                gt["doc_python"],
                limit=gt["doc_python"],
                substring="python",
                context="full match",
            )
            assert_get_where_document_count(
                ns,
                where_document,
                5,
                limit=5,
                substring="python",
                context="capped limit",
            )
        finally:
            cleanup(db_client, collection)


class TestNamespaceFilterMultiNs:
    """P0 #5: filter delete isolated per namespace (same collection)."""

    def test_delete_where_only_affects_target_namespace(self, db_client):
        """Test delete where only affects target namespace."""
        collection = create_ns_collection(db_client, suffix="_filt_mns")
        ns_a = collection.create_namespace("ns_alpha")
        ns_b = collection.create_namespace("ns_beta")
        ns_a.prewarm()
        ns_b.prewarm()
        corpus_a, gt = build_filter_corpus(
            keep_count=FILTER_MULTI_NS_KEEP,
            purge_count=FILTER_MULTI_NS_PURGE,
            ai_in_keep=200,
            python_in_keep=80,
            id_prefix="alpha",
            ns_tag="alpha",
        )
        corpus_b, _ = build_filter_corpus(
            keep_count=FILTER_MULTI_NS_KEEP,
            purge_count=FILTER_MULTI_NS_PURGE,
            ai_in_keep=200,
            python_in_keep=80,
            id_prefix="beta",
            ns_tag="beta",
        )
        try:
            load_dml_corpus(ns_a, corpus_a)
            load_dml_corpus(ns_b, corpus_b)
            assert ns_a.count() == gt["total"]
            assert ns_b.count() == gt["total"]

            ns_a.delete(where={"tag": "purge"})

            assert ns_a.count() == gt["keep"]
            assert ns_b.count() == gt["total"]

            assert_get_where_count(ns_a, {"tag": "purge"}, 0)
            assert_get_where_count(
                ns_b,
                {"tag": "purge"},
                10,
                limit=10,
                check_meta={"tag": "purge", "ns_tag": "beta"},
            )
            assert_get_where_count(
                ns_a,
                {"tag": "keep"},
                min(20, gt["keep"]),
                limit=20,
                check_meta={"tag": "keep", "ns_tag": "alpha"},
            )
        finally:
            cleanup(db_client, collection)


class TestNamespaceCrossNsGetWhere:
    """Regression: get(where=...) after switching namespace on one connection.

    Client sets @collection_id, @namespace_id, and @ltable_id before each DQL call;
    OB logic-table kernel must honor the session context when metadata filters are used.
    """

    def test_get_where_alternates_namespaces_after_delete(self, db_client):
        """Test get where alternates namespaces after delete."""
        collection = create_ns_collection(db_client, suffix="_cross_ns_get")
        ns_a = collection.create_namespace("cross_alpha")
        ns_b = collection.create_namespace("cross_beta")
        ns_a.prewarm()
        ns_b.prewarm()
        corpus_a, gt = build_filter_corpus(
            keep_count=50,
            purge_count=50,
            id_prefix="alpha",
            ns_tag="alpha",
        )
        corpus_b, _ = build_filter_corpus(
            keep_count=50,
            purge_count=50,
            id_prefix="beta",
            ns_tag="beta",
        )
        try:
            load_dml_corpus(ns_a, corpus_a)
            load_dml_corpus(ns_b, corpus_b)
            ns_a.delete(where={"tag": "purge"})

            assert_get_where_count(ns_a, {"tag": "purge"}, 0)
            assert_get_where_count(
                ns_b,
                {"tag": "purge"},
                10,
                limit=10,
                check_meta={"tag": "purge", "ns_tag": "beta"},
            )
            assert_get_where_count(
                ns_a,
                {"tag": "keep"},
                min(20, gt["keep"]),
                limit=20,
                check_meta={"tag": "keep", "ns_tag": "alpha"},
            )
        finally:
            cleanup(db_client, collection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
