"""
Namespace DML integration tests.

Small-scale: add / update / upsert / delete / get.
Large-scale (>=1000 rows): count / peek boundaries and multi-namespace orthogonality.
"""

from __future__ import annotations

import pytest
from namespace_dml_helpers import (
    LARGE_DML_CORPUS_SIZE,
    assert_get_absent,
    assert_get_present,
    assert_peek_result,
    build_large_dml_corpus,
    cleanup,
    corpus_id_set,
    create_ns_collection,
    load_dml_corpus,
)


def _create_namespace(collection, name: str):
    """Create namespace."""
    ns = collection.create_namespace(name)
    ns.prewarm()
    return ns


class TestNamespaceDML:
    """Small-scale DML smoke tests."""

    def test_add_single(self, db_client):
        """Test add single."""
        collection = create_ns_collection(db_client, suffix="_add1")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0], documents="Hello", metadatas={"tag": "a"})
            result = ns.get(ids="d1")
            assert len(result["ids"]) == 1
            assert result["ids"][0] == "d1"
        finally:
            cleanup(db_client, collection)

    def test_ops_blocked_after_namespace_deleted(self, db_client):
        """Test ops blocked after namespace deleted."""
        collection = create_ns_collection(db_client, suffix="_delns")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0])
            collection.delete_namespace("dml_ns")
            with pytest.raises(ValueError, match="no longer exists"):
                ns.add(ids="d2", embeddings=[4.0, 5.0, 6.0])
            with pytest.raises(ValueError, match="no longer exists"):
                ns.get(ids="d1")
            with pytest.raises(ValueError, match="no longer exists"):
                ns.query(query_embeddings=[1.0, 2.0, 3.0], n_results=1)
            with pytest.raises(ValueError, match="no longer exists"):
                ns.count()
            with pytest.raises(ValueError, match="no longer exists"):
                ns.prewarm()
        finally:
            cleanup(db_client, collection)

    def test_ops_blocked_after_collection_deleted(self, db_client):
        """Test ops blocked after collection deleted."""
        collection = create_ns_collection(db_client, suffix="_delcoll")
        ns = _create_namespace(collection, "dml_ns")
        ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0])
        db_client.delete_collection(name=collection.name)
        with pytest.raises(ValueError, match=r"no longer exists|does not exist"):
            ns.add(ids="d2", embeddings=[4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match=r"no longer exists|does not exist"):
            ns.query(query_embeddings=[1.0, 2.0, 3.0], n_results=1)

    def test_add_batch(self, db_client):
        """Test add batch."""
        collection = create_ns_collection(db_client, suffix="_addb")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(
                ids=["d1", "d2", "d3"],
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                documents=["Doc A", "Doc B", "Doc C"],
                metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
            )
            result = ns.get(ids=["d1", "d2", "d3"])
            assert len(result["ids"]) == 3
        finally:
            cleanup(db_client, collection)

    def test_get_by_id(self, db_client):
        """Test get by id."""
        collection = create_ns_collection(db_client, suffix="_getid")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(
                ids=["g1", "g2"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                documents=["First", "Second"],
                metadatas=[{"k": 1}, {"k": 2}],
            )
            result = ns.get(ids="g1", include=["documents", "metadatas"])
            assert len(result["ids"]) == 1
            assert result["documents"][0] == "First"
            assert result["metadatas"][0]["k"] == 1
        finally:
            cleanup(db_client, collection)

    def test_get_with_limit(self, db_client):
        """Test get with limit."""
        collection = create_ns_collection(db_client, suffix="_getlim")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(
                ids=["l1", "l2", "l3"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
            result = ns.get(limit=2)
            assert len(result["ids"]) == 2
        finally:
            cleanup(db_client, collection)

    def test_update_metadata(self, db_client):
        """Test update metadata."""
        collection = create_ns_collection(db_client, suffix="_upd")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(ids="u1", embeddings=[1.0, 2.0, 3.0], metadatas={"score": 10})
            ns.update(ids="u1", metadatas={"score": 99})
            result = ns.get(ids="u1", include=["metadatas"])
            assert result["metadatas"][0]["score"] == 99
        finally:
            cleanup(db_client, collection)

    def test_update_document_and_embedding(self, db_client):
        """Test update document and embedding."""
        collection = create_ns_collection(db_client, suffix="_upddoc")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(ids="u2", embeddings=[1.0, 2.0, 3.0], documents="Original")
            ns.update(ids="u2", embeddings=[9.0, 8.0, 7.0], documents="Updated")
            result = ns.get(ids="u2", include=["documents", "embeddings"])
            assert result["documents"][0] == "Updated"
        finally:
            cleanup(db_client, collection)

    def test_upsert_existing(self, db_client):
        """Test upsert existing."""
        collection = create_ns_collection(db_client, suffix="_upsex")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(ids="up1", embeddings=[1.0, 2.0, 3.0], metadatas={"v": 1})
            ns.upsert(ids="up1", embeddings=[4.0, 5.0, 6.0], metadatas={"v": 2})
            result = ns.get(ids="up1", include=["metadatas"])
            assert result["metadatas"][0]["v"] == 2
        finally:
            cleanup(db_client, collection)

    def test_upsert_new(self, db_client):
        """Test upsert new."""
        collection = create_ns_collection(db_client, suffix="_upsnew")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.upsert(ids="up_new", embeddings=[1.0, 1.0, 1.0], metadatas={"fresh": True})
            result = ns.get(ids="up_new", include=["metadatas"])
            assert len(result["ids"]) == 1
            assert result["metadatas"][0]["fresh"] is True
        finally:
            cleanup(db_client, collection)

    def test_delete_by_ids(self, db_client):
        """Test delete by ids."""
        collection = create_ns_collection(db_client, suffix="_del")
        ns = _create_namespace(collection, "dml_ns")
        try:
            ns.add(
                ids=["del1", "del2"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            ns.delete(ids="del1")
            assert len(ns.get(ids="del1")["ids"]) == 0
            assert len(ns.get(ids="del2")["ids"]) == 1
        finally:
            cleanup(db_client, collection)


class TestNamespaceDMLCountPeekAtScale:
    """count / peek with >=1000 rows in a single namespace."""

    @pytest.fixture
    def large_ns(self, db_client):
        """Large ns."""
        collection = create_ns_collection(db_client, suffix="_large_cp")
        ns = _create_namespace(collection, "large_ns")
        corpus = build_large_dml_corpus(LARGE_DML_CORPUS_SIZE, id_prefix="large")
        load_dml_corpus(ns, corpus)
        yield collection, ns, corpus
        cleanup(db_client, collection)

    def test_count_empty_namespace(self, db_client):
        """Test count empty namespace."""
        collection = create_ns_collection(db_client, suffix="_cnt_empty")
        ns = _create_namespace(collection, "empty_ns")
        try:
            assert ns.count() == 0
        finally:
            cleanup(db_client, collection)

    def test_count_after_bulk_load_1000(self, large_ns):
        """Test count after bulk load 1000."""
        _, ns, corpus = large_ns
        assert ns.count() == len(corpus) == LARGE_DML_CORPUS_SIZE

    def test_count_after_partial_delete(self, large_ns):
        """Test count after partial delete."""
        _, ns, corpus = large_ns
        to_delete = [corpus[i].doc_id for i in range(100)]
        ns.delete(ids=to_delete)
        assert ns.count() == LARGE_DML_CORPUS_SIZE - 100
        assert len(ns.get(ids=to_delete)["ids"]) == 0
        assert len(ns.get(ids=corpus[500].doc_id)["ids"]) == 1

    def test_peek_empty_namespace(self, db_client):
        """Test peek empty namespace."""
        collection = create_ns_collection(db_client, suffix="_peek_empty")
        ns = _create_namespace(collection, "peek_empty")
        try:
            result = ns.peek(limit=10)
            assert_peek_result(result, expected_len=0)
        finally:
            cleanup(db_client, collection)

    @pytest.mark.parametrize(
        ("limit", "expected_len"),
        [
            (0, 0),
            (1, 1),
            (10, 10),
            (100, 100),
            (999, 999),
            (1000, 1000),
            (1001, 1000),
            (5000, 1000),
        ],
        ids=[
            "limit_0",
            "limit_1",
            "limit_10",
            "limit_100",
            "limit_999",
            "limit_1000",
            "limit_over_total",
            "limit_far_over_total",
        ],
    )
    def test_peek_limit_boundaries(self, large_ns, limit, expected_len):
        """Test peek limit boundaries."""
        _, ns, corpus = large_ns
        allowed = corpus_id_set(corpus)
        result = ns.peek(limit=limit)
        assert_peek_result(result, expected_len=expected_len, allowed_ids=allowed)

    def test_peek_default_limit_is_10(self, large_ns):
        """Test peek default limit is 10."""
        _, ns, corpus = large_ns
        result = ns.peek()
        assert_peek_result(
            result,
            expected_len=10,
            allowed_ids=corpus_id_set(corpus),
        )


class TestNamespaceDMLCountPeekMultiNs:
    """count / peek orthogonality: same collection, different namespaces at scale."""

    NS_A_SIZE = 1000
    NS_B_SIZE = 800

    @pytest.fixture
    def dual_ns_ctx(self, db_client):
        """Dual ns ctx."""
        collection = create_ns_collection(db_client, suffix="_dual_cp")
        ns_a = _create_namespace(collection, "ns_alpha")
        ns_b = _create_namespace(collection, "ns_beta")
        corpus_a = build_large_dml_corpus(self.NS_A_SIZE, id_prefix="alpha", ns_tag="alpha")
        corpus_b = build_large_dml_corpus(self.NS_B_SIZE, id_prefix="beta", ns_tag="beta")
        load_dml_corpus(ns_a, corpus_a)
        load_dml_corpus(ns_b, corpus_b)
        ctx = {
            "collection": collection,
            "ns_a": ns_a,
            "ns_b": ns_b,
            "corpus_a": corpus_a,
            "corpus_b": corpus_b,
            "ids_a": corpus_id_set(corpus_a),
            "ids_b": corpus_id_set(corpus_b),
        }
        yield ctx
        cleanup(db_client, collection)

    def test_count_isolated_per_namespace(self, dual_ns_ctx):
        """Test count isolated per namespace."""
        assert dual_ns_ctx["ns_a"].count() == self.NS_A_SIZE
        assert dual_ns_ctx["ns_b"].count() == self.NS_B_SIZE

    @pytest.mark.parametrize(
        ("limit", "expected_len"),
        [(1, 1), (10, 10), (100, 100), (800, 800), (1000, 1000), (1500, 1000)],
        ids=["lim_1", "lim_10", "lim_100", "lim_800", "lim_1000", "lim_over_alpha"],
    )
    def test_peek_alpha_boundaries(self, dual_ns_ctx, limit, expected_len):
        """Test peek alpha boundaries."""
        result = dual_ns_ctx["ns_a"].peek(limit=limit)
        assert_peek_result(
            result,
            expected_len=expected_len,
            allowed_ids=dual_ns_ctx["ids_a"],
            ns_tag="alpha",
        )

    @pytest.mark.parametrize(
        ("limit", "expected_len"),
        [(1, 1), (10, 10), (500, 500), (800, 800), (1000, 800), (2000, 800)],
        ids=["lim_1", "lim_10", "lim_500", "lim_800", "lim_over_beta", "lim_far_over"],
    )
    def test_peek_beta_boundaries(self, dual_ns_ctx, limit, expected_len):
        """Test peek beta boundaries."""
        result = dual_ns_ctx["ns_b"].peek(limit=limit)
        assert_peek_result(
            result,
            expected_len=expected_len,
            allowed_ids=dual_ns_ctx["ids_b"],
            ns_tag="beta",
        )

    def test_peek_does_not_leak_across_namespaces(self, dual_ns_ctx):
        """Test peek does not leak across namespaces."""
        peek_a = dual_ns_ctx["ns_a"].peek(limit=50)
        peek_b = dual_ns_ctx["ns_b"].peek(limit=50)
        assert_peek_result(
            peek_a,
            expected_len=50,
            allowed_ids=dual_ns_ctx["ids_a"],
            ns_tag="alpha",
        )
        assert_peek_result(
            peek_b,
            expected_len=50,
            allowed_ids=dual_ns_ctx["ids_b"],
            ns_tag="beta",
        )
        assert set(peek_a["ids"]).isdisjoint(dual_ns_ctx["ids_b"])
        assert set(peek_b["ids"]).isdisjoint(dual_ns_ctx["ids_a"])

    def test_delete_in_one_namespace_only_affects_its_count(self, dual_ns_ctx):
        """Test delete in one namespace only affects its count."""
        ns_a = dual_ns_ctx["ns_a"]
        ns_b = dual_ns_ctx["ns_b"]
        corpus_a = dual_ns_ctx["corpus_a"]
        delete_ids = [corpus_a[i].doc_id for i in range(150)]
        ns_a.delete(ids=delete_ids)

        assert ns_a.count() == self.NS_A_SIZE - 150
        assert ns_b.count() == self.NS_B_SIZE

        peek_a = ns_a.peek(limit=20)
        assert_peek_result(
            peek_a,
            expected_len=20,
            allowed_ids=dual_ns_ctx["ids_a"] - set(delete_ids),
            ns_tag="alpha",
        )
        assert_peek_result(
            ns_b.peek(limit=20),
            expected_len=20,
            allowed_ids=dual_ns_ctx["ids_b"],
            ns_tag="beta",
        )


class TestNamespaceDMLFullCycle:
    """End-to-end DML cycle (small scale, verified by get)."""

    def test_full_dml_cycle_verified_by_get(self, db_client):
        """Test full dml cycle verified by get."""
        collection = create_ns_collection(db_client, suffix="_cycle")
        ns = _create_namespace(collection, "cycle_ns")
        doc_id = "cycle_doc"
        try:
            ns.add(
                ids=doc_id,
                embeddings=[0.1, 0.2, 0.3],
                documents="Original document",
                metadatas={"stage": "added"},
            )
            assert_get_present(
                ns,
                doc_id,
                embeddings=[0.1, 0.2, 0.3],
                documents="Original document",
                metadatas={"stage": "added"},
            )

            ns.update(
                ids=doc_id,
                embeddings=[0.4, 0.5, 0.6],
                documents="Updated document",
                metadatas={"stage": "updated"},
            )
            assert_get_present(
                ns,
                doc_id,
                embeddings=[0.4, 0.5, 0.6],
                documents="Updated document",
                metadatas={"stage": "updated"},
            )

            ns.upsert(
                ids=doc_id,
                embeddings=[0.7, 0.8, 0.9],
                documents="Upserted document",
                metadatas={"stage": "upserted"},
            )
            assert_get_present(
                ns,
                doc_id,
                embeddings=[0.7, 0.8, 0.9],
                documents="Upserted document",
                metadatas={"stage": "upserted"},
            )

            ns.delete(ids=doc_id)
            assert_get_absent(ns, doc_id)
            assert ns.count() == 0
        finally:
            cleanup(db_client, collection)

    def test_batch_add_then_get_each(self, db_client):
        """Test batch add then get each."""
        collection = create_ns_collection(db_client, suffix="_batch")
        ns = _create_namespace(collection, "batch_ns")
        try:
            ns.add(
                ids=["b1", "b2", "b3"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                documents=["Doc 1", "Doc 2", "Doc 3"],
                metadatas=[{"n": 1}, {"n": 2}, {"n": 3}],
            )
            result = ns.get(ids=["b1", "b2", "b3"], include=["documents", "metadatas"])
            assert len(result["ids"]) == 3
            assert set(result["ids"]) == {"b1", "b2", "b3"}
            assert set(result["documents"]) == {"Doc 1", "Doc 2", "Doc 3"}
            assert {m["n"] for m in result["metadatas"]} == {1, 2, 3}
        finally:
            cleanup(db_client, collection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
