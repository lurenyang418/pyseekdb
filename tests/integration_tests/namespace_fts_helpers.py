"""
Helpers for namespace hybrid_search full-text integration tests.

Ground-truth matching uses substring checks on document text (aligned with
``$contains`` / ``$not_contains`` semantics). Relevance ranking uses term
occurrence counts so the highest-ranked documents are deterministic.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import FulltextIndexConfig, VectorIndexConfig
from pyseekdb.client.schema import Schema

VectorDistanceMetric = Literal["l2", "cosine"]
VECTOR_DISTANCE_METRICS: tuple[VectorDistanceMetric, ...] = ("l2", "cosine")

# Rare ASCII tokens to reduce IK segmentation surprises in assertions.
TOKEN_ZPX = "TOKENZPX"
TOKEN_ALP = "TOKENALP"
FORBIDDEN_PHRASE = "FORBIDPHRASE"

# Tier base for $or matches that hit multiple branches (query_string OR multi-term boost).
_OR_MULTI_BRANCH_HINT_BASE = 10_000

CORPUS_SIZE = 1101  # strictly > 1000
BATCH_SIZE = 100
INDEX_SETTLE_SECONDS = 3


@dataclass(frozen=True)
class CorpusRecord:
    """CorpusRecord class."""

    doc_id: str
    document: str
    embedding: list[float]
    metadata: dict[str, Any]
    rel_hint: int  # ground-truth relevance tier for positive FTS queries


@dataclass(frozen=True)
class FtsQueryCase:
    """FtsQueryCase class."""

    name: str
    where_document: dict[str, Any] | str
    n_results: int
    check_ranking: bool = True
    where: dict[str, Any] | None = None


def ns_schema(distance: VectorDistanceMetric = "l2") -> Schema:
    """Ns schema."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance=distance, centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
        fulltext_index=FulltextIndexConfig(analyzer="ik"),
    )


def flat_hybrid_search_schema(distance: VectorDistanceMetric = "l2") -> Schema:
    """Schema for flat collection baseline (``use_namespace=False``, HNSW + FTS)."""
    from pyseekdb import HNSWConfiguration

    return Schema(
        vector_index=VectorIndexConfig(
            hnsw=HNSWConfiguration(dimension=3, distance=distance),
            embedding_function=None,
        ),
    )


def build_large_fts_corpus(size: int = CORPUS_SIZE) -> list[CorpusRecord]:
    """Build a deterministic corpus with known full-text relevance tiers."""
    records: list[CorpusRecord] = []

    def _add(
        doc_id: str,
        document: str,
        rel_hint: int,
        embedding: list[float] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Add."""
        idx = len(records)
        meta = {"rel_hint": rel_hint, "seq": idx}
        if extra_meta:
            meta.update(extra_meta)
        records.append(
            CorpusRecord(
                doc_id=doc_id,
                document=document,
                embedding=embedding or _embedding_for_index(idx),
                metadata=meta,
                rel_hint=rel_hint,
            )
        )

    # Tiered hits for TOKEN_ZPX ($contains / string shorthand).
    for repeat in range(5, 0, -1):
        zpx_hint = repeat * 10
        _add(
            f"zpx_top_{repeat}",
            f"Primary {TOKEN_ZPX} " * repeat + f" focus document repeat {repeat}",
            rel_hint=zpx_hint,
            extra_meta={"zpx_hint": zpx_hint, "alp_hint": 0},
        )

    for i in range(15):
        _add(
            f"zpx_mid_{i:02d}",
            f"Secondary mention {TOKEN_ZPX} in document index {i}",
            rel_hint=5,
            extra_meta={"zpx_hint": 5, "alp_hint": 0},
        )

    # Both tokens for $and.
    _add(
        "both_tokens",
        f"Combined {TOKEN_ZPX} and {TOKEN_ALP} in one document for conjunction tests",
        rel_hint=55,
        extra_meta={"zpx_hint": 5, "alp_hint": 5, "has_both": True},
    )

    # TOKEN_ALP tiered hits for $or.
    for repeat in range(4, 0, -1):
        alp_hint = repeat * 8
        _add(
            f"alp_top_{repeat}",
            f"Alpha {TOKEN_ALP} " * repeat + f" branch document repeat {repeat}",
            rel_hint=alp_hint,
            extra_meta={"zpx_hint": 0, "alp_hint": alp_hint},
        )

    for i in range(10):
        _add(
            f"alp_mid_{i:02d}",
            f"Secondary {TOKEN_ALP} mention variant {i}",
            rel_hint=4,
            extra_meta={"zpx_hint": 0, "alp_hint": 4},
        )

    # Rows containing TOKEN_ZPX — must not appear in $not_contains results.
    for i in range(8):
        _add(
            f"zpx_forbid_{i}",
            f"This row contains {TOKEN_ZPX} and must not appear in not_contains results {i}",
            rel_hint=0,
            extra_meta={"zpx_hint": 1, "alp_hint": 0},
        )

    # Filler rows (no rare tokens) — bulk of the corpus.
    while len(records) < size:
        i = len(records)
        _add(
            f"filler_{i:05d}",
            f"Filler document {i} about oceanbase distributed storage systems. "
            f"Neutral content without rare tokens. {FORBIDDEN_PHRASE} anchor {i % 17}",
            rel_hint=1 if i % 50 == 0 else 0,
        )

    if len(records) > size:
        del records[size:]
    return records


def _embedding_for_index(index: int) -> list[float]:
    """Embedding for index."""
    return [
        float((index + 1) % 7) / 7.0,
        float((index + 2) % 5) / 5.0,
        float((index + 3) % 3) / 3.0,
    ]


def doc_matches_where_document(document: str, where_document: dict[str, Any] | str) -> bool:
    """Doc matches where document."""
    text = document.lower()
    if isinstance(where_document, str):
        return where_document.lower() in text
    if "$contains" in where_document:
        return where_document["$contains"].lower() in text
    if "$not_contains" in where_document:
        return where_document["$not_contains"].lower() not in text
    if "$and" in where_document:
        return all(doc_matches_where_document(document, sub) for sub in where_document["$and"])
    if "$or" in where_document:
        return any(doc_matches_where_document(document, sub) for sub in where_document["$or"])
    raise ValueError(f"Unsupported where_document for hybrid_search: {where_document!r}")


def doc_matches_where_metadata(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Evaluate metadata filter (aligned with hybrid_search ``query.where`` semantics)."""
    if "$and" in where:
        return all(doc_matches_where_metadata(metadata, sub) for sub in where["$and"])
    if "$or" in where:
        return any(doc_matches_where_metadata(metadata, sub) for sub in where["$or"])
    if "$not" in where:
        return not doc_matches_where_metadata(metadata, where["$not"])

    for key, value in where.items():
        if key in ("$and", "$or", "$not"):
            continue
        actual = metadata.get(key)
        if isinstance(value, dict):
            if "$eq" in value and actual != value["$eq"]:
                return False
            if "$ne" in value and actual == value["$ne"]:
                return False
            if "$lt" in value and (actual is None or actual >= value["$lt"]):
                return False
            if "$lte" in value and (actual is None or actual > value["$lte"]):
                return False
            if "$gt" in value and (actual is None or actual <= value["$gt"]):
                return False
            if "$gte" in value and (actual is None or actual < value["$gte"]):
                return False
            if "$in" in value and actual not in value["$in"]:
                return False
            if "$nin" in value and actual in value["$nin"]:
                return False
        elif actual != value:
            return False
    return True


def corpus_matches_fts(
    record: CorpusRecord,
    where_document: dict[str, Any] | str,
    where: dict[str, Any] | None = None,
) -> bool:
    """Corpus matches fts."""
    if not doc_matches_where_document(record.document, where_document):
        return False
    return not (where is not None and not doc_matches_where_metadata(record.metadata, where))


def effective_rel_hint(record: CorpusRecord, where_document: dict[str, Any] | str) -> int:
    """Map corpus ``rel_hint`` to the active query (AND uses min; OR uses max or multi-branch sum)."""
    if isinstance(where_document, str):
        if where_document == TOKEN_ZPX:
            return record.metadata.get("zpx_hint", record.rel_hint)
        if where_document == TOKEN_ALP:
            return record.metadata.get("alp_hint", record.rel_hint)
        return record.rel_hint
    if "$contains" in where_document:
        term = where_document["$contains"]
        if term == TOKEN_ZPX:
            return record.metadata.get("zpx_hint", 0)
        if term == TOKEN_ALP:
            return record.metadata.get("alp_hint", 0)
        if term in ("Primary", "Alpha"):
            return record.rel_hint
        return record.rel_hint
    if "$and" in where_document:
        hints = [effective_rel_hint(record, sub) for sub in where_document["$and"]]
        return min(hints) if hints else 0
    if "$or" in where_document:
        hints = [effective_rel_hint(record, sub) for sub in where_document["$or"]]
        active = [h for h in hints if h > 0]
        if not active:
            return 0
        if len(active) >= 2:
            # Matches multiple OR branches (e.g. both TOKENZPX and TOKENALP): ranks above
            # any single-branch hit, consistent with query_string OR + BM25 on OceanBase.
            return _OR_MULTI_BRANCH_HINT_BASE + sum(active)
        return max(active)
    return record.rel_hint


def expected_best_id(
    corpus: list[CorpusRecord],
    where_document: dict[str, Any] | str,
    where: dict[str, Any] | None = None,
) -> str:
    """Expected best id."""
    matched = [
        (rec.doc_id, effective_rel_hint(rec, where_document))
        for rec in corpus
        if corpus_matches_fts(rec, where_document, where)
    ]
    matched.sort(key=lambda item: (-item[1], item[0]))
    return matched[0][0]


def insert_corpus_in_batches(target: Any, corpus: list[CorpusRecord], batch_size: int = BATCH_SIZE) -> None:
    """Bulk ``add`` on a namespace or flat collection target."""
    for start in range(0, len(corpus), batch_size):
        chunk = corpus[start : start + batch_size]
        target.add(
            ids=[r.doc_id for r in chunk],
            embeddings=[r.embedding for r in chunk],
            documents=[r.document for r in chunk],
            metadatas=[r.metadata for r in chunk],
        )


def insert_corpus_into_collection(collection: Any, corpus: list[CorpusRecord]) -> None:
    """``collection.add(...)`` on a flat collection (``use_namespace=False``)."""
    if getattr(collection, "use_namespace", None) is not False:
        raise ValueError(
            f"insert_corpus_into_collection requires use_namespace=False, "
            f"got {getattr(collection, 'use_namespace', None)!r} on {collection!r}"
        )
    insert_corpus_in_batches(collection, corpus)


def insert_corpus_into_namespace(namespace: Any, corpus: list[CorpusRecord]) -> None:
    """``namespace.add(...)`` under a namespace-enabled collection."""
    insert_corpus_in_batches(namespace, corpus)


def assert_score_order_non_increasing(distances: list[float]) -> None:
    """Hybrid search exposes BM25-like scores in ``distances`` (higher = more relevant)."""
    if not distances or len(set(distances)) <= 1:
        return
    for i in range(len(distances) - 1):
        assert distances[i] >= distances[i + 1], (
            f"scores must be non-increasing: index {i}={distances[i]!r} < "
            f"index {i + 1}={distances[i + 1]!r}, full={distances!r}"
        )


def assert_hybrid_fulltext_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    where_document: dict[str, Any] | str,
    n_results: int,
    *,
    where: dict[str, Any] | None = None,
    check_ranking: bool = True,
) -> None:
    """Assert hybrid fulltext result."""
    assert result is not None
    assert result.get("ids")
    ids = result["ids"][0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else []
    documents = result.get("documents", [[]])[0] if result.get("documents") else []
    metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []

    assert len(ids) <= n_results
    assert len(ids) > 0, "expected at least one full-text hit"
    if distances:
        assert len(distances) == len(ids)
        assert_score_order_non_increasing(distances)

    id_to_doc = {rec.doc_id: rec.document for rec in corpus}
    id_to_meta = {rec.doc_id: rec.metadata for rec in corpus}

    meta_rows = metadatas if metadatas else [None] * len(ids)
    for doc_id, doc_text, meta in zip(ids, documents, meta_rows, strict=False):
        assert doc_id in id_to_doc, f"unknown id {doc_id!r} in search results"
        body = doc_text if doc_text is not None else id_to_doc[doc_id]
        assert doc_matches_where_document(body, where_document), (
            f"id={doc_id!r} document={body!r} does not satisfy where_document={where_document!r}"
        )
        if where is not None:
            row_meta = meta if meta is not None else id_to_meta[doc_id]
            assert doc_matches_where_metadata(row_meta, where), (
                f"id={doc_id!r} metadata={row_meta!r} does not satisfy where={where!r}"
            )

    if not check_ranking:
        return

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    hints = [effective_rel_hint(corpus_by_id[doc_id], where_document) for doc_id in ids]

    # 1) Top-1 must be globally most relevant (by corpus rel_hint tiers).
    assert ids[0] == expected_best_id(corpus, where_document, where), (
        f"top-1 must be the highest-relevance document, got {ids[0]!r} "
        f"expected {expected_best_id(corpus, where_document, where)!r}"
    )

    # 2) No matching document outside the window may have a higher tier than the last hit.
    if len(ids) == n_results:
        min_hint_in_page = hints[-1]
        for rec in corpus:
            if not corpus_matches_fts(rec, where_document, where):
                continue
            rec_hint = effective_rel_hint(rec, where_document)
            if rec_hint > min_hint_in_page:
                assert rec.doc_id in ids, (
                    f"higher-relevance match {rec.doc_id!r} (hint={rec_hint}) "
                    f"missing from top-{n_results} (min returned hint={min_hint_in_page})"
                )

    # 3) When the engine returns distinct scores, they must be non-increasing.
    if distances:
        assert_score_order_non_increasing(distances)
        if any(not math.isclose(d, 0.0) for d in distances):
            for i in range(len(ids) - 1):
                assert distances[i] >= distances[i + 1] or math.isclose(distances[i], distances[i + 1])


# All where_document shapes supported by hybrid_search ``_build_document_query``.
HYBRID_SEARCH_FTS_CASES: list[FtsQueryCase] = [
    FtsQueryCase(
        name="contains_token_zpx",
        where_document={"$contains": TOKEN_ZPX},
        n_results=15,
    ),
    FtsQueryCase(
        name="string_shorthand_token_zpx",
        where_document=TOKEN_ZPX,
        n_results=15,
    ),
    FtsQueryCase(
        name="not_contains_token_zpx",
        where_document={"$not_contains": TOKEN_ZPX},
        n_results=20,
        check_ranking=False,
    ),
    FtsQueryCase(
        name="and_zpx_alp",
        where_document={
            "$and": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        n_results=10,
    ),
    FtsQueryCase(
        name="or_zpx_alp",
        where_document={
            "$or": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        n_results=20,
    ),
    FtsQueryCase(
        name="contains_token_alp",
        where_document={"$contains": TOKEN_ALP},
        n_results=15,
    ),
    FtsQueryCase(
        name="and_multi_contains_phrase",
        where_document={
            "$and": [
                {"$contains": "Primary"},
                {"$contains": TOKEN_ZPX},
            ],
        },
        n_results=10,
    ),
    FtsQueryCase(
        name="contains_zpx_filter_has_both",
        where_document={"$contains": TOKEN_ZPX},
        where={"has_both": True},
        n_results=5,
    ),
    FtsQueryCase(
        name="contains_zpx_filter_zpx_hint_50",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": 50},
        n_results=5,
    ),
    FtsQueryCase(
        name="contains_zpx_filter_zpx_hint_gte_40",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        n_results=10,
    ),
    FtsQueryCase(
        name="and_zpx_alp_filter_has_both",
        where_document={
            "$and": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where={"has_both": True},
        n_results=5,
    ),
    FtsQueryCase(
        name="contains_zpx_filter_zpx_hint_and_alp_zero",
        where_document={"$contains": TOKEN_ZPX},
        where={
            "$and": [
                {"zpx_hint": {"$gte": 40}},
                {"alp_hint": 0},
            ],
        },
        n_results=10,
    ),
]


def get_fts_case(name: str) -> FtsQueryCase:
    """Get fts case."""
    for case in HYBRID_SEARCH_FTS_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown FTS case: {name!r}")


def setup_large_fts_collection(
    db_client: Any,
    *,
    distance: VectorDistanceMetric = "l2",
) -> tuple[list[CorpusRecord], Any]:
    """Create one namespace-enabled collection with a deterministic large corpus."""
    corpus = build_large_fts_corpus(CORPUS_SIZE)
    if len(corpus) <= 1000:
        raise ValueError(f"corpus must exceed 1000 rows, got {len(corpus)}")

    name = f"test_ns_hs_ft_{distance}_{int(time.time() * 1000)}"
    collection = db_client.create_collection(
        name=name,
        schema=ns_schema(distance),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    return corpus, collection


def setup_fts_namespace_with_corpus(
    collection: Any,
    corpus: list[CorpusRecord],
    *,
    namespace_name: str,
) -> Any:
    """Create a namespace under an existing collection and bulk-load the corpus."""
    namespace = collection.create_namespace(namespace_name)
    namespace.prewarm()
    insert_corpus_into_namespace(namespace, corpus)
    expected = len(corpus)
    actual = namespace.count()
    assert actual == expected, (
        f"namespace {namespace.name!r} (id={namespace.namespace_id}) "
        f"expected {expected} rows after bulk load, got {actual}"
    )
    time.sleep(INDEX_SETTLE_SECONDS)
    return namespace


def setup_large_fts_namespace(
    db_client: Any, *, namespace_name: str = "ns_hs_ft_large"
) -> tuple[list[CorpusRecord], Any, Any]:
    """Create collection + namespace with corpus (single-test convenience wrapper)."""
    corpus, collection = setup_large_fts_collection(db_client)
    namespace = setup_fts_namespace_with_corpus(collection, corpus, namespace_name=namespace_name)
    return corpus, collection, namespace


def teardown_large_fts_collection(db_client: Any, collection: Any) -> None:
    """Teardown large fts collection."""
    with contextlib.suppress(Exception):
        db_client.delete_collection(name=collection.name)


def teardown_large_fts_namespace(db_client: Any, collection: Any) -> None:
    """Teardown large fts namespace."""
    teardown_large_fts_collection(db_client, collection)


# Quadrant keys: coll_1/coll_2 x ns_x/ns_y (aligned with DML multi-coll-multi-ns tests).
MULTI_COLL_MULTI_NS_QUADRANT_KEYS: tuple[str, ...] = ("c1_x", "c1_y", "c2_x", "c2_y")

# One loaded namespace per collection (OceanBase: FTS index applies to the last-loaded ns in a coll).
MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS: tuple[str, ...] = ("c1_x", "c2_y")


def _create_multi_coll_multi_ns_layout(
    db_client: Any,
    *,
    name_prefix: str,
    distance: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """Create multi coll multi ns layout."""
    ts = int(time.time() * 1000)
    coll_1 = db_client.create_collection(
        name=f"{name_prefix}_{distance}_{ts}_c1",
        schema=ns_schema(distance),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    coll_2 = db_client.create_collection(
        name=f"{name_prefix}_{distance}_{ts}_c2",
        schema=ns_schema(distance),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    ctx: dict[str, Any] = {"coll_1": coll_1, "coll_2": coll_2}
    for coll_tag, collection in (("c1", coll_1), ("c2", coll_2)):
        for ns_suffix in ("x", "y"):
            ns = collection.create_namespace(f"ns_{ns_suffix}")
            ns.prewarm()
            ctx[f"{coll_tag}_{ns_suffix}"] = ns
    return ctx


def setup_multi_coll_multi_ns_fts(
    db_client: Any,
    *,
    distance: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """
    2 collections x 2 namespaces; load the large FTS corpus into ``c1_x`` and ``c2_y`` only.

    Empty quadrants (``c1_y``, ``c2_x``) are kept for isolation checks. Within one collection,
    only the namespace that received the bulk insert is expected to serve hybrid_search FTS
    (see ``setup_same_collection_both_ns_fts`` for same-collection multi-ns behavior).
    """
    ctx = _create_multi_coll_multi_ns_layout(db_client, name_prefix="test_ns_hs_ft_mcmn", distance=distance)
    corpus = build_large_fts_corpus(CORPUS_SIZE)
    if len(corpus) <= 1000:
        raise ValueError(f"corpus must exceed 1000 rows, got {len(corpus)}")

    for key in MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS:
        insert_corpus_in_batches(ctx[key], corpus)

    for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
        ns = ctx[key]
        ctx[key] = (corpus, ns)

    ctx["_corpus"] = corpus
    time.sleep(INDEX_SETTLE_SECONDS)
    return ctx


def setup_same_collection_both_ns_fts(db_client: Any) -> dict[str, Any]:
    """
    Single collection with ``ns_x`` and ``ns_y``; corpus inserted into both (x then y).

    hybrid_search FTS is expected on the last-loaded namespace (``ns_y``); ``ns_x`` rows
    remain visible via get.
    """
    ts = int(time.time() * 1000)
    collection = db_client.create_collection(
        name=f"test_ns_hs_ft_sc2ns_{ts}",
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    corpus = build_large_fts_corpus(CORPUS_SIZE)
    ns_x = collection.create_namespace("ns_x")
    ns_y = collection.create_namespace("ns_y")
    ns_x.prewarm()
    ns_y.prewarm()
    insert_corpus_in_batches(ns_x, corpus)
    insert_corpus_in_batches(ns_y, corpus)
    time.sleep(INDEX_SETTLE_SECONDS)
    return {
        "collection": collection,
        "corpus": corpus,
        "ns_x": (corpus, ns_x),
        "ns_y": (corpus, ns_y),
    }


def setup_multi_coll_multi_ns_fts_single_loaded(
    db_client: Any,
    *,
    loaded_quadrant: str = "c1_x",
    distance: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """Same layout as :func:`setup_multi_coll_multi_ns_fts` but corpus only in ``loaded_quadrant``."""
    if loaded_quadrant not in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
        raise ValueError(f"unknown quadrant {loaded_quadrant!r}")

    ctx = _create_multi_coll_multi_ns_layout(db_client, name_prefix="test_ns_hs_ft_mcmn1", distance=distance)
    corpus = build_large_fts_corpus(CORPUS_SIZE)
    insert_corpus_in_batches(ctx[loaded_quadrant], corpus)

    for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
        ns = ctx[key]
        ctx[key] = (corpus, ns)

    ctx["_corpus"] = corpus
    time.sleep(INDEX_SETTLE_SECONDS)
    return ctx


def teardown_multi_coll_multi_ns_fts(db_client: Any, ctx: dict[str, Any]) -> None:
    """Teardown multi coll multi ns fts."""
    for coll_key in ("coll_1", "coll_2", "collection"):
        collection = ctx.get(coll_key)
        if collection is not None:
            with contextlib.suppress(Exception):
                db_client.delete_collection(name=collection.name)


def assert_hybrid_search_no_hits(
    namespace: Any,
    where_document: dict[str, Any] | str,
    *,
    n_results: int = 15,
) -> dict[str, Any]:
    """Assert hybrid_search returns no rows (namespace isolation)."""
    result = namespace.hybrid_search(
        query={"where_document": where_document, "n_results": n_results},
        n_results=n_results,
        include=["documents"],
    )
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    assert len(ids) == 0, f"expected no full-text hits in namespace {namespace.name!r}, got {len(ids)} ids: {ids[:5]!r}"
    return result


def run_hybrid_search_fts_case_on_quadrants(
    ctx: dict[str, Any],
    case: FtsQueryCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
) -> None:
    """Run one FTS case on the selected quadrants."""
    fts_case = case if isinstance(case, FtsQueryCase) else get_fts_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_search_fts_case(namespace, corpus, fts_case)


def run_hybrid_search_fts_case_all_quadrants(
    ctx: dict[str, Any],
    case: FtsQueryCase | str,
) -> None:
    """Run one FTS case on every FTS-loaded quadrant (``c1_x`` + ``c2_y``)."""
    run_hybrid_search_fts_case_on_quadrants(ctx, case, MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS)


def run_hybrid_search_fts_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: FtsQueryCase,
) -> dict[str, Any]:
    """Run hybrid search fts case."""
    query: dict[str, Any] = {
        "where_document": case.where_document,
        "n_results": case.n_results,
    }
    if case.where is not None:
        query["where"] = case.where
    result = namespace.hybrid_search(
        query=query,
        n_results=case.n_results,
        include=["documents", "metadatas"],
    )
    assert_hybrid_fulltext_result(
        corpus,
        result,
        case.where_document,
        case.n_results,
        where=case.where,
        check_ranking=case.check_ranking,
    )
    return result


def assert_not_contains_no_token_leak(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    forbidden_token: str,
) -> None:
    """Assert not contains no token leak."""
    result_ids = set(result["ids"][0])
    forbidden_ids = {rec.doc_id for rec in corpus if forbidden_token.lower() in rec.document.lower()}
    leaked = result_ids & forbidden_ids
    assert not leaked, f"$not_contains leaked forbidden ids: {sorted(leaked)[:10]}"
