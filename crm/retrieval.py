"""Phase 11 hybrid article search for the service agent: BM25 (FTS5) + bge-small passage embeddings, fused by RRF.

  uv run python -m crm.retrieval build --org all
  uv run python -m crm.retrieval search --org b2b "extended technical support offer"
  uv run python -m crm.retrieval --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:  # numpy and fastembed are optional: without them knowledge search is BM25 only, never a crash
    import numpy as np
except ImportError:
    np = None

from crm.actions import _parse_rows
from crm.env import DATA, CRMEnv, Observation, TaskSession

log = logging.getLogger("crm.retrieval")

MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_VERSION = 1  # bump when the passage splitting changes, so old caches are rebuilt
CHUNK_MIN, CHUNK_MAX, CHUNK_PARAS = 600, 1200, 3
TAIL_MIN, TAIL_MAX = 300, 1500
RRF_K = 60
VEC_TOP = 30  # articles taken from the vector ranking into the fusion
SNIPPET_CHARS = 300
ABBREV = {"inc", "corp", "ltd", "co", "llc", "vs", "etc", "e.g", "i.e", "dr", "mr", "mrs", "ms", "st", "no", "u.s"}

_model = None
_model_failed = False
_model_lock = threading.Lock()
_embed_lock = threading.Lock()
_indexes: dict[str, "Index | None"] = {}
_index_lock = threading.Lock()


@dataclass
class Index:
    """All passage vectors of one org (rows aligned with ids, pidx, texts)."""
    vecs: np.ndarray
    ids: list[str]
    pidx: list[int]
    texts: list[str]


@dataclass
class Hit:
    """One fused article: its best passage and where each ranking put it (None = not in that list)."""
    Id: str
    field: str
    snippet: str
    bm25_rank: int | None
    vec_rank: int | None
    cosine: float | None
    score: float


# ---------------------------------------------------------------- passages

def _sentences(para: str) -> list[str]:
    """Split on sentence ends, re-joining after abbreviations such as 'Inc.' or 'e.g.'."""
    out: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", para.strip()):
        last = out[-1].rsplit(None, 1)[-1].rstrip(".").lower() if out else ""
        if out and out[-1].endswith(".") and last in ABBREV:
            out[-1] += " " + piece
        else:
            out.append(piece)
    return out


def split_passages(text: str) -> list[str]:
    """Pack paragraphs into passages of about 600-1200 chars (at most 3 paragraphs), cutting only at sentence ends."""
    units: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text or "")):
        if not para:
            continue
        if len(para) <= CHUNK_MAX:
            units.append(para)
            continue
        cur = ""
        for s in _sentences(para):
            if cur and len(cur) + 1 + len(s) > CHUNK_MAX:
                units.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            units.append(cur)
    passages: list[list[str]] = []
    for u in units:
        if passages and len("\n\n".join(passages[-1] + [u])) <= CHUNK_MAX and len(passages[-1]) < CHUNK_PARAS \
                and len("\n\n".join(passages[-1])) < CHUNK_MIN:
            passages[-1].append(u)
        else:
            passages.append([u])
    out = ["\n\n".join(p) for p in passages]
    if len(out) > 1 and len(out[-1]) < TAIL_MIN and len(out[-2]) + 2 + len(out[-1]) <= TAIL_MAX:
        out[-2:] = [out[-2] + "\n\n" + out[-1]]
    return out


def article_passages(art: dict) -> list[tuple[int, str]]:
    """(passage index, text): index 0 is the Summary, 1.. are the FAQ answer passages."""
    out = [(0, (art["Summary"] or "").strip())] if (art["Summary"] or "").strip() else []
    out += [(i, p) for i, p in enumerate(split_passages(art["FAQ_Answer__c"] or ""), 1)]
    return out


# ---------------------------------------------------------------- cache

def cache_path(org: str) -> Path:
    return DATA / "derived" / f"kb_vectors_{org}.npz"


def load_articles(org: str) -> list[dict]:
    """Every knowledge article of the org's derived database, ordered by Id."""
    conn = sqlite3.connect(f"file:{DATA / 'derived' / f'{org}.db'}?mode=ro", uri=True)
    try:
        cur = conn.execute("SELECT Id, Title, Summary, FAQ_Answer__c FROM Knowledge__kav ORDER BY Id")
        return [dict(zip(("Id", "Title", "Summary", "FAQ_Answer__c"), r)) for r in cur.fetchall()]
    finally:
        conn.close()


def fingerprint(articles: list[dict]) -> tuple[int, str]:
    """(article count, sha256 over model, chunking version and every Id, Title, Summary and FAQ text)."""
    h = hashlib.sha256(f"{MODEL}|{CHUNK_VERSION}".encode())
    for a in articles:
        for k in ("Id", "Title", "Summary", "FAQ_Answer__c"):
            h.update(b"\x00" + str(a[k] or "").encode())
    return len(articles), h.hexdigest()


def _get_model():
    """The shared fastembed model, loaded once per process; None (logged once) if it cannot load."""
    global _model, _model_failed
    with _model_lock:
        if _model is None and not _model_failed:
            try:
                if np is None:
                    raise ImportError("numpy is not installed")
                from fastembed import TextEmbedding
                cache = os.environ.get("FASTEMBED_CACHE_PATH") or str(DATA / "fastembed-cache")
                _model = TextEmbedding(model_name=MODEL, cache_dir=cache)
            except Exception as e:  # noqa: BLE001 - any failure means BM25-only search
                _model_failed = True
                log.warning("embedding model unavailable (%s: %s); knowledge search falls back to BM25 only",
                            type(e).__name__, e)
        return _model


def _embed(texts: list[str], query: bool = False) -> np.ndarray:
    """Normalized float32 vectors for the texts; raises if the model is unavailable."""
    model = _get_model()
    if model is None:
        raise RuntimeError("embedding model unavailable")
    with _embed_lock:
        gen = model.query_embed(texts) if query else model.embed(texts, batch_size=32)
        v = np.asarray(list(gen), dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def build_index(articles: list[dict], path: Path) -> Index:
    """Embed every passage (each prefixed with its article Title) and write the cache atomically."""
    ids, pidx, texts, embed_texts = [], [], [], []
    for a in articles:
        for i, p in article_passages(a):
            ids.append(a["Id"])
            pidx.append(i)
            texts.append(p)
            embed_texts.append(f"{a['Title']}\n{p}")
    vecs = _embed(embed_texts)
    count, digest = fingerprint(articles)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(fd)
    np.savez(tmp, vecs=vecs, ids=np.array(ids), pidx=np.array(pidx, dtype=np.int32), texts=np.array(texts),
             fp_count=np.array(count), fp_hash=np.array(digest), model=np.array(MODEL))
    os.replace(tmp, path)
    return Index(vecs, ids, pidx, texts)


def load_cache(articles: list[dict], path: Path) -> Index | None:
    """The cached index if the file exists and its fingerprint matches the articles, else None."""
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        count, digest = fingerprint(articles)
        if int(z["fp_count"]) != count or str(z["fp_hash"]) != digest or str(z["model"]) != MODEL:
            return None
        return Index(z["vecs"].astype(np.float32), [str(x) for x in z["ids"]], [int(x) for x in z["pidx"]],
                     [str(x) for x in z["texts"]])
    except Exception:  # noqa: BLE001 - a corrupt cache is just rebuilt
        return None


def get_index(org: str) -> Index | None:
    """The org's index, loaded or rebuilt once per process; None (logged once) if anything fails."""
    with _index_lock:
        if org not in _indexes:
            try:
                arts = load_articles(org)
                _indexes[org] = load_cache(arts, cache_path(org)) or build_index(arts, cache_path(org))
            except Exception as e:  # noqa: BLE001
                log.warning("article index for %s unavailable (%s: %s); BM25 only", org, type(e).__name__, e)
                _indexes[org] = None
        return _indexes[org]


# ---------------------------------------------------------------- search

def bm25_articles(session: TaskSession, query: str, limit: int) -> tuple[Observation, dict[str, dict]]:
    """FTS5 search over Knowledge__kav: the raw Observation and {Id: best row} in rank order."""
    obs = session.search(str(query or ""), ["Knowledge__kav"], limit=limit)
    best: dict[str, dict] = {}
    if obs.ok:
        for r in _parse_rows(obs.text):
            best.setdefault(r["Id"], r)
    return obs, best


def vector_articles(org: str, query: str) -> list[tuple[str, float, str]] | None:
    """[(article Id, cosine, best passage)] best first, one entry per article; None if embeddings are unavailable."""
    idx = get_index(org)
    if idx is None or _get_model() is None:
        return None
    try:
        sims = idx.vecs @ _embed([query], query=True)[0]
    except Exception as e:  # noqa: BLE001
        log.warning("query embedding failed (%s: %s); BM25 only", type(e).__name__, e)
        return None
    best: dict[str, int] = {}
    for i in np.argsort(-sims):
        best.setdefault(idx.ids[i], int(i))
    return [(a, float(sims[i]), idx.texts[i]) for a, i in best.items()]


def _cut(text: str, n: int = SNIPPET_CHARS) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."


def hybrid_rank(org: str, query: str, bm25: dict[str, dict], n: int) -> list[Hit] | None:
    """Fuse the vector and BM25 article rankings with RRF; None if the vector side is unavailable."""
    vec = vector_articles(org, query)
    if vec is None:
        return None
    vec = vec[:VEC_TOP]
    vrank = {a: (i + 1, c, t) for i, (a, c, t) in enumerate(vec)}
    brank = {a: i + 1 for i, a in enumerate(bm25)}
    hits = []
    for a in dict.fromkeys([*bm25, *vrank]):
        score = sum(1 / (RRF_K + r) for r in (brank.get(a), vrank[a][0] if a in vrank else None) if r)
        if a in vrank:
            field, snippet = "passage", _cut(vrank[a][2])
        else:
            field, snippet = bm25[a]["field"], bm25[a]["snippet"]
        hits.append(Hit(a, field, snippet, brank.get(a), vrank[a][0] if a in vrank else None,
                        vrank[a][1] if a in vrank else None, score))
    hits.sort(key=lambda h: (-h.score, min(h.bm25_rank or 10**6, h.vec_rank or 10**6), h.Id))
    return hits[:n]


# ---------------------------------------------------------------- cli and selftest

KNOWN = {699: "ka0Wt000000EoD3IAK", 717: "ka0Wt000000EoUpIAK", 732: "ka0Wt000000EoUpIAK",
         737: "ka0Wt000000EpYxIAK", 762: "ka0Wt000000EozTIAS", 765: "ka0Wt000000EoTBIA0",
         767: "ka0Wt000000EoGIIA0", 768: "ka0Wt000000EpqfIAC", 773: "ka0Wt000000EpNfIAK"}


def _cmd_build(orgs: list[str], force: bool) -> int:
    for org in orgs:
        t0 = time.monotonic()
        arts = load_articles(org)
        idx = None if force else load_cache(arts, cache_path(org))
        state = "cache already up to date" if idx else "built"
        idx = idx or build_index(arts, cache_path(org))
        _indexes[org] = idx
        print(f"{org}: {len(arts)} articles, {len(idx.ids)} passages, {state} in {time.monotonic() - t0:.1f}s "
              f"-> {cache_path(org)}")
    return 0


def _cmd_search(org: str, query: str, n: int) -> int:
    s = CRMEnv(org).open_task()
    obs, best = bm25_articles(s, query, min(n * 8, s.row_cap))
    if not obs.ok:
        print(obs.text)
        return 1
    hits = hybrid_rank(org, query, best, n)
    if hits is None:
        print("embeddings unavailable; BM25 order:")
        hits = [Hit(a, r["field"], r["snippet"], i, None, None, 0.0) for i, (a, r) in enumerate(best.items(), 1)][:n]
    titles = dict(s.conn.execute("SELECT Id, Title FROM Knowledge__kav").fetchall())
    for h in hits:
        cos = f"{h.cosine:.3f}" if h.cosine is not None else "  -  "
        print(f"{h.score:.4f}  bm25 #{h.bm25_rank or '-':<3} vec #{h.vec_rank or '-':<3} cos {cos}  {h.Id}  "
              f"{titles.get(h.Id)!r}\n        [{h.field}] {h.snippet}")
    s.close()
    return 0


def _selftest() -> int:
    from crm.tasks import load

    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    # --- passage splitting on every b2b and b2c article
    for org in ("b2b", "b2c"):
        arts = load_articles(org)
        lens, bad_end, lost = [], 0, 0
        for a in arts:
            ps = split_passages(a["FAQ_Answer__c"])
            lens += [len(p) for p in ps]
            bad_end += sum(not re.search(r"[.!?:\"')]$", p.strip()) for p in ps)
            lost += " ".join(a["FAQ_Answer__c"].split()) != " ".join(" ".join(ps).split())
        check(f"{org}: passages cover the whole text, end at sentence ends, <= {TAIL_MAX} chars",
              lost == 0 and bad_end == 0 and max(lens) <= TAIL_MAX,
              f"{len(arts)} articles, {len(lens)} passages, len min/median/max "
              f"{min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}, bad ends {bad_end}, lost text {lost}")
    long_para = "Acme Inc. makes chips. " * 200
    ps = split_passages(long_para)
    check("long paragraph split only at sentence ends, never after 'Inc.'",
          len(ps) > 1 and all(p.endswith("chips.") and len(p) <= CHUNK_MAX for p in ps), f"{len(ps)} passages")

    # --- fingerprint and stale cache
    arts = load_articles("b2b")
    f0 = fingerprint(arts)
    changed = [dict(a) for a in arts]
    changed[0]["FAQ_Answer__c"] += " x"
    check("fingerprint changes when one article text changes", fingerprint(changed) != f0 and fingerprint(arts) == f0)
    small = arts[:6]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "kb_vectors_test.npz"
        idx = build_index(small, p)
        ok = load_cache(small, p)
        check("cache round trip: same ids, vectors normalized float32",
              ok is not None and ok.ids == idx.ids and ok.vecs.dtype == np.float32
              and np.allclose(np.linalg.norm(ok.vecs, axis=1), 1.0, atol=1e-5), f"{len(idx.ids)} passages")
        mod = [dict(a) for a in small]
        mod[1]["Summary"] = (mod[1]["Summary"] or "") + " changed"
        check("stale cache (an article changed) is rejected", load_cache(mod, p) is None
              and load_cache(small[:5], p) is None)

    # --- the 9 known dev questions
    tasks = {t.idx: t for t in load("b2b")}
    env = CRMEnv("b2b")
    s = env.open_task()
    print(f"{'idx':>4} {'right article':<20} {'BM25 rank':>9} {'hybrid rank':>11}  question")
    b_top3 = h_top3 = 0
    b_ranks, h_ranks = {}, {}
    for idx_, want in KNOWN.items():
        q = tasks[idx_].query
        obs, best = bm25_articles(s, q, min(5 * 8, s.row_cap))
        order = list(best)[:5]
        hits = hybrid_rank("b2b", q, best, 5)
        if hits is None:
            check("hybrid available", False, "embeddings unavailable")
            return 1
        horder = [h.Id for h in hits]
        br = order.index(want) + 1 if want in order else None
        hr = horder.index(want) + 1 if want in horder else None
        b_ranks[idx_], h_ranks[idx_] = br, hr
        b_top3 += bool(br and br <= 3)
        h_top3 += bool(hr and hr <= 3)
        print(f"{idx_:>4} {want:<20} {br or '-':>9} {hr or '-':>11}  {q[:70]}")
    print(f"right article in top 3: BM25 {b_top3}/9, hybrid {h_top3}/9")
    check("hybrid is not worse than BM25 at top 3", h_top3 >= b_top3, f"{h_top3} >= {b_top3}")

    # --- fallback when the model cannot load
    global _model, _model_failed
    saved = (_model, _model_failed, dict(_indexes))
    _model, _model_failed = None, True
    _indexes.clear()
    try:
        check("model unavailable -> hybrid_rank returns None (caller uses BM25)",
              hybrid_rank("b2b", "billing", {}, 3) is None)
    finally:
        _model, _model_failed = saved[0], saved[1]
        _indexes.update(saved[2])
    s.close()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="hybrid knowledge-article retrieval")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="embed all passages and write the caches")
    b.add_argument("--org", choices=["b2b", "b2c", "all"], required=True)
    b.add_argument("--force", action="store_true", help="re-embed even if the cache is up to date")
    q = sub.add_parser("search", help="print the ranked articles with both ranks and the fused score")
    q.add_argument("--org", choices=["b2b", "b2c"], required=True)
    q.add_argument("--n", type=int, default=8)
    q.add_argument("query")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cmd == "build":
        return _cmd_build(["b2b", "b2c"] if a.org == "all" else [a.org], a.force)
    if a.cmd == "search":
        return _cmd_search(a.org, a.query, a.n)
    ap.print_help()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
