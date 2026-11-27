"""
search.py -- Brave query, parallel page fetch, text extraction. Each stage timed
separately, because the phase headline is where the user's wait actually goes.

  python -m gateway.search --selftest
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import httpx

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
N_RESULTS = int(os.environ.get("GW_SEARCH_N", "5"))  # still one Brave call; more chances of a fetchable page
FETCH_TIMEOUT_S = float(os.environ.get("GW_FETCH_TIMEOUT_S", "2.0"))
CHARS_PER_SOURCE = int(os.environ.get("GW_CHARS_PER_SOURCE", "6000"))
CHARS_PER_TOKEN = float(os.environ.get("GW_CHARS_PER_TOKEN", "4.0"))
# Most sites reject an unrecognised client outright, so the default UA measured
# rejections rather than page fetches: 2 of 3 sources 403d from the box.
USER_AGENT = os.environ.get("GW_USER_AGENT",
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_DROP = {"script", "style", "noscript", "template", "svg", "head",
         "nav", "header", "footer", "aside", "form", "button", "select"}
_MAIN = {"main", "article"}
# Below this many chars, a <main> is a stub and the whole page is the better source.
MAIN_MIN_CHARS = int(os.environ.get("GW_MAIN_MIN_CHARS", "500"))


@dataclass
class Source:
    """One retrieved page. `ok` false means it is excluded from the prompt, not fatal."""
    url: str
    title: str = ""
    text: str = ""
    chars: int = 0
    tokens_est: int = 0
    ok: bool = False
    error: str | None = None
    snippet: str = ""  # Brave's own excerpts of the page, the fallback when the page cannot be fetched
    kind: str = "page"  # "page": text fetched from the site; "summary": Brave's excerpts only


# A short page saying one of these is a bot wall or an outage notice, not content (autotrader, 2026-11-27)
BLOCK_MAX_CHARS = int(os.environ.get("GW_BLOCK_MAX_CHARS", "800"))
BLOCK_RE = re.compile(r"access denied|site is currently unavailable|are you a (?:human|robot)|verify you are (?:a )?human|"
                      r"enable javascript|unusual traffic|request (?:was )?blocked|captcha|incident number|"
                      r"pardon our interruption|request unsuccessful", re.I)
SNIPPET_CHARS = int(os.environ.get("GW_SNIPPET_CHARS", "1500"))


def blocked(text: str) -> bool:
    """A short page that is a bot wall or an outage notice rather than the content asked for."""
    return len(text) <= BLOCK_MAX_CHARS and bool(BLOCK_RE.search(text))


def _plain(fragment: str) -> str:
    """Brave's snippet HTML (<strong>, entities) as plain text."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment or ""))).strip()


@dataclass
class SearchOutcome:
    """Everything the gateway needs: the sources, and the three spans they cost."""
    query: str
    sources: list[Source] = field(default_factory=list)
    search_ms: float | None = None
    fetch_ms: float | None = None
    extract_ms: float | None = None
    n_sources: int = 0
    error: str | None = None


class _Text(HTMLParser):
    """Collects visible text twice: everything, and only what sits inside main/article."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main: list[str] = []
        self.skip = 0
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self.skip += 1
        elif tag in _MAIN:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in _DROP and self.skip:
            self.skip -= 1
        elif tag in _MAIN and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.skip:
            return
        d = data.strip()
        if not d:
            return
        self.parts.append(d)
        if self.depth:
            self.main.append(d)


def extract(html: str, cap_chars: int = CHARS_PER_SOURCE) -> str:
    """HTML to plain text, whitespace collapsed, truncated at the character cap."""
    p = _Text()
    try:
        p.feed(html)
    except Exception:
        pass
    # The cap is binding on nearly every real page, so what fills it decides what the
    # model sees. Prefer the article body; falling back to the whole page spends the
    # window on navigation, measured on 2 of 3 sources before this existed.
    chosen = p.main if len(" ".join(p.main)) > MAIN_MIN_CHARS else p.parts
    text = re.sub(r"\s+", " ", " ".join(chosen)).strip()
    return text[:cap_chars]


def load_key() -> str | None:
    """Env first, then .env beside the repo, then the box's own file. Never logged."""
    if os.environ.get("BRAVE_API_KEY"):
        return os.environ["BRAVE_API_KEY"]
    for p in (Path(__file__).resolve().parent.parent / ".env", Path("/opt/llm/.brave-key")):
        try:
            body = p.read_text()
        except OSError:
            continue
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("BRAVE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
            if p.name == ".brave-key" and line and not line.startswith("#"):
                return line
    return None


async def brave(query: str, client: httpx.AsyncClient, n: int = N_RESULTS,
                key: str | None = None) -> tuple[list[Source], float, str | None]:
    """Query Brave. Returns candidate sources with URLs only, plus elapsed ms."""
    key = key or load_key()
    if not key:
        return [], 0.0, "no BRAVE_API_KEY"
    t0 = time.perf_counter()
    try:
        r = await client.get(BRAVE_URL, params={"q": query, "count": n, "extra_snippets": "true"},
                             headers={"Accept": "application/json",
                                      "X-Subscription-Token": key}, timeout=10.0)
        r.raise_for_status()
        hits = (r.json().get("web") or {}).get("results") or []
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1e3, f"{type(e).__name__}: {e}"
    out = [Source(url=h.get("url", ""), title=_plain(h.get("title", "")), snippet=_snippet(h))
           for h in hits[:n] if h.get("url")]
    return out, (time.perf_counter() - t0) * 1e3, None


def _snippet(hit: dict) -> str:
    """The hit's description and extra snippets, deduplicated, as one capped plain-text excerpt."""
    parts = []
    for raw in [hit.get("description", ""), *(hit.get("extra_snippets") or [])]:
        text = _plain(raw)
        if text and text not in parts:
            parts.append(text)
    return " ... ".join(parts)[:SNIPPET_CHARS]


async def _fetch_one(src: Source, client: httpx.AsyncClient, timeout: float) -> None:
    """Fetch one page in place. A failure marks the source, it never raises."""
    try:
        r = await client.get(src.url, timeout=timeout,
                             follow_redirects=True,
                             headers={"User-Agent": USER_AGENT,
                                      "Accept": "text/html,application/xhtml+xml",
                                      "Accept-Language": "en-US,en;q=0.9"})
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "").lower():
            src.error = "not html"
            return
        src.text = r.text
        src.ok = True
    except httpx.HTTPStatusError as e:
        src.error = f"HTTP {e.response.status_code}"
    except Exception as e:
        src.error = f"{type(e).__name__}"


async def fetch_all(sources: list[Source], client: httpx.AsyncClient,
                    timeout: float = FETCH_TIMEOUT_S) -> float:
    """Fetch every source concurrently. p95 is set by the slowest site, so cap it."""
    t0 = time.perf_counter()
    await asyncio.gather(*(_fetch_one(s, client, timeout) for s in sources))
    return (time.perf_counter() - t0) * 1e3


async def run_search(query: str, client: httpx.AsyncClient, n: int = N_RESULTS,
                     cap_chars: int = CHARS_PER_SOURCE) -> SearchOutcome:
    """The whole pipeline with its three spans. Degrades to fewer sources, never raises."""
    out = SearchOutcome(query=query)
    sources, out.search_ms, out.error = await brave(query, client, n)
    out.sources = sources
    if not sources:
        out.fetch_ms = out.extract_ms = 0.0
        return out
    out.fetch_ms = await fetch_all(sources, client)
    t0 = time.perf_counter()
    for s in sources:
        if s.ok:
            s.text = extract(s.text, cap_chars)
            s.ok = len(s.text) > 0
            if s.ok and blocked(s.text):
                s.ok, s.error = False, "blocked page"
        if not s.ok and s.snippet:  # the site refused us; Brave's excerpts are what we know of it
            s.text, s.ok, s.kind = s.snippet, True, "summary"
        if not s.ok:
            s.text = ""
        s.chars = len(s.text)
        s.tokens_est = int(s.chars / CHARS_PER_TOKEN)
    out.extract_ms = (time.perf_counter() - t0) * 1e3
    out.n_sources = sum(1 for s in sources if s.ok)
    return out


def render_block(out: SearchOutcome) -> str:
    """The text injected into the prompt. Empty when nothing was retrieved."""
    good = [s for s in out.sources if s.ok]
    if not good:
        return ""
    parts = [f"[{i}] {s.title}\n{s.url}\n" + (SUMMARY_NOTE if s.kind == "summary" else "") + s.text
             for i, s in enumerate(good, 1)]
    return "Search results:\n\n" + "\n\n".join(parts)


SUMMARY_NOTE = "(This page could not be opened; below is only the search engine's excerpt of it.)\n"


def selftest() -> int:
    """Offline. Proves extraction, the caps, and that failures degrade rather than raise."""
    fails = []

    html = ("<html><head><title>t</title><style>.a{color:red}</style></head><body>"
            "<script>var x = 'NOTPROSE';</script><h1>Heading</h1>"
            "<p>First   paragraph.</p><p>Second paragraph.</p></body></html>")
    got = extract(html)
    if "NOTPROSE" in got or "color:red" in got:
        fails.append(f"extract leaked script/style: {got!r}")
    if got != "Heading First paragraph. Second paragraph.":
        fails.append(f"extract text wrong: {got!r}")

    if len(extract("<p>" + "x" * 50_000 + "</p>", cap_chars=100)) != 100:
        fails.append("cap_chars not enforced")

    if extract("<p>unclosed <b>bold</p>") == "":
        fails.append("malformed html produced nothing")

    body = "Real article content. " * 40
    nav = "<nav>Home Docs API Community</nav><header>Skip to content</header>"
    if "Home Docs API" in extract(f"<html><body>{nav}<main><p>{body}</p></main></body></html>"):
        fails.append("navigation survived when an article body was present")
    stub = extract(f"<html><body>{nav}<main>tiny</main><p>{body}</p></body></html>")
    if "Real article content" not in stub:
        fails.append("a stub <main> should fall back to the whole page")

    empty = SearchOutcome(query="q")
    if render_block(empty) != "":
        fails.append("render_block should be empty with no sources")

    ok = SearchOutcome(query="q", sources=[Source(url="u", title="T", text="body", ok=True),
                                           Source(url="v", error="Timeout")])
    block = render_block(ok)
    if "[1] T" not in block or "v" in block.split("\n")[0:1]:
        fails.append(f"render_block wrong: {block!r}")
    if "[2]" in block:
        fails.append("render_block included a failed source")

    wall = "We're sorry for any inconvenience, but the site is currently unavailable. Incident Number: 18.c6"
    if not blocked(wall) or blocked("Access denied. " + "Real article content. " * 60) or blocked("Real article."):
        fails.append("blocked(): a short outage page must count, a long page or a plain short page must not")
    hit = {"description": "Search from <strong>133</strong> Used Ford Mustang cars &amp; trucks",
           "extra_snippets": ["This 2014 Ford Mustang V6 Premium Coupe, 61,000 miles.",
                              "Search from 133 Used Ford Mustang cars & trucks"]}
    if _snippet(hit) != ("Search from 133 Used Ford Mustang cars & trucks ... "
                         "This 2014 Ford Mustang V6 Premium Coupe, 61,000 miles."):
        fails.append(f"_snippet: tags and entities stripped, repeats dropped: {_snippet(hit)!r}")
    mixed = SearchOutcome(query="q", sources=[
        Source(url="a", title="A", text="page text", ok=True),
        Source(url="b", title="B", text="the excerpt", ok=True, kind="summary"),
        Source(url="c", error="HTTP 403")])
    block = render_block(mixed)
    if ("[2] B\nb\n" + SUMMARY_NOTE + "the excerpt") not in block or SUMMARY_NOTE in block.split("[2]")[0] \
            or "[3]" in block:
        fails.append(f"render_block: a summary source is labelled, a page is not, a source with nothing is left out")

    async def _fallback():
        async def fake_brave(query, client, n=N_RESULTS, key=None):
            return [Source(url="https://x/blocked", title="X", snippet="2017 Mustang GT, $18,900."),
                    Source(url="https://y/wall", title="Y", snippet="Y's excerpt."),
                    Source(url="https://z/gone", title="Z")], 1.0, None

        async def fake_fetch(sources, client, timeout=FETCH_TIMEOUT_S):
            sources[0].error = "HTTP 403"
            sources[1].text, sources[1].ok = f"<html><body><p>{wall}</p></body></html>", True
            sources[2].error = "ConnectTimeout"
            return 1.0
        global brave, fetch_all
        saved = brave, fetch_all
        brave, fetch_all = fake_brave, fake_fetch
        try:
            async with httpx.AsyncClient() as c:
                return await run_search("q", c)
        finally:
            brave, fetch_all = saved
    fb = asyncio.run(_fallback())
    got = [(s.ok, s.kind, s.error, s.text) for s in fb.sources]
    if got != [(True, "summary", "HTTP 403", "2017 Mustang GT, $18,900."),
               (True, "summary", "blocked page", "Y's excerpt."), (False, "page", "ConnectTimeout", "")] \
            or fb.n_sources != 2:
        fails.append(f"run_search fallback: 403 and wall pages fall back to their excerpts, no excerpt drops: {got}")

    async def _degrade():
        async with httpx.AsyncClient() as c:
            r = await run_search("q", c, n=1, cap_chars=10)
            return r
    res = asyncio.run(_degrade())
    if res.error is None and res.n_sources == 0 and res.search_ms is None:
        fails.append("run_search returned no spans and no error")
    if res.n_sources == 0 and res.fetch_ms is None:
        fails.append("spans must be set even when nothing is retrieved")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "what is paged attention"

    async def _main():
        async with httpx.AsyncClient() as c:
            out = await run_search(q, c)
        print(f"query        {out.query!r}")
        print(f"search_ms    {out.search_ms:.1f}" if out.search_ms is not None else "search_ms    -")
        print(f"fetch_ms     {out.fetch_ms:.1f}" if out.fetch_ms is not None else "fetch_ms     -")
        print(f"extract_ms   {out.extract_ms:.1f}" if out.extract_ms is not None else "extract_ms   -")
        print(f"n_sources    {out.n_sources}   error {out.error}")
        for s in out.sources:
            print(f"  {'ok ' if s.ok else 'DROP'} {s.chars:>6} chars  ~{s.tokens_est:>5} tok  "
                  f"{(s.error or ''):<18} {s.url[:70]}")
    asyncio.run(_main())
