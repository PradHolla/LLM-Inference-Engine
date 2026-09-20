#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
artifact-strip.py -- turn a rendered artifact (as Artifact `read` saves it) into the author
source, and tell you whether the live page has moved since we last published.

  uv run tools/artifact-strip.py --rendered <saved.html> --out docs/artifact/engine.html
  uv run tools/artifact-strip.py --check docs/artifact/engine.html
"""
import argparse, hashlib, pathlib, sys

MARKERS = ("__FRAME_PREAMBLE", "frame-runtime")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def strip(rendered: str) -> str:
    """Author content only: everything the publish-time skeleton wraps, and nothing it adds."""
    i = rendered.index("<body>") + len("<body>")
    body = rendered[i:rendered.rindex("</body>")].strip()
    if not body.startswith("<title>"):
        raise SystemExit(f"author source should open with <title>, got {body[:60]!r}")
    for m in MARKERS:
        if m in body:
            raise SystemExit(f"host runtime leaked into the source: {m}")
    return body + "\n"


def main(a) -> int:
    if a.check:
        src = pathlib.Path(a.check)
        stamp = src.with_suffix(".sha256")
        if not stamp.exists():
            print(f"no {stamp.name}; cannot tell whether the live page has moved")
            return 1
        want, got = stamp.read_text().split()[0], digest(src.read_text())
        print(f"local  {got}\nstamp  {want}")
        # Equal means the local file is what we last published -- it does NOT prove the live
        # page still matches, only that WE have not edited since. Re-read live before editing.
        print("local file matches the last publish" if want == got
              else "LOCAL FILE HAS UNPUBLISHED EDITS")
        return 0 if want == got else 2

    body = strip(pathlib.Path(a.rendered).read_text())
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = out.read_text() if out.exists() else None
    out.write_text(body)
    out.with_suffix(".sha256").write_text(digest(body) + f"  {out.name}\n")
    print(f"wrote {out} ({len(body)} bytes)")
    if prior is not None and prior != body:
        print("  NOTE: this differs from the committed copy -- diff it before assuming "
              "the change is yours; someone may have republished the live page")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rendered", help="the file Artifact `read` saved")
    p.add_argument("--out", default="docs/artifact/engine.html")
    p.add_argument("--check", help="report whether this source matches its .sha256 stamp")
    a = p.parse_args()
    if not a.rendered and not a.check:
        p.error("need --rendered or --check")
    raise SystemExit(main(a))
