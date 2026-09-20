#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]   # only so the selftest can import qualeval.normalize
# ///
"""
mkretrieval.py -- the Phase 7 `retrieval` slice: real questions a web search returns
pages for, with integer answers so the existing grader needs no new code.

  uv run tools/mkretrieval.py --out results/p7-retrieval-items.jsonl
  uv run tools/mkretrieval.py --selftest
"""
import argparse, json, re, sys

ANSWER_INT = ("End your reply with the final answer on its own line, "
              "in exactly this form:\nANSWER: <integer>")

# (query, answer). The query is what goes to Brave verbatim -- it carries no answer-format
# instruction, which would otherwise be 90 chars of noise in every search.
# Answers are stable facts, not anything that moves with the calendar or a record book.
QA = [
    ("in what year did the Berlin Wall fall", "1989"),
    ("in what year did Apollo 11 land on the Moon", "1969"),
    ("in what year did the Titanic sink", "1912"),
    ("in what year was the Eiffel Tower completed", "1889"),
    ("in what year was the first iPhone released", "2007"),
    ("in what year did the Chernobyl disaster happen", "1986"),
    ("in what year was the Soviet Union dissolved", "1991"),
    ("in what year did the Second World War end", "1945"),
    ("in what year was the Magna Carta sealed", "1215"),
    ("in what year was the United States Declaration of Independence adopted", "1776"),
    ("in what year did the Suez Canal open", "1869"),
    ("in what year did the Panama Canal open", "1914"),
    ("in what year did the Wright brothers make their first powered flight", "1903"),
    ("in what year did Alexander Fleming discover penicillin", "1928"),
    ("in what year were the first Nobel Prizes awarded", "1901"),
    ("in what year was the Hubble Space Telescope launched", "1990"),
    ("in what year did euro banknotes and coins enter circulation", "2002"),
    ("in what year was Mount Everest first summited", "1953"),
    ("in what year did the Golden Gate Bridge open", "1937"),
    ("in what year was the Great Fire of London", "1666"),
    ("in what year were the first modern Olympic Games held in Athens", "1896"),
    ("in what year was the Hoover Dam completed", "1936"),
    ("in what year was the transistor invented at Bell Labs", "1947"),
    ("in what year was Sputnik 1 launched", "1957"),
    ("in what year was the ENIAC computer unveiled to the public", "1946"),
    ("in what year did the Channel Tunnel open", "1994"),
    ("in what year was the first Star Wars film released", "1977"),
    ("in what year did the Sydney Opera House open", "1973"),
    ("in what year was the Rosetta Stone discovered", "1799"),
    ("in what year was the Statue of Liberty dedicated", "1886"),
    ("in what year was the first Harry Potter book published in the United Kingdom", "1997"),
    ("in what year was Wikipedia launched", "2001"),
    ("in what year was YouTube founded", "2005"),
    ("in what year was Google founded", "1998"),
    ("in what year did Expedition 1 become the first crew of the International Space Station", "2000"),
    ("in what year did the Large Hadron Collider first circulate beams", "2008"),
    ("in what year was the Human Genome Project declared complete", "2003"),
    ("in what year was Dolly the sheep born", "1996"),
    ("in what year did Ray Tomlinson send the first network email", "1971"),
    ("in what year did Tim Berners-Lee write his proposal for the World Wide Web at CERN", "1989"),
    ("how tall is the Burj Khalifa in metres", "828"),
    ("how many floors does the Empire State Building have", "102"),
    ("how many bones are in the adult human body", "206"),
    ("how many chromosomes are in a human body cell", "46"),
    ("how many elements are on the periodic table", "118"),
    ("what is the atomic number of gold", "79"),
    ("what is the atomic number of carbon", "6"),
    ("what is the atomic number of uranium", "92"),
    ("what is the atomic number of oxygen", "8"),
    ("how many keys are on a standard piano", "88"),
    ("how many squares are on a chessboard", "64"),
    ("how many players from one team are on the field in association football", "11"),
    ("how many states are in the United States of America", "50"),
    ("how many member states does the United Nations have", "193"),
    ("how many countries are in the European Union", "27"),
    ("how many moons does Mars have", "2"),
    ("how many planets are in the solar system", "8"),
    ("what is the speed of light in a vacuum in metres per second", "299792458"),
    ("how many lines are in a sonnet", "14"),
    ("how many symphonies did Beethoven complete", "9"),
    ("how many Apollo missions landed humans on the Moon", "6"),
    ("how many people have walked on the Moon", "12"),
    ("how many Great Lakes are in North America", "5"),
    ("how many amendments are in the United States Bill of Rights", "10"),
    ("how many Grand Slam tournaments are played in tennis each year", "4"),
    ("how many rings are in the Olympic symbol", "5"),
    ("how many teeth does a healthy adult human have including wisdom teeth", "32"),
]


def build():
    """One item per question. `query` is the search string, `prompt` is what the model sees."""
    out = []
    for i, (q, a) in enumerate(QA):
        question = q[0].upper() + q[1:] + "?"
        out.append({"id": f"retrieval-{i:04d}", "slice": "retrieval",
                    "query": q, "answer": a,
                    "prompt": f"{question}\n\n{ANSWER_INT}"})
    return out


def selftest() -> int:
    """Offline. Every property the run depends on, including the ones that fail quietly."""
    fails, items = [], build()
    if len(items) < 60:
        fails.append(f"slice is {len(items)} items, want at least 60")
    if len({i["id"] for i in items}) != len(items):
        fails.append("duplicate ids")
    if len({i["query"] for i in items}) != len(items):
        fails.append("duplicate queries -- the prefix cache would make arms incomparable")
    for it in items:
        if not re.fullmatch(r"-?\d+", it["answer"]):
            fails.append(f"{it['id']} answer {it['answer']!r} is not an integer")
        # The query is sent to Brave verbatim; an answer-format instruction in it is noise.
        if "ANSWER" in it["query"] or len(it["query"]) > 120:
            fails.append(f"{it['id']} query unfit for a search engine: {it['query']!r}")
        if it["query"].lower() not in it["prompt"].lower():
            fails.append(f"{it['id']} prompt does not contain its own query")
        if not it["prompt"].endswith("ANSWER: <integer>"):
            fails.append(f"{it['id']} prompt lost the answer instruction")
    # The grader is qualeval's, so prove the gold answers survive ITS normalizer rather
    # than assuming: a gold that normalizes to None grades every arm 0% and looks real.
    sys.path.insert(0, "tools")
    try:
        from qualeval import normalize
        for it in items:
            if normalize(it["answer"], "retrieval") != it["answer"]:
                fails.append(f"{it['id']} gold {it['answer']!r} does not survive normalize()")
    except ImportError:
        fails.append("could not import qualeval.normalize -- gold answers unverified")
    for f in fails[:10]:
        print(f"  FAIL {f}")
    print(f"selftest: {len(items)} items, {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/p7-retrieval-items.jsonl")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    items = build()
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items to {a.out}")
