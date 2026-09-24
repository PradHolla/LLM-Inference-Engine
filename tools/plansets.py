#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""Convert MTRAG, QReCC and FreshQA into planeval label files, and merge label files.

  uv run tools/plansets.py mtrag --n 200
  uv run tools/plansets.py merge a.jsonl b.jsonl --out data/plansets/all.jsonl
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "plansets"
MTRAG_RAW = "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/mtrag-human"
MTRAG_DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
QRECC_ZIP = "https://raw.githubusercontent.com/apple/ml-qrecc/main/dataset/qrecc_data.zip"
# The README links a Google Sheet that is re-issued weekly; this is "FreshQA April 21, 2026".
FRESHQA_SHEET = "1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM"
FRESHQA_CATEGORIES = ("never-changing", "slow-changing", "fast-changing", "false-premise")

STOPWORDS = frozenset("""
a an the and or but nor so yet of in on at to for from by with about as into onto over under
than then that this these those there here what which who whom whose when where why how
is are was were be been being am do does did done have has had having can could will would
shall should may might must not no yes its it's his her hers their theirs our ours your yours
my mine they them he she we you i me him us all any some each every both either neither
one ones also just very more most much many such own same other another only too again
tell me please give know like get got let make made say said way thing things
""".split())


def content_words(text: str) -> list[str]:
    """Lowercase alphanumeric runs, length >= 3, not stopwords, first occurrence order."""
    seen: dict[str, None] = {}
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) >= 3 and word not in STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


def must_contain(rewrite: str, question: str) -> list[str]:
    """Content words of the gold rewrite absent from the original question."""
    asked = set(content_words(question))
    return [word for word in content_words(rewrite) if word not in asked]


def fetch(url: str, dest: Path) -> Path:
    """Download once into the cache; an existing file is reused, never re-fetched."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.write_bytes(response.content)
    partial.rename(dest)
    return dest


def spread(groups: dict[str, list], n: int, rng: random.Random) -> list:
    """Round-robin over shuffled groups, one random member each pass, until n are taken."""
    pools = {key: rng.sample(members, len(members)) for key, members in sorted(groups.items())}
    order = rng.sample(sorted(pools), len(pools))
    picked: list = []
    while len(picked) < n and any(pools[key] for key in order):
        for key in order:
            if pools[key] and len(picked) < n:
                picked.append(pools[key].pop())
    return picked


def write_jsonl(items: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def strip_speaker(text: str) -> str:
    return re.sub(r"^\|user\|:\s*", "", text.strip())


def convert_mtrag(tasks: list[dict], rewrites: dict[str, str], n: int,
                  rng: random.Random) -> list[dict]:
    """One item per turn of the human generation tasks; conversational turns are search false."""
    by_conv: dict[str, list[dict]] = {}
    for task in tasks:
        by_conv.setdefault(task["conversation_id"], []).append(task)
    picked = spread(by_conv, n, rng)
    chosen = {task["task_id"] for task in picked}
    convs = {task["conversation_id"] for task in picked}
    extra = [task for task in tasks if task["conversation_id"] in convs
             and task["task_id"] not in chosen and conversational(task)]
    droppable = [task for task in reversed(picked) if not conversational(task)]
    drop = {task["task_id"] for task in droppable[:len(extra)]}
    kept = [task for task in picked if task["task_id"] not in drop] + extra
    return [mtrag_item(task, rewrites.get(task["task_id"])) for task in kept]


def conversational(task: dict) -> bool:
    return "CONVERSATIONAL" in task["Answerability"]


def mtrag_item(task: dict, rewrite: str | None) -> dict:
    *earlier, last = task["input"]
    roles = {"user": "user", "agent": "assistant"}
    multi = task["Multi-Turn"][0] if task["Multi-Turn"] else "N/A"
    category = task["Answerability"][0].lower() + ("" if multi == "N/A" else f"/{multi.lower()}")
    search = not conversational(task)
    gold = strip_speaker(rewrite) if rewrite else None
    words = must_contain(gold, last["text"]) if gold and search else []
    return {"id": f"mtrag-{task['conversation_id'][:10]}-t{task['turn']}", "source": "mtrag",
            "category": category,
            "history": [{"role": roles[m["speaker"]], "content": m["text"]} for m in earlier],
            "user": last["text"], "search": search, "think": None, "must_contain": words,
            "hit_fraction": 0.5, "rewrite": gold}


def load_mtrag(cache: Path) -> tuple[list[dict], dict[str, str]]:
    ref = fetch(f"{MTRAG_RAW}/generation_tasks/reference.jsonl", cache / "mtrag" / "reference.jsonl")
    tasks = [json.loads(line) for line in open(ref, encoding="utf-8") if line.strip()]
    rewrites: dict[str, str] = {}
    for domain in MTRAG_DOMAINS:
        path = fetch(f"{MTRAG_RAW}/retrieval_tasks/{domain}/{domain}_rewrite.jsonl",
                     cache / "mtrag" / f"{domain}_rewrite.jsonl")
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                rewrites[row["_id"]] = row["text"]
    return tasks, rewrites


def convert_qrecc(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Turns after the first, history rebuilt from the ORIGINAL earlier questions and answers."""
    turns = {(r["Conversation_no"], r["Turn_no"]): r for r in records}
    groups: dict[str, list[dict]] = {}
    for record in records:
        conv, turn = record["Conversation_no"], record["Turn_no"]
        if turn > 1 and all((conv, t) in turns for t in range(1, turn)):
            groups.setdefault(str(conv), []).append(record)
    items = []
    for record in spread(groups, n, rng):
        conv, turn = record["Conversation_no"], record["Turn_no"]
        history = []
        for t in range(1, turn):
            earlier = turns[(conv, t)]
            history.append({"role": "user", "content": earlier["Question"]})
            if earlier["Answer"].strip():
                history.append({"role": "assistant", "content": earlier["Answer"]})
        items.append({"id": f"qrecc-{conv}-t{turn}", "source": "qrecc",
                      "category": record["Conversation_source"], "history": history,
                      "user": record["Question"], "search": True, "think": None,
                      "must_contain": must_contain(record["Rewrite"], record["Question"]),
                      "hit_fraction": 0.5, "rewrite": record["Rewrite"]})
    return items


def load_qrecc(cache: Path) -> list[dict]:
    archive = fetch(QRECC_ZIP, cache / "qrecc" / "qrecc_data.zip")
    with zipfile.ZipFile(archive) as zf:
        return json.loads(zf.read("qrecc_test.json"))


def freshqa_rows(text: str) -> list[dict]:
    """Parse the sheet export: a warning banner precedes the header row whose first cell is id."""
    rows = list(csv.reader(io.StringIO(text)))
    start = next(i for i, row in enumerate(rows) if row and row[0] == "id")
    header = rows[start]
    return [dict(zip(header, row)) for row in rows[start + 1:] if row and row[0].strip()]


def freshqa_category(row: dict) -> str:
    return "false-premise" if row["false_premise"].strip().upper() == "TRUE" else row["fact_type"]


def freshqa_answers(row: dict) -> list[str]:
    return [row[f"answer_{i}"].strip() for i in range(10) if row.get(f"answer_{i}", "").strip()]


def convert_freshqa(rows: list[dict], n: int, rng: random.Random, split: str = "TEST") -> list[dict]:
    """Stratified n/4 per category from one split; search labelled only where facts change."""
    per = n // len(FRESHQA_CATEGORIES)
    items = []
    for category in FRESHQA_CATEGORIES:
        pool = [row for row in rows if row["split"] == split and freshqa_category(row) == category]
        for row in sorted(rng.sample(pool, min(per, len(pool))), key=lambda r: int(r["id"])):
            items.append({"id": f"freshqa-{row['id']}", "source": "freshqa", "category": category,
                          "history": [], "user": row["question"].strip(),
                          "search": True if category in ("fast-changing", "slow-changing") else None,
                          "think": None, "must_contain": [], "answer": freshqa_answers(row)})
    return items


def convert_gsm8k(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Maths word problems as real-data search:false negatives; MTRAG alone has only 10."""
    picked = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    return [{"id": f"gsm8k-nosearch-{i}", "source": "gsm8k", "category": "maths",
             "history": [], "user": rows[i]["question"].strip(), "search": False,
             "think": None, "must_contain": []} for i in picked]


def load_freshqa(cache: Path, sheet: str = FRESHQA_SHEET) -> list[dict]:
    url = f"https://docs.google.com/spreadsheets/d/{sheet}/export?format=csv"
    path = fetch(url, cache / "freshqa" / f"freshqa-{sheet[:12]}.csv")
    return freshqa_rows(path.read_text(encoding="utf-8"))


def merge(paths: list[str]) -> list[dict]:
    """Concatenate label files; source defaults to the file stem; duplicate ids are an error."""
    items, seen = [], set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("source", Path(path).stem)
            if item["id"] in seen:
                raise SystemExit(f"duplicate id {item['id']!r} in {path}")
            seen.add(item["id"])
            items.append(item)
    return items


def summarize(items: list[dict]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.get("category") or "-"] = counts.get(item.get("category") or "-", 0) + 1
    search = {str(v): sum(item["search"] is v for item in items) for v in (True, False, None)}
    scored = sum(bool(item["must_contain"]) for item in items)
    return (f"items {len(items)}; categories {dict(sorted(counts.items()))}; search labels "
            f"{search}; must_contain non-empty {scored}")


def selftest() -> int:
    fails: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    check("must_contain keeps only rewrite words the question lacks",
          must_contain("What are the educational requirements to become a physician's assistant?",
                       "What are the educational requirements required to become one?")
          == ["physician", "assistant"])
    check("must_contain drops stopwords and short words",
          must_contain("Tell me more about Tesla the car company.", "Tell me more about it")
          == ["tesla", "car", "company"])
    check("must_contain is empty for a self-contained question",
          must_contain("Who won the 2010 World Cup?", "who won the 2010 world cup") == [])

    def msg(speaker: str, text: str) -> dict:
        return {"speaker": speaker, "text": text,
                "metadata": {"author_type": "human", "author_id": "x", "created_at": 1}}

    def task(conv: str, turn: int, ans: str, multi: str, texts: list[str]) -> dict:
        turns = [msg("user" if i % 2 == 0 else "agent", t) for i, t in enumerate(texts)]
        return {"conversation_id": conv, "task_id": f"{conv}<::>{turn}", "task_type": "rag",
                "turn": str(turn), "dataset": "MT-RAG Authors (Internal)", "input": turns,
                "targets": [], "Question Type": ["Factoid"], "Multi-Turn": [multi],
                "Answerability": [ans], "Collection": "mt-rag-clapnq-elser-512-100-20240503"}

    tasks = [task("c1", 1, "ANSWERABLE", "N/A", ["where do the arizona cardinals play"]),
             task("c1", 2, "ANSWERABLE", "Follow-up",
                  ["where do the arizona cardinals play", "In Glendale.", "how many titles do they have?"]),
             task("c1", 3, "CONVERSATIONAL", "Follow-up",
                  ["where do the arizona cardinals play", "In Glendale.",
                   "how many titles do they have?", "Two.", "Thank you!"])]
    rewrites = {"c1<::>1": "|user|: Where do the Arizona Cardinals play?",
                "c1<::>2": "|user|: How many titles do the Arizona Cardinals have?"}
    runs = [convert_mtrag(tasks, rewrites, 2, random.Random(seed)) for seed in range(12)]
    check("mtrag keeps the conversational turn in a sampled conversation, at size n",
          all(len(run) == 2 and any(i["id"].endswith("-t3") for i in run) for run in runs))
    by_turn = {item["id"][-2:]: item for item in runs[0]}
    check("mtrag conversational turn is search false, no must_contain",
          by_turn["t3"]["search"] is False and by_turn["t3"]["must_contain"] == [])
    check("mtrag category joins answerability and multi-turn",
          by_turn["t3"]["category"] == "conversational/follow-up")
    full = {i["id"][-2:]: i for i in convert_mtrag(tasks, rewrites, 3, random.Random(0))}
    check("mtrag follow-up history, roles and rewrite words",
          full["t2"]["history"] == [{"role": "user", "content": "where do the arizona cardinals play"},
                                    {"role": "assistant", "content": "In Glendale."}]
          and full["t2"]["must_contain"] == ["arizona", "cardinals"]
          and full["t2"]["think"] is None and full["t1"]["category"] == "answerable")

    def rec(conv: int, turn: int, q: str, rw: str, a: str, ctx: list[str]) -> dict:
        return {"Context": ctx, "Question": q, "Rewrite": rw, "Answer": a, "Answer_URL": "",
                "Conversation_no": conv, "Turn_no": turn, "Conversation_source": "quac"}

    records = [rec(100, 1, "what happened in 2010?", "what happened in 2010 to Kevin Durant?",
                   "He signed an extension.", []),
               rec(100, 2, "how did people react?", "how did people react to Durant's extension?",
                   "", ["what happened in 2010 to Kevin Durant?", "He signed an extension."]),
               rec(100, 3, "how did he do with the thunder?", "how did Kevin Durant do with the thunder?",
                   "They won 55 games.", ["what happened in 2010 to Kevin Durant?",
                                          "He signed an extension.",
                                          "how did people react to Durant's extension?"])]
    q = {item["id"]: item for item in convert_qrecc(records, 5, random.Random(0))}
    check("qrecc skips the first turn", set(q) == {"qrecc-100-t2", "qrecc-100-t3"})
    check("qrecc history uses original questions and skips empty answers",
          [m["content"] for m in q["qrecc-100-t3"]["history"]]
          == ["what happened in 2010?", "He signed an extension.", "how did people react?"])
    check("qrecc must_contain and labels",
          q["qrecc-100-t3"]["must_contain"] == ["kevin", "durant"] and q["qrecc-100-t3"]["search"]
          and q["qrecc-100-t3"]["think"] is None and q["qrecc-100-t3"]["category"] == "quac")

    sheet = ("\"Warning: do not click thumbs up/down\",,,,,,,,,,\n,,,,,,,,,,\n"
             "id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,"
             "source,answer_0,answer_1,note\n"
             "0,TEST,What year did the first human land on Mars?,before 2022,occasionally,TRUE,"
             "one-hop,slow-changing,https://x,No humans have landed on Mars yet.,,\n"
             "1,TEST,What was the first animal to orbit the Earth?,before 2022,N/A,FALSE,one-hop,"
             "never-changing,https://x,Laika,,\n"
             "2,TEST,Who is the CEO of Twitter?,2023,weekly,FALSE,one-hop,fast-changing,https://x,"
             "Linda Yaccarino,Elon Musk,\n"
             "3,TEST,How many moons does Jupiter have?,2023,yearly,FALSE,one-hop,slow-changing,"
             "https://x,95,,\n"
             "4,DEV,Who wrote Hamlet?,before 2022,N/A,FALSE,one-hop,never-changing,https://x,"
             "Shakespeare,,\n")
    fq = {item["id"]: item for item in convert_freshqa(freshqa_rows(sheet), 4, random.Random(0))}
    check("freshqa takes TEST only, one per category at n=4",
          set(fq) == {"freshqa-0", "freshqa-1", "freshqa-2", "freshqa-3"})
    check("freshqa false_premise column overrides fact_type",
          fq["freshqa-0"]["category"] == "false-premise" and fq["freshqa-0"]["search"] is None)
    check("freshqa search labels by category",
          fq["freshqa-1"]["search"] is None and fq["freshqa-2"]["search"] is True
          and fq["freshqa-3"]["search"] is True)
    check("freshqa keeps every gold answer",
          fq["freshqa-2"]["answer"] == ["Linda Yaccarino", "Elon Musk"])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
        write_jsonl([{"id": "x1", "source": "mtrag"}], a)
        write_jsonl([{"id": "y1"}], b)
        merged = merge([str(a), str(b)])
        check("merge preserves source and fills a missing one from the file stem",
              [m["source"] for m in merged] == ["mtrag", "b"])
        try:
            merge([str(a), str(a)])
            check("merge rejects duplicate ids", False)
        except SystemExit:
            pass
    check("spread takes from every group before repeating one",
          sorted(k for k, _ in spread({"a": [("a", 1), ("a", 2)], "b": [("b", 1)], "c": [("c", 1)]},
                                      3, random.Random(1))) == ["a", "b", "c"])
    gsm = convert_gsm8k([{"question": f" q{i} "} for i in range(5)], 3, random.Random(0))
    check("gsm8k items are search:false negatives with unique ids",
          len(gsm) == 3 and all(i["search"] is False and i["think"] is None for i in gsm)
          and len({i["id"] for i in gsm}) == 3 and gsm[0]["user"].startswith("q"))

    print("\n".join(fails) if fails else "plansets selftest: PASS")
    return 1 if fails else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, n, aliases in (("mtrag", 200, []), ("qrecc", 150, ["qreCC"]), ("freshqa", 160, []),
                             ("gsm8k", 60, [])):
        p = sub.add_parser(name, aliases=aliases)
        p.add_argument("--n", type=int, default=n)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cache", default=str(CACHE))
        p.add_argument("--out", default=str(CACHE / f"{name}.jsonl"))
        p.set_defaults(cmd=name)
        if name == "freshqa":
            p.add_argument("--sheet", default=FRESHQA_SHEET, help="Google Sheet id from the README")
    m = sub.add_parser("merge")
    m.add_argument("files", nargs="+")
    m.add_argument("--out", default=str(CACHE / "all.jsonl"))
    args = parser.parse_args()
    if args.cmd == "merge":
        items = merge(args.files)
    else:
        rng, cache = random.Random(args.seed), Path(args.cache)
        if args.cmd == "mtrag":
            items = convert_mtrag(*load_mtrag(cache), args.n, rng)
        elif args.cmd == "qrecc":
            items = convert_qrecc(load_qrecc(cache), args.n, rng)
        elif args.cmd == "gsm8k":
            gsm = Path(__file__).resolve().parent.parent / "data" / "gsm8k-test.jsonl"
            items = convert_gsm8k([json.loads(l) for l in gsm.open() if l.strip()], args.n, rng)
        else:
            items = convert_freshqa(load_freshqa(cache, args.sheet), args.n, rng)
    write_jsonl(items, Path(args.out))
    print(f"{args.out}: {summarize(items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
