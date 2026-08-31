#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
curve.py -- collapse bench.py sweep records into latency-vs-throughput curve points.
Recomputes from raw per-request JSONL rather than trusting bench.py's summary; throughput
uses the observed span, not the nominal window (incident 10). See NOTES/code-notes.md.

  uv run tools/curve.py > curve.json
"""
import json, sys, glob, os
def pct(xs,p):
    if not xs: return None
    s=sorted(xs); k=(len(s)-1)*p/100; lo=int(k); hi=min(lo+1,len(s)-1)
    return s[lo]+(s[hi]-s[lo])*(k-lo)

def curve(path):
    by={}
    for line in open(path):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        if d.get("warmup") or d.get("status")!="ok": continue
        by.setdefault(d["rate"],[]).append(d)
    out=[]
    for rate in sorted(by):
        rs=by[rate]
        if len(rs)<5: continue
        t0=min(r["t_sent"] for r in rs)
        t1=max(r["t_sent"]+r["e2e"] for r in rs)
        span=max(t1-t0,1e-9)
        itl=[i for r in rs for i in (r.get("itls") or [])]
        out.append({"rate":rate,"n":len(rs),"rps":len(rs)/span,
                    "ttft_p50":pct([r["ttft"] for r in rs],50)*1000,
                    "ttft_p95":pct([r["ttft"] for r in rs],95)*1000,
                    "itl_p50":(pct(itl,50) or 0)*1000,
                    "itl_p95":(pct(itl,95) or 0)*1000})
    return out

SERIES=[
 ("Phase 1 baseline","results/phase1-sweep.jsonl"),
 ("Our engine","results/phase2-step3b-sweep.jsonl"),
 ("vLLM, unique prompts","results/phase3-vllm-sweep-unique.jsonl"),
 ("vLLM, cached prompts","results/phase3-vllm-sweep-cached.jsonl"),
 ("vLLM, cached (high rates)","results/phase3-vllm-sweep-cached-hi.jsonl"),
 ("no chunked prefill","results/phase3-ablate-B-nochunked-unique.jsonl"),
 ("no prefix caching","results/phase3-ablate-A-nocache-unique.jsonl"),
 ("fp8 KV cache","results/phase3-ablate-D-fp8kv-unique.jsonl"),
 ("max-num-seqs 16","results/phase3-ablate-G-seqs16.jsonl"),
 ("max-num-seqs 32","results/phase3-ablate-H-seqs32.jsonl"),
 ("max-num-seqs 64","results/phase3-ablate-I-seqs64.jsonl"),
]
data={}
for name,p in SERIES:
    if os.path.exists(p):
        c=curve(p)
        if c: data[name]=c
    else: print(f"MISSING {p}", file=sys.stderr)
print(json.dumps(data,indent=1))
