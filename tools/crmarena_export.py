# /// script
# dependencies = ["simple-salesforce"]
# ///
"""Snapshot a CRMArena-Pro Salesforce org into SQLite (read-only SOQL, one table per object).
Usage: uv run tools/crmarena_export.py --env data/crmarena/org.env --prefix SALESFORCE_B2B_ --schema data/crmarena/b2b_schema.json --out data/crmarena/b2b.sqlite"""
import argparse, json, sqlite3, time
from datetime import datetime, timezone

from simple_salesforce import Salesforce

SQL_TYPE = {"int": "INTEGER", "boolean": "INTEGER", "double": "REAL", "currency": "REAL", "percent": "REAL"}
SKIP_TYPES = {"address", "location", "base64"}  # compound or binary; components of compounds are kept


def login(env_path: str, prefix: str) -> Salesforce:
    env = dict(l.strip().split("=", 1) for l in open(env_path) if "=" in l and not l.startswith("#"))
    return Salesforce(username=env[f"{prefix}USERNAME"], password=env[f"{prefix}PASSWORD"],
                      security_token=env[f"{prefix}SECURITY_TOKEN"])


def cell(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def export_object(sf: Salesforce, db: sqlite3.Connection, name: str) -> dict:
    fields = [f for f in getattr(sf, name).describe()["fields"] if f["type"] not in SKIP_TYPES]
    cols = [f["name"] for f in fields]
    db.execute(f'DROP TABLE IF EXISTS "{name}"')
    db.execute(f'CREATE TABLE "{name}" (' + ", ".join(
        f'"{f["name"]}" {SQL_TYPE.get(f["type"], "TEXT")}' + (" PRIMARY KEY" if f["name"] == "Id" else "")
        for f in fields) + ")")
    soql = f"SELECT {', '.join(cols)} FROM {name}"
    try:
        records = sf.query_all(soql)["records"]
    except Exception as e:  # article versions may demand a PublishStatus filter
        if not name.endswith("__kav"):
            raise
        records = sf.query_all(soql + " WHERE PublishStatus = 'Online'")["records"]
        print(f"  {name}: retried with PublishStatus filter after {type(e).__name__}")
    db.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(cols))})',
                   [[cell(r.get(c)) for c in cols] for r in records])
    return {"rows": len(records), "columns": len(cols),
            "skipped": [f["name"] for f in getattr(sf, name).describe()["fields"] if f["type"] in SKIP_TYPES]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sf = login(a.env, a.prefix)
    db = sqlite3.connect(a.out)
    manifest = {"instance": sf.sf_instance, "started": datetime.now(timezone.utc).isoformat(), "objects": {}}
    for obj in json.load(open(a.schema)):
        t = time.time()
        manifest["objects"][obj["object"]] = info = export_object(sf, db, obj["object"])
        db.commit()
        print(f"{obj['object']:28s} {info['rows']:>7,} rows {info['columns']:>4} cols  {time.time() - t:.1f}s", flush=True)
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest["api_remaining"] = sf.limits()["DailyApiRequests"]["Remaining"]
    db.execute("DROP TABLE IF EXISTS _manifest")
    db.execute("CREATE TABLE _manifest (json TEXT)")
    db.execute("INSERT INTO _manifest VALUES (?)", [json.dumps(manifest)])
    db.commit()
    print(f"total {sum(o['rows'] for o in manifest['objects'].values()):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
