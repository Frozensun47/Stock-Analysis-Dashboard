"""Sync the local SQLite databases to S3-compatible object storage (R2 / S3 / B2).

Everything under cache/ is derived data that must NOT live in git — it is either
too large or rebuildable. Object storage is the right home once the minute-bar
archive and the news corpus start growing past a few hundred MB.

Cloudflare R2 is the cheapest fit here: S3-compatible API, no egress fees, and a
10 GB free tier — so the same boto3 code below works for R2, AWS S3 and Backblaze
B2 by changing only the endpoint. (Firestore is a document store with per-write
billing; for append-only bar/news blobs it is both slower and more expensive.)

Configure in .env:
    S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com   # no bucket in the path
    S3_BUCKET=<bucket name>
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...

Usage
    python storage.py push            # upload every cache/*.sqlite (+ .pkl)
    python storage.py pull            # download them onto a fresh machine
    python storage.py list
Requires: pip install boto3
"""
import os, sys, glob
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
load_dotenv(os.path.join(HERE, ".env"))
BUCKET = os.getenv("S3_BUCKET", "stockdash")
PREFIX = os.getenv("S3_PREFIX", "stockdash/")

def client():
    import boto3
    kw = {}
    if os.getenv("S3_ENDPOINT_URL"):
        kw["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
        # R2 has no regions but boto3 insists on one; "auto" is what R2 expects.
        kw["region_name"] = os.getenv("S3_REGION", "auto")
    elif os.getenv("S3_REGION"):
        kw["region_name"] = os.environ["S3_REGION"]
    return boto3.client("s3", **kw)

# upstox_15m_partial.sqlite is the fetcher's resume scratch file — it duplicates
# upstox_15m.pkl at 1.5x the size and is worthless on any other machine.
SKIP = ("upstox_15m_partial.sqlite",)

def _files(patterns=("*.sqlite", "*.pkl", "*.joblib")):
    """The .joblib is the trained model — the deployed app needs it to serve picks."""
    out = []
    for p in patterns:
        out += sorted(glob.glob(os.path.join(CACHE, p)))
    return [f for f in out if os.path.basename(f) not in SKIP]

def push():
    """Checkpoint WAL first so the uploaded .sqlite is self-contained."""
    import sqlite3
    s3, n = client(), 0
    for f in _files():
        if f.endswith(".sqlite"):
            con = sqlite3.connect(f); con.execute("PRAGMA wal_checkpoint(TRUNCATE)"); con.close()
        key = PREFIX + os.path.basename(f)
        s3.upload_file(f, BUCKET, key)
        print(f"  ↑ {key}  ({os.path.getsize(f)/1e6:.1f} MB)")
        n += 1
    print(f"pushed {n} objects to s3://{BUCKET}/{PREFIX}")

def pull():
    os.makedirs(CACHE, exist_ok=True)
    s3, n = client(), 0
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX).get("Contents", []):
        dest = os.path.join(CACHE, os.path.basename(o["Key"]))
        s3.download_file(BUCKET, o["Key"], dest)
        print(f"  ↓ {o['Key']}  ({o['Size']/1e6:.1f} MB)"); n += 1
    print(f"pulled {n} objects")

def ls():
    for o in client().list_objects_v2(Bucket=BUCKET, Prefix=PREFIX).get("Contents", []):
        print(f"  {o['Key']:50s} {o['Size']/1e6:8.1f} MB  {o['LastModified']:%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("No S3/R2 credentials in .env — see the docstring above."); sys.exit(1)
    {"push": push, "pull": pull, "list": ls}.get(cmd, lambda: print(__doc__))()
