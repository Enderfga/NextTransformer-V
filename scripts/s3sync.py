"""Mirror a local directory to an S3 prefix or back, with multipart transfers.

Credentials come from the standard AWS environment; nothing is printed about them.
Objects whose size already matches are skipped, so a re-run resumes an interrupted sync.

  python scripts/s3sync.py up   <local_dir> s3://bucket/prefix
  python scripts/s3sync.py down s3://bucket/prefix <local_dir>
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

CONFIG = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8)
WORKERS = 24


def split(uri):
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix.rstrip("/")


def remote_sizes(s3, bucket, prefix):
    sizes = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for item in page.get("Contents", []):
            sizes[item["Key"][len(prefix) + 1:]] = item["Size"]
    return sizes


def main(direction, source, target):
    s3 = boto3.client("s3", config=Config(max_pool_connections=WORKERS * 8, retries={"max_attempts": 10}))
    if direction == "up":
        root, (bucket, prefix) = Path(source), split(target)
        done = remote_sizes(s3, bucket, prefix)
        jobs = [p for p in sorted(root.rglob("*")) if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
                and done.get(p.relative_to(root).as_posix()) != p.stat().st_size]
        run = lambda p: s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(root).as_posix()}", Config=CONFIG)
    else:
        (bucket, prefix), root = split(source), Path(target)
        sizes = remote_sizes(s3, bucket, prefix)
        jobs = [k for k, n in sorted(sizes.items()) if not ((root / k).exists() and (root / k).stat().st_size == n)]
        def run(key):
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            s3.download_file(bucket, f"{prefix}/{key}", str(partial), Config=CONFIG)
            partial.rename(path)
    print(f"{direction}: {len(jobs)} files to transfer", flush=True)
    failures = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # report and continue; a re-run retries the rest
                failures += 1
                print(f"failed: {futures[future]}: {type(error).__name__}: {error}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} transferred, {failures} failed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])
