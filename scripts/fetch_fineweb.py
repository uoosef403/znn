#!/usr/bin/env python3
# fetch_fineweb.py — one-time data prep (Python only; the framework itself is Zig).
#
# Streams FineWeb-Edu examples from HuggingFace and writes their `text` field,
# newline-separated, into a single raw-bytes file (data/text.bin). The Zig
# pipeline then tokenizes + packs + batches from that file.
#
# Streaming means we never hold the dataset in memory and stop at a byte
# budget, so this works fine on a 6GB machine. For "real deal" scale, raise
# --bytes or run this on RunPod (the file is just bytes; the Zig side doesn't
# care where it came from).
#
#   deps:  pip install datasets
#   run:   python scripts/fetch_fineweb.py --bytes 100_000_000 --out data/text.bin
#
# If HF is unreachable, pass --fallback to emit a small bundled prose corpus so
# the downstream pipeline can still be smoke-tested end-to-end offline.

import argparse
import os
import sys

BUILTIN = """\
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be, or not to be, that is the question.
In the beginning the universe was created. This made a lot of people angry.
The only thing we have to fear is fear itself.
Ask not what your country can do for you; ask what you can do for your country.
Knowledge is power. Information is liberating. Education is the premise of progress.
The function of education is to teach one to think intensively and to think critically.
Science is the poetry of reality. We are made of star-stuff.
The universe is under no obligation to make sense to you.
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a FineWeb-Edu slice to data/text.bin")
    ap.add_argument("--out", default="data/text.bin")
    ap.add_argument("--bytes", type=int, default=100_000_000, help="target byte budget")
    ap.add_argument("--config", default="sample-10BT",
                    help="HF config (sample-10BT / sample-100BT / a crawl id)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--fallback", action="store_true",
                    help="emit the bundled prose corpus instead of fetching (offline smoke test)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.fallback:
        # repeat the small corpus to roughly the byte budget
        with open(args.out, "wb") as f:
            total = 0
            while total < args.bytes:
                f.write(BUILTIN.encode("utf-8")); f.write(b"\n")
                total += len(BUILTIN) + 1
        print(f"[fallback] wrote {total} bytes to {args.out}")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        print("       (or use --fallback for an offline smoke test)", file=sys.stderr)
        return 1

    print(f"streaming HuggingFaceFW/fineweb-edu config={args.config} "
          f"split={args.split} budget={args.bytes} bytes ...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", args.config,
                      split=args.split, streaming=True)
    n = 0
    total = 0
    with open(args.out, "wb") as f:
        for ex in ds:
            text = ex["text"]
            b = text.encode("utf-8")
            f.write(b); f.write(b"\n")
            total += len(b) + 1
            n += 1
            if n % 1000 == 0:
                print(f"  {n} docs, {total/1e6:.1f} MB", flush=True)
            if total >= args.bytes:
                break
    print(f"done: {n} docs, {total} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())