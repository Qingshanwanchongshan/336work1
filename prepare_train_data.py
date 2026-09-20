"""Download the full TinyStories training set and encode it to token ids.

Uses the already-trained tokenizer (data/vocab.json + data/merges.txt), so this
only needs to download the raw text and encode it. Encoding is streamed through
an ``array('H')`` (a C uint16 array) rather than a Python list, so ~530M tokens
only cost ~1GB of RAM instead of ~15GB.

Output: data/train.npy  (the full training set, ~530M uint16 token ids)
"""

import argparse
import os
import time
from array import array

import numpy as np

from cs336_basics.tokenizer import Tokenizer

TRAIN_URL = "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest) / 1e9:.2f} GB)")
        return
    import subprocess

    print(f"downloading {url}")
    subprocess.run(
        ["curl", "-L", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", dest, url],
        check=True,
    )
    print(f"downloaded -> {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")


def encode_to_npy(tokenizer: Tokenizer, txt_path: str, npy_path: str) -> None:
    """Stream-encode `txt_path` (one doc per line) into a uint16 .npy file."""
    ids = array("H")  # uint16 C array: ~2 bytes/token instead of ~28 for a Python int
    n_lines = 0
    t0 = time.time()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ids.extend(tokenizer.encode(line + "<|endoftext|>"))
            n_lines += 1
            if n_lines % 200000 == 0:
                dt = time.time() - t0
                print(f"  ... {n_lines:>8} docs, {len(ids):>10} tokens, {dt:.0f}s elapsed")

    arr = np.array(ids, dtype=np.uint16)
    np.save(npy_path, arr)
    print(f"encoded {n_lines} docs -> {len(ids)} tokens in {time.time() - t0:.0f}s")
    print(f"saved -> {npy_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab", type=str, default="data/vocab.json")
    parser.add_argument("--merges", type=str, default="data/merges.txt")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    txt_path = os.path.join(args.data_dir, "tinystories_train.txt")
    npy_path = os.path.join(args.data_dir, "train.npy")

    download(TRAIN_URL, txt_path)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    print(f"loaded tokenizer (vocab size {len(tokenizer.vocab)})")

    encode_to_npy(tokenizer, txt_path, npy_path)
    print("done. train with:  python train.py --data data/train.npy --val_data data/val.npy ...")


if __name__ == "__main__":
    main()
