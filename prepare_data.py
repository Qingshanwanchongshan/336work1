"""Download TinyStories, train a byte-level BPE tokenizer, and encode the data.

This is the data-preparation pipeline that feeds the training loop in
``train.py``. By default it uses the TinyStories *validation* set (22K
documents, tens of MB) so the whole pipeline runs in a couple of minutes on
CPU -- exactly what you want for a quick "observe the effect" run. Pass
``--split train`` to use the full 2.12M-document training set instead.

Outputs (under ``--out_dir``):
    vocab.json, merges.txt   -- serialized tokenizer
    train.npy, val.npy       -- token-id sequences (uint16)
"""

import argparse
import json
import os
import time
import urllib.request

import numpy as np

from cs336_basics.tokenizer import Tokenizer, bytes_to_unicode, train_bpe

TINYSTORIES = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="data", help="where raw .txt files are stored")
    p.add_argument("--out_dir", type=str, default="data", help="where vocab/merges/npy are written")
    p.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--val_fraction", type=float, default=0.1, help="fraction held out as val.npy")
    return p.parse_args()


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already exists")
        return
    print(f"downloading {url}")
    print(f"      -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("      done")


def save_tokenizer(vocab: dict[int, bytes], merges, vocab_path: str, merges_path: str) -> None:
    """Serialize vocab + merges in the GPT-2 bytes-to-unicode format."""
    encoder = bytes_to_unicode()
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(k): "".join(encoder[b] for b in v) for k, v in vocab.items()}, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write("".join(encoder[x] for x in a) + " " + "".join(encoder[x] for x in b) + "\n")


def encode_corpus(tokenizer: Tokenizer, txt_path: str) -> np.ndarray:
    """Encode a TinyStories file (one document per line) into a uint16 id array."""
    ids: list[int] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Append the end-of-sequence token to delimit this document.
            ids.extend(tokenizer.encode(line + "<|endoftext|>"))
    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < 65536, "vocab larger than uint16 can represent"
    return arr


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    txt_path = os.path.join(args.data_dir, f"tinystories_{args.split}.txt")
    download(TINYSTORIES[args.split], txt_path)

    # --- Train the tokenizer on the (downloaded) corpus ---
    t0 = time.time()
    print(f"training BPE tokenizer (vocab_size={args.vocab_size})...")
    vocab, merges = train_bpe(txt_path, args.vocab_size, special_tokens=["<|endoftext|>"])
    print(f"tokenizer trained in {time.time() - t0:.1f}s, vocab size {len(vocab)}")

    vocab_path = os.path.join(args.out_dir, "vocab.json")
    merges_path = os.path.join(args.out_dir, "merges.txt")
    save_tokenizer(vocab, merges, vocab_path, merges_path)
    print(f"saved {vocab_path} and {merges_path}")

    # --- Encode the corpus into token ids and split train / val ---
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    t0 = time.time()
    print(f"encoding {txt_path}...")
    ids = encode_corpus(tokenizer, txt_path)
    print(f"encoded {len(ids)} tokens in {time.time() - t0:.1f}s")

    n_val = int(len(ids) * args.val_fraction)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    train_path = os.path.join(args.out_dir, "train.npy")
    val_path = os.path.join(args.out_dir, "val.npy")
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    print(f"saved {train_path} ({len(train_ids)} tokens) and {val_path} ({len(val_ids)} tokens)")
    print("done. now run:  python train.py --data data/train.npy --val_data data/val.npy")


if __name__ == "__main__":
    main()
