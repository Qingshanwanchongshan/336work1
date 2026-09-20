"""Retrain the BPE tokenizer and serialize it in the (fixed) GPT-2 byte format."""

import json
import time

from cs336_basics.tokenizer import Tokenizer, bytes_to_unicode, train_bpe

t0 = time.time()
vocab, merges = train_bpe("data/tinystories_valid.txt", 10000, special_tokens=["<|endoftext|>"])
print(f"trained in {time.time() - t0:.1f}s: vocab {len(vocab)}, merges {len(merges)}")

encoder = bytes_to_unicode()
with open("data/vocab.json", "w", encoding="utf-8") as f:
    json.dump({str(k): "".join(encoder[b] for b in v) for k, v in vocab.items()}, f)
with open("data/merges.txt", "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for a, b in merges:
        f.write("".join(encoder[x] for x in a) + " " + "".join(encoder[x] for x in b) + "\n")

# Verify the round-trip is now lossless.
tok = Tokenizer.from_files("data/vocab.json", "data/merges.txt", special_tokens=["<|endoftext|>"])
print(f"reloaded: vocab {len(tok.vocab)}, merges {len(tok.merges)}")
missing = [(a, b) for a, b in tok.merges if (a + b) not in tok._bytes_to_id]
print(f"merges whose result NOT in vocab: {len(missing)}")
print("b'upon' in vocab:", b"upon" in tok._bytes_to_id)
sample = tok.encode("Once upon a time there was a little girl")
print("encode/decode sanity:", repr(tok.decode(sample)))
