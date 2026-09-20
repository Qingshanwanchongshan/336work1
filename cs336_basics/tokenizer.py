"""Byte-level Byte-Pair Encoding (BPE) tokenizer, in the style of GPT-2."""

from collections import Counter
from collections.abc import Iterable, Iterator
import json
import os
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN = regex.compile(PAT)


def bytes_to_unicode() -> dict[int, str]:

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):

        self.vocab = dict(vocab)
        self.merges = merges

        self._bytes_to_id = {token: i for i, token in self.vocab.items()}

        self._merge_ranks = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = list(special_tokens) if special_tokens else []

        self._special_token_ids = {}
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes in self._bytes_to_id:
                self._special_token_ids[special] = self._bytes_to_id[special_bytes]
            else:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = special_bytes
                self._bytes_to_id[special_bytes] = new_id
                self._special_token_ids[special] = new_id

        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(regex.escape(t) for t in sorted_tokens) + ")"
            self._special_regex = regex.compile(special_pattern)
        else:
            self._special_regex = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从磁盘持久化文件中直接构建 Tokenizer 实例（官方测试强制要求实现）[cite: 1]。"""
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            vocab = {
                int(k): bytes([byte_decoder[ch] for ch in v]) if isinstance(v, str) else bytes(v)
                for k, v in raw_vocab.items()
            }

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    token1 = bytes([byte_decoder[ch] for ch in parts[0]])
                    token2 = bytes([byte_decoder[ch] for ch in parts[1]])
                    merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self._special_regex is None:
            return self._encode_ordinary(text)

        ids = []
        for part in self._special_regex.split(text):
            if not part:
                continue
            if part in self._special_token_ids:
                ids.append(self._special_token_ids[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids = []
        for match in PATTERN.finditer(text):
            pretoken = match.group()
            token_bytes = list(pretoken.encode("utf-8"))
            for merged in self._bpe(token_bytes):
                ids.append(self._bytes_to_id[merged])
        return ids

    def _bpe(self, token_bytes: list[int]) -> list[bytes]:
        parts = [bytes([b]) for b in token_bytes]
        while len(parts) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self._merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            parts = parts[:best_idx] + [parts[best_idx] + parts[best_idx + 1]] + parts[best_idx + 2:]
        return parts


    def decode(self, ids: list[int]) -> str:

        raw = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return raw.decode("utf-8", errors="replace")


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            cut = self._safe_prefix_len(buffer)
            if cut > 0:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def _safe_prefix_len(self, text: str) -> int:
        matches = list(PATTERN.finditer(text))
        if not matches:
            return 0
        cut = matches[-1].start()

        if self._special_regex is not None:
            for m in self._special_regex.finditer(text):
                if m.start() < cut < m.end():
                    cut = m.start()
            for special in self.special_tokens:
                for k in range(len(special) - 1, 0, -1):
                    if text.endswith(special[:k]):
                        start = len(text) - k
                        if start < cut:
                            cut = start
                        break
        return cut



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(regex.escape(t) for t in sorted_tokens) + ")"
        special_set = set(special_tokens)
 
        segments = [p for p in regex.split(special_pattern, corpus) if p and p not in special_set]
    else:
        segments = [corpus]


    word_freqs = Counter()
    for seg in segments:
        for match in PATTERN.finditer(seg):
            pretoken = match.group()
            token_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            word_freqs[token_bytes] += 1

    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for special in special_tokens:
        st_bytes = special.encode("utf-8")
        if st_bytes not in vocab:
            vocab.append(st_bytes)

    pair_counts = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], p)
        )

        if pair_counts[best_pair] <= 0:
            break

        a, b = best_pair
        merged_bytes = a + b
        merges.append((a, b))
        vocab.append(merged_bytes)

        new_word_freqs = {}
        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue

            has_target = False
            for i in range(len(word) - 1):
                if word[i] == a and word[i + 1] == b:
                    has_target = True
                    break

            if not has_target:
                new_word_freqs[word] = freq
                continue

            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] -= freq

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            nw = tuple(new_word)
            for i in range(len(nw) - 1):
                pair_counts[(nw[i], nw[i + 1])] += freq

            new_word_freqs[nw] = freq

        word_freqs = new_word_freqs
        pair_counts.pop(best_pair, None)

    return {i: token for i, token in enumerate(vocab)}, merges
