"""Generate text from a trained TransformerLM checkpoint (Problem `generate`).

Usage example:
    python generate.py \
        --checkpoint runs/tinystories/ckpt_20000.pt \
        --vocab runs/tinystories/vocab.json \
        --merges runs/tinystories/merges.txt \
        --prompt "Once upon a time" \
        --max_tokens 100 --temperature 0.8 --top_p 0.9
"""

import argparse

import torch

from cs336_basics.decoding import generate
from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Checkpoint + tokenizer files.
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .pt checkpoint")
    p.add_argument("--vocab", type=str, required=True, help="path to tokenizer vocab.json")
    p.add_argument("--merges", type=str, required=True, help="path to tokenizer merges.txt")
    # Model hyperparameters (must match the checkpoint).
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Generation controls.
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=None, help="nucleus-sampling threshold, e.g. 0.9")
    p.add_argument("--no_eos", action="store_true", help="ignore <|endoftext|> and write until --max_tokens (for longer stories)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    print(f"vocab size: {len(tokenizer.vocab)}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"loaded checkpoint from {args.checkpoint} (iteration {checkpoint.get('iteration', '?')})")

    ids = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.context_length,
        device=device,
        suppress_eos=args.no_eos,
    )

    text = tokenizer.decode(ids)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
