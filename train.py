"""Train a TransformerLM on a tokenized dataset (Problem `training_together`).

Also implements experiment logging (Problem `experiment_log`): every
``--log_every`` steps we evaluate the validation loss (if a validation set is
given) and append a JSONL record with step, train loss, validation loss,
learning rate, and wall-clock time so learning curves can be plotted later.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.nn import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    # Data / output
    p.add_argument("--data", type=str, required=True, help="path to .npy file of token ids")
    p.add_argument("--val_data", type=str, default=None, help="optional .npy validation split")
    p.add_argument("--out_dir", type=str, default="runs/tinystories", help="dir to save checkpoints/logs")
    # Model (TinyStories base config from the assignment: ~17M non-embedding params)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=5000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Logging / checkpointing / evaluation
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument("--eval_iters", type=int, default=20, help="validation batches to average")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, vocab_size, device, eval_iters):
    """Average validation loss over ``eval_iters`` fresh batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.jsonl")

    # --- Load tokenized data (1D numpy array of int64 token ids) ---
    data = np.asarray(np.load(args.data), dtype=np.int64)
    print(f"loaded {len(data)} tokens from {args.data}")
    val_data = np.asarray(np.load(args.val_data), dtype=np.int64) if args.val_data else None
    if val_data is not None:
        print(f"loaded {len(val_data)} validation tokens from {args.val_data}")

    # --- Build the model ---
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model has {n_params / 1e6:.1f}M parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # --- Training loop ---
    t0 = time.time()
    for step in range(args.max_steps):
        lr = get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(data, args.batch_size, args.context_length, device)

        optimizer.zero_grad()
        logits = model(x)  # (batch, seq, vocab)
        loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            wall_time = time.time() - t0
            val_loss = None
            if val_data is not None:
                val_loss = evaluate(
                    model, val_data, args.batch_size, args.context_length,
                    args.vocab_size, device, args.eval_iters,
                )
            record = {
                "step": step,
                "train_loss": round(loss.item(), 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "lr": lr,
                "wall_time": round(wall_time, 3),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            print(f"step {step:6d} | loss {loss.item():.4f} | val {val_str} | "
                  f"lr {lr:.2e} | {wall_time:.1f}s")

        if (step + 1) % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(args.out_dir, f"ckpt_{args.max_steps}.pt")
    save_checkpoint(model, optimizer, args.max_steps, ckpt_path)
    print(f"done. final checkpoint saved to {ckpt_path}; log at {log_path}")


if __name__ == "__main__":
    main()
