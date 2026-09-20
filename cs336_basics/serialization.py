"""Checkpointing: serialize and restore model + optimizer state."""

import torch


def save_checkpoint(model, optimizer, iteration, out):
    """Save model state, optimizer state, and the iteration counter to `out`."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    """Restore model/optimizer state from `src` and return the iteration count."""
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
