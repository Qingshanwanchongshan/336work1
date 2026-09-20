"""Data loading: sample language-modeling batches from a token sequence."""

import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    """Sample a batch of (x, y) pairs from a 1D token sequence.

    dataset: 1D numpy array of integer token ids.
    Returns two torch.LongTensors of shape (batch_size, context_length): the
    input sequences and their corresponding next-token labels (offset by one).
    """
    n = len(dataset)
    # Valid start indices go from 0 to n - context_length - 1 so that
    # both x[i:i+L] and y[i+1:i+1+L] stay in bounds.
    start_idx = np.random.randint(0, n - context_length, size=(batch_size,))
    x = torch.tensor(np.stack([dataset[i : i + context_length] for i in start_idx]), dtype=torch.long)
    y = torch.tensor(np.stack([dataset[i + 1 : i + context_length + 1] for i in start_idx]), dtype=torch.long)
    return x.to(device), y.to(device)
