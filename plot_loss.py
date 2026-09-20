"""Plot the train/val loss curves from runs/tinystories/log.jsonl."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

recs = [json.loads(line) for line in open("runs/tinystories/log.jsonl", encoding="utf-8")]
steps = [r["step"] for r in recs]
train = [r["train_loss"] for r in recs]
val = [r["val_loss"] for r in recs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, train, label="train loss", linewidth=1.2)
ax.plot(steps, val, label="val loss", linewidth=1.2)
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy loss")
ax.set_title("TransformerLM on TinyStories (22.7M params, 40000 steps)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("runs/tinystories/loss_curve.png", dpi=150)
print("saved runs/tinystories/loss_curve.png")
print(f"final val loss = {val[-1]:.4f} (best {min(val):.4f})")
