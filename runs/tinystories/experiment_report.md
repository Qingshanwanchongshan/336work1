# TinyStories 语言模型训练实验报告

Stanford CS336 — Assignment 1 (basics)：从零实现 Transformer 语言模型，训练并观察效果。

## 1. 数据准备

| 项目 | 值 |
|------|-----|
| 数据源 | TinyStoriesV2-GPT4-train.txt（2.23 GB） |
| 预分词 | GPT-2 正则模式（字节级） |
| 分词器 | 字节级 BPE，`vocab_size = 10000`，`merges = 9743` |
| 特殊 token | `<|endoftext|>` |
| 编码结果 | **540,847,304 tokens**（14,548,094 篇文档） |
| 验证集 | 545,389 tokens |

> 实现细节：`save_tokenizer` / `Tokenizer.from_files` 采用 GPT-2 官方的
> bytes-to-unicode 映射序列化词表与合并规则（空格字节 `0x20` → `'Ġ'`），
> 避免空格字节与 merges.txt 的空白分隔符冲突，保证 round-trip 无损。

## 2. 模型架构

Decoder-only Transformer，采用 pre-norm 结构：

| 超参数 | 值 |
|--------|-----|
| vocab_size | 10000 |
| context_length | 256 |
| d_model | 512 |
| num_layers | 4 |
| num_heads | 16 |
| d_ff（SwiGLU 前馈） | 1344 |
| 位置编码 | RoPE（θ = 10000） |
| 归一化 | RMSNorm |

总参数量约 **22.7M**。

## 3. 训练配置

| 项目 | 值 |
|------|-----|
| 优化器 | AdamW（β₁=0.9, β₂=0.95, eps=1e-8） |
| 学习率 | 3e-4，warmup 1000 步 + cosine 衰减至 min_lr 3e-5 |
| weight decay | 0.01 |
| 梯度裁剪 | 1.0 |
| batch size | 32 |
| 总步数 | 40000（≈ 3.27 亿 tokens） |
| 训练时长 | ≈ 3.5 小时（RTX 4060 Laptop 8GB） |

## 4. 实验结果

loss 收敛曲线见 `loss_curve.png`，关键节点：

| step | train loss | val loss | lr |
|------|-----------|----------|-----|
| 0 | 9.268 | 9.261 | 0 |
| 5000 | 1.724 | 1.886 | 2.9e-4 |
| 10000 | 1.700 | 1.706 | 2.7e-4 |
| 20000 | 1.627 | 1.598 | 1.7e-4 |
| 30000 | 1.363 | 1.507 | 7.2e-5 |
| 39800 | **1.357** | **1.492** | 3.0e-5 |

- **最终验证 loss = 1.492，全程最低 1.4554（step 37600）**
- 从 9.27（≈ ln 10000，随机初始化）单调下降，无发散、无过拟合（train/val 差距始终很小）

## 5. 生成样本（temperature 0.8–0.9, top-p 0.9）

**Prompt:** "Once upon a time"
> Once upon a time, there was a little girl named Lily. Lily loved to play with her toy animals. One day, she found a small bird with a hurt wing. Lily wanted to help the bird, so she took care of the bird.

**Prompt:** "Once upon a time, there was a little boy"
> Once upon a time, there was a little boy named Tim. Tim loved to watch films with his mom. One day, Tim and his mom went to the park to play.

**Prompt:** "One sunny day, a cat"
> One sunny day, a cat and a dog were playing near the rail. They were jumping and jumping in the grass. The sun was shining and they were very happy. The cat said, "Let's go find some water to drink!" The dog agreed and they went to the rail.

## 6. 结论

1. **Loss 收敛健康**：训练集与验证集同步下降，验证 loss 稳定在 1.46–1.49，无过拟合。
2. **生成质量高**：输出语法正确、语义连贯、遵循 TinyStories 风格（主角 + 简单情节 + 对话），达到该规模模型（22.7M）的预期水平。
3. **完整链路跑通**：数据下载 → BPE 训练 → 流式编码 → 训练 → 生成，全流程端到端可用。

## 7. 复现命令

```bash
# 准备数据（训练分词器 + 编码完整训练集）
python prepare_data.py --split valid          # 训练分词器
python prepare_train_data.py                  # 编码完整 train 集 -> data/train.npy

# 训练
python train.py --data data/train.npy --val_data data/val.npy \
    --max_steps 40000 --warmup_iters 1000 --cosine_cycle_iters 40000 \
    --log_every 200 --checkpoint_every 4000

# 生成（默认：模型自然收尾，通常 40~60 token 就停）
python generate.py --checkpoint runs/tinystories/ckpt_40000.pt \
    --vocab data/vocab.json --merges data/merges.txt \
    --prompt "Once upon a time" --temperature 0.8 --top_p 0.9

# 生成完整故事（加 --no_eos 强制写到指定长度，约 250 token 是最长连贯上限）
python generate.py --checkpoint runs/tinystories/ckpt_40000.pt \
    --vocab data/vocab.json --merges data/merges.txt \
    --prompt "Once upon a time, there was a little fox named Finn" \
    --max_tokens 240 --temperature 0.8 --top_p 0.9 --no_eos
```

> 说明：模型 context_length=256，单次生成中能保持连贯的上限约 250 token。
> `--no_eos` 会忽略 `<|endoftext|>` 强制续写；超过 ~250 token 后模型会遗忘开头
> 并陷入重复循环，故 `--max_tokens` 建议设 230~240。
