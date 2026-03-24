# NanoGPT From Scratch

A step-by-step implementation of a GPT-style language model built from scratch using PyTorch, trained on the Tiny Shakespeare dataset.

## Overview

This notebook follows Andrej Karpathy's approach to building a character-level GPT model incrementally — starting from a simple bigram model and progressively adding self-attention, multi-head attention, feed-forward layers, and transformer blocks.

## Dataset

**Tiny Shakespeare** — a ~1MB text file containing works of Shakespeare, used for character-level language modeling.

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Model Architecture (Final GPT Decoder)

The final model is a decoder-only transformer with the following components:

- **Token + Position Embeddings**
- **Multi-Head Self-Attention** (with causal masking)
- **Feed-Forward Network** (per-token MLP)
- **Transformer Blocks** (stacked attention + FFN)
- **Layer Normalization** (BatchNorm1d explored)
- **Linear head** for next-token prediction

## Hyperparameters

| Parameter      | Value |
|----------------|-------|
| `batch_size`   | 16    |
| `block_size`   | 32    |
| `n_embd`       | 64    |
| `n_head`       | 4     |
| `n_layer`      | 4     |
| `max_iters`    | 5000  |
| `learning_rate`| 1e-3  |
| `dropout`      | 0.0   |

## Building Steps

1. **Data loading & tokenization** — character-level vocabulary, encode/decode
2. **Bigram Language Model** — baseline model using embedding table
3. **Self-Attention** — mathematical trick with `tril` masking and softmax
4. **Single Head Attention** — `Head` class with key, query, value projections
5. **Multi-Head Attention** — parallel attention heads concatenated
6. **Feed-Forward layer** — simple MLP applied after attention
7. **Transformer Block** — combines attention + FFN with residual connections
8. **Full GPT Decoder** — stacked blocks with layer norm and final linear head

## Requirements

```
torch
```

## Usage

Run the notebook cells in order. The final cell trains the model and generates Shakespeare-like text.

## References

- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
