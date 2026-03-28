# minGRU Character-Level Language Model

A from-scratch implementation of **minGRU** and **DeepminGRU** in PyTorch, trained on a character-level corpus for autoregressive text generation.

---

## Overview

This project implements the **minGRU** architecture — a simplified Gated Recurrent Unit that strips the GRU down to its essential components while preserving its sequential modeling power. A stacked **DeepminGRU** variant is built on top, trained end-to-end with backpropagation through time (BPTT) to learn character-level language structure.

The model supports two inference modes:
- **Full-sequence forward pass** — processes an entire sequence in one call (used during training)
- **Stateful single-step inference** — processes one token at a time using a saved hidden state (used for autoregressive generation)

---

## Architecture

### `minGRU`

A single minGRU layer with the following components:

| Component | Formula | Description |
|---|---|---|
| Gate | `g_t = sigmoid(x_t @ Ug + bg)` | Controls how much new information to let in |
| Candidate hidden | `h̃_t = x_t @ U + b` | Proposed new hidden state |
| Hidden state | `h_t = (1 - g_t) * h_{t-1} + g_t * h̃_t` | Interpolates old and new state |
| Output | `y_t = φ(h_t @ V + by)` | Projection to output space |

The gate and candidate are computed purely from the input `x_t` (no recurrent weight matrix on the hidden state), making this a minimal but effective recurrent cell.

### `DeepminGRU`

Stacks `n` minGRU layers sequentially:
- Intermediate layers use **Tanh** activation
- The final layer uses **LogSoftmax** to produce a log-probability distribution over characters
- Both `forward` (full sequence) and `step` (single token) are supported at every layer

---

## Training

- **Corpus:** character-level text (Origin dataset, ~10,000 characters)
- **Sequence length:** 15 characters
- **Architecture:** 3-layer DeepminGRU, hidden dim 300
- **Optimizer:** Adam (`lr=0.001`, `β1` tuned)
- **Training:** BPTT, minibatch SGD (`batch_size=128`), 200 epochs
- **Vocabulary:** 27 characters (a–z + space/punctuation)

---

## Results

After training, the model is evaluated on next-character prediction accuracy across all 9,999 sequences in the dataset:

```
Next-letter prediction accuracy: XXXX/9999 = X.XXXX (XX.X%)
```

Autoregressive generation example (prompt in white, model output in red):

```
higher forms of life [... 160 predicted characters ...]
```

---

## Usage

### Install dependencies

```bash
pip install torch numpy matplotlib tqdm
```

### Train the model

```python
from model import DeepminGRU
import torch

net = DeepminGRU(n_layers=3, in_dim=27, hidden_dim=300)
net.bptt(dataloader, epochs=200, lr=0.001)
```

### Generate text

```python
# Pass a prompt sequence, generate 160 characters
pred = net.predict(prompt, n=160)
```

### Single-step inference

```python
net.reset()
for token in sequence:
    y = net.step(token.unsqueeze(0).unsqueeze(0))  # (1, 1, in_dim)
```

### Save / load

```python
torch.save(net.cpu(), 'model.pt')
net = torch.load('model.pt').to(device)
```

---

## File Structure

```
.
├── a4q4.ipynb          # Full implementation and training notebook
├── oos_dataset.py      # Origin dataset loader and character encoding
└── README.md
```

---

## Key Concepts Demonstrated

- **Recurrent architecture from scratch** — gate, candidate hidden state, and interpolation update rule implemented without using `nn.GRU`
- **Stateful vs. stateless inference** — `forward()` for training over full sequences, `step()` for autoregressive decoding
- **Deep stacking** — residual-style layer chaining with per-layer activation control
- **BPTT training** — backpropagation through time over variable-length sequences
- **Autoregressive generation** — sampling continuations token-by-token using the saved hidden state

---

## References

- [Minimal GRU (minGRU) — Feng et al., 2024](https://arxiv.org/abs/2410.01201)
- [Empirical Evaluation of Gated Recurrent Neural Networks — Chung et al., 2014](https://arxiv.org/abs/1412.3555)
