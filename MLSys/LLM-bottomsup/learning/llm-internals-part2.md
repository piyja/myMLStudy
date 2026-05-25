# LLM Internals Part 2 — RoPE, Causal Masking, Sampling, Training Loop

## RoPE (Rotary Positional Encoding)

### One-liner
RoPE encodes position by rotating Q and K vectors before the dot product, so attention scores naturally capture relative distance between tokens rather than absolute positions.

### Why it exists
Attention is permutation-invariant — without positional encoding the model has no sense of word order. Old absolute embeddings (add a fixed vector at input) fail to generalize beyond training length and encode position globally rather than per-pair.

### How it works
- Split each Q/K head dimension into 2D pairs
- Rotate each pair by angle `pos × θᵢ` (different frequency per pair)
- Because both Q and K are rotated before the dot product, the score becomes a function of `(pos_q - pos_k)` — relative distance falls out automatically
- No learned parameters — purely geometric

```
Q_rotated = Rotate(Q, pos_q × θ)
K_rotated = Rotate(K, pos_k × θ)
score = Q_rotated · K_rotated   → encodes relative distance
```

### Key tradeoffs
| Property | Absolute PE | RoPE |
|---|---|---|
| Encodes relative distance | No | Yes |
| Generalizes beyond training length | Poor | Much better |
| Applied | Once at input | Every layer, on Q and K |
| Learned params | Sometimes | None |

### Mental model
Instead of stamping a position tag on each token at the door, RoPE rotates each token's "question" and "answer" vectors like a clock hand. Two tokens close together have similar rotations; far apart tokens have very different rotations. The dot product measures how aligned the rotations are — which is exactly how far apart they are.

---

## Causal Masking

### One-liner
Causal masking fills future positions in the attention score matrix with -inf before softmax, forcing each position to only attend to itself and past tokens — preserving the autoregressive constraint during training.

### Why it exists
Training processes full sequences in one forward pass for efficiency. Without a mask, token at position 2 could attend to position 5, leak the answer, learn nothing useful, and fail at inference when position 5 doesn't exist yet.

### How it works
```
scores[i][j] = Q[i] · K[j] / √head_size

Mask: if j > i → scores[i][j] = -inf

After softmax: e^(-inf) = 0   → future positions contribute nothing
```

Result: lower-triangular attention — each token only aggregates from itself and earlier tokens.

### Is it only used in training?
- **Training**: yes, essential — full sequence processed at once
- **Inference (one token at a time)**: structurally not needed — future tokens don't exist
- **Prompt prefill**: yes — model runs a full forward pass on the prompt, same as training
- **With KV cache**: implicit — you only ever attend to cached past positions

### Mental model
An exam where every student can see their own work and all earlier questions — but a cardboard sheet covers every question that comes after. The mask is that cardboard sheet. It makes training look like inference so the model learns the right behavior.

---

## Sampling Strategies

### One-liner
Sampling strategies control how a next token is chosen from the model's output probability distribution — trading off between determinism, creativity, and quality.

### The pipeline
```
logits (vocab_size raw scores)
  → temperature scaling
  → top-K filter (optional)
  → top-P (nucleus) filter
  → renormalize
  → sample
```

### Strategies

**Greedy (argmax):** always pick highest probability token. Deterministic, repetitive, boring. Only good for strict factual tasks.

**Temperature:**
```
logits_scaled = logits / T
```
- T < 1 → sharpens distribution → more deterministic
- T = 1 → unchanged
- T > 1 → flattens distribution → more creative/random

**Top-K:** keep only the K highest probability tokens, zero the rest, renormalize, sample. Problem: K is fixed regardless of how confident the model is.

**Top-P (nucleus):** keep the smallest set of tokens whose cumulative probability ≥ P. When model is confident the nucleus is small (2-3 tokens); when uncertain it expands. Adapts to model confidence at each step — strictly better than top-K.

### Key tradeoffs
| Strategy | Deterministic | Handles confidence variation | Notes |
|---|---|---|---|
| Greedy | Yes | — | Boring, loops |
| Temperature | No | No | Global reshaping only |
| Top-K | No | No | Fixed window |
| Top-P | No | Yes | Default in production |

### Mental model
The model spreads probability mass like butter across the vocabulary. Temperature controls how thick or thin the spread is. Top-K cuts off everything past the 50th slice. Top-P says "stop once you've covered 90% of the bread" — the number of slices adjusts automatically.

---

## Training Loop (to revisit — incomplete understanding)

### One-liner
The training loop repeatedly runs: forward pass → cross-entropy loss → backprop → AdamW optimizer step, over trillions of tokens, to update every weight in the model.

### The 5 steps

```
for each batch:
  1. forward pass     → logits (B, T, vocab_size)
  2. cross-entropy    → scalar loss = mean(-log(prob[correct_token]))
  3. loss.backward()  → gradient on every weight via chain rule
  4. grad clip        → scale down if global norm > 1.0 (safety valve)
  5. optimizer.step() → AdamW update
```

### Cross-entropy loss
```
loss = -log(prob[correct_token])
```
High confidence on the right answer → small loss. Confidently wrong → large loss (punished exponentially). Perplexity = e^loss — a perplexity of 10 means the model is as confused as a uniform choice over 10 options.

### AdamW optimizer
Tracks two moving averages per weight:
```
m = β₁ × m + (1 - β₁) × gradient      # momentum: smoothed direction
v = β₂ × v + (1 - β₂) × gradient²     # velocity: smoothed magnitude

update = lr × m / (√v + ε)
weight -= update + lr × λ × weight     # λ × weight = weight decay
```
- Momentum smooths noisy gradients → consistent direction
- Velocity gives per-weight adaptive learning rate → rare weights get bigger steps
- Weight decay (the "W") pulls weights toward zero → regularization

### Learning rate schedule
```
warmup (1-2% of steps): lr ramps from ~0 to max_lr
cosine decay (~98%):    lr follows cosine from max_lr to min_lr
```
Warmup prevents catastrophic early steps when gradients are chaotic.

### TODO — come back to
- Intuition for why AdamW beats plain SGD in practice (loss landscape geometry)
- Batch size vs. gradient accumulation
- Mixed precision (fp16/bf16) and why it matters for memory
- Pretraining vs. fine-tuning differences in the loop

---

## How all four pieces connect

```
Token position → RoPE rotates Q,K → relative attention scores
                                          ↓
                              Causal mask blocks future positions
                                          ↓
                              Logits at each position
                         (training: loss vs. known next token)
                         (inference: sampling strategy picks token)
                                          ↓
                              Cross-entropy loss
                                          ↓
                              Backprop → AdamW updates weights
```