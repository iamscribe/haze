# knock.knock


**Hybrid Attention Language Model — NumPy Inference**

A character-level language model that combines two attention mechanisms:

- **Reweight Attention**: learns positional patterns directly (rhythm, n-grams)
- **Content Attention**: classic QK^T semantic similarity

Plus **entropy-aware adaptive temperature** for self-regulating generation.

## Philosophy

Traditional attention computes relevance dynamically via `softmax(QK^T/√d)`. Reweight-GPT simplifies this: each position learns a static “attention pattern” over the context window, modulated by content.

This creates a model that’s:

- **Simpler** — fewer parameters, faster inference
- **Pattern-aware** — captures linguistic rhythm and structure
- **Interpretable** — attention weights are directly learned

The hybrid approach combines both: positional patterns + semantic similarity.

## Architecture

```
Input → Embedding + PosEmb → [Block × N] → LayerNorm → Output
                                ↓
                     ┌─────────────────────┐
                     │   Hybrid Attention  │
                     │  α·Reweight + (1-α)·Content │
                     └─────────────────────┘
                                ↓
                          GELU MLP
                                ↓
                         Residual + LN
```

### Attention Types

|Type        |Mechanism                   |Captures                   |
|------------|----------------------------|---------------------------|
|**Reweight**|`x @ W_reweight → (T,T)`    |Positional patterns, rhythm|
|**Content** |`(Q @ K^T) / √d`            |Semantic similarity        |
|**Hybrid**  |`α·Reweight + (1-α)·Content`|Both                       |

### Entropy-Aware Sampling

Instead of fixed temperature, the model adapts based on output entropy:

- **High entropy** (uncertain) → lower temperature (more focused)
- **Low entropy** (confident) → higher temperature (more exploration)

This maintains consistent “surprise level” across different contexts.

## Files

```
reweight-gpt/
├── mini_nn.py              # NumPy primitives (activations, sampling, metrics)
├── reweight_gpt_numpy.py   # Main model (inference only)
├── run.py                  # Interactive REPL
├── example.py              # Quick demo script
├── train.py                # PyTorch training (optional)
├── text.txt                # Your corpus (create this)
└── reweight_gpt_weights.npz # Trained weights (after training)
```

## Quick Start

### 1. Create corpus

```bash
echo "your text here..." > text.txt
```

### 2. Run demo (random weights)

```bash
python example.py
```

### 3. Interactive REPL

```bash
python run.py
```

### 4. Train (optional, requires PyTorch)

```bash
python train.py
```

## REPL Commands

```
/len N          generation length (default: 300)
/temp X         base temperature (default: 1.0)
/sampling MODE  basic|top_k|top_p|entropy
/topk K         top-k value (default: 40)
/topp P         top-p nucleus (default: 0.9)
/entropy T      target entropy for adaptive mode
/bounds MIN MAX temperature bounds for adaptive
/stats          toggle stats display
/config         show current config
/help           show help
/quit           exit
```

## Sampling Strategies

### Basic

Standard temperature sampling.

### Top-K

Only sample from top K most likely tokens.

### Top-P (Nucleus)

Dynamic vocabulary based on cumulative probability. More adaptive than top-k.

### Entropy-Aware

Adaptive temperature that targets specific entropy level. Self-regulating:

```python
model.generate(
    seed_seq=idx,
    sampling="entropy",
    target_entropy=3.0,  # bits
    min_temp=0.3,
    max_temp=2.0,
)
```

## API

```python
from reweight_gpt_numpy import Vocab, ReweightGPT

# Build from text
vocab = Vocab.from_text("your corpus here")
model = ReweightGPT(
    vocab_size=vocab.vocab_size,
    T=32,              # context window
    n_emb=64,          # embedding dim
    nodes=64,          # MLP hidden dim
    n_blocks=3,        # transformer blocks
    n_heads=4,         # attention heads
    head_type="hybrid", # "hybrid", "reweight", or "content"
    alpha=0.5,         # reweight/content mix
)

# Generate
seed_idx = vocab.encode("starting text")
tokens, stats = model.generate(
    seed_seq=seed_idx,
    length=200,
    sampling="entropy",
    target_entropy=3.0,
)
text = vocab.decode(tokens)
print(text)
print(f"Mean entropy: {stats['mean_entropy']:.2f}")
```

## Stats Output

```
┌─────────────────────────────────────┐
│           Generation Stats          │
├─────────────────────────────────────┤
│  Mean entropy:      2.45 bits       │
│  Entropy range:   [1.20, 3.80]      │
│  Entropy σ:       0.523             │
│  Mean confidence: 0.312             │
│  Mean temperature:0.850             │
└─────────────────────────────────────┘
```

## Dependencies

**Inference** (NumPy only):

```bash
pip install numpy
```

**Training** (optional):

```bash
pip install torch numpy
```

## Credits

Architecture inspired by experiments with attention mechanism simplification and entropy-based sampling regulation.

Part of a larger exploration into emergent linguistic behavior in minimal neural architectures.

-----

*presence > intelligence*
