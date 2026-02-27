# NanoGPT-RigVeda

Character-level language modeling experiments on Rig Veda text using PyTorch.

This repo contains:
- `bigram.py`: baseline bigram language model
- `self_attention.py`: GPT-style decoder with masked self-attention
- `download_data.py`: builds `data/input.txt` from the CLTK Sanskrit corpus layout

## Project Structure

- `data/input.txt`: training corpus consumed by both models
- `data/output.txt`: generated text output from `self_attention.py`
- `check_gpu.py`: quick PyTorch + CUDA check
- `test.py`: small LayerNorm/Dropout behavior check

## Data Source

The corpus is sourced from:
- `https://github.com/cltk/sanskrit_parallel_sacred_texts`

`download_data.py` expects the local path:
- `data/rig_veda/`

It scans subfolders and concatenates files ending with `eng.txt` into `data/input.txt`.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install torch
```

3. (Optional) Verify GPU:

```bash
python check_gpu.py
```

## Prepare Data

1. Place the Rig Veda corpus under:

```text
data/rig_veda/
```

2. Build the training text file:

```bash
python download_data.py
```

This creates/overwrites:
- `data/input.txt`

## Train Models

### 1) Bigram Baseline

```bash
python bigram.py
```

Expected behavior:
- prints periodic train/validation loss
- prints generated characters to stdout

### 2) Self-Attention Model

```bash
python self_attention.py
```

Expected behavior:
- prints periodic train/validation loss
- writes generated text to:
  - `data/output.txt`

## Main Hyperparameters (`self_attention.py`)

- `chunk_size`: context window length
- `batch_size`: sequences per training step
- `learning_rate`: optimizer step size
- `max_iters`: total training iterations
- `n_embed`: embedding width
- `n_head`: number of attention heads
- `n_layer`: number of transformer blocks
- `dropout`: dropout probability

## Notes

- This is a character-level model (not subword tokenization).
- `data/input.txt` must exist before training.
- Larger `n_embed`/`n_layer` improves capacity but increases memory usage.
