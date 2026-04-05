# abstractrl

Research workspace around **neural program synthesis** and **library learning**: we extend and modify the **DeepCoder-style** guided search stack (LambdaBeam / AbstractBeam lineage) implemented in PyTorch, with room for **reinforcement learning** on top of the policy defined by the LSTM argument selectors and encoders.

## What lives here

- **`src/AbstractBeam/`** — vendored **AbstractBeam** / Crossbeam code: bottom-up synthesis with **library learning** (Stitch), `DeepCoderModel`, encoders (`LambdaSig*`), `LSTMArgSelector`, training in `crossbeam/experiment/train_eval.py`. This is the natural place to **edit DeepCoder behaviour** (architecture, loss, search, RL hooks).
- **Repo root (`pyproject.toml`)** — **JAX**-oriented dependencies (experiments, notebooks) managed with **uv** (`package = false`: no installable root package).

## Installation (one line of flow)

From the repository root:

```bash
./launch.sh
```

This script: installs **uv** via the official **curl** installer, runs **`uv sync --group dev`**, then **`uv run pytest tests/`**.

Manual equivalent:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv sync --group dev
```

Optional — PyTorch stack for AbstractBeam (watch disk: one CUDA **or** CPU torch build only):

```bash
uv sync --group dev --extra abstractbeam
```

See `src/AbstractBeam/README.md` for training configs, TSP domain notes, and cache cleanup (`make clean-caches`).

## Tests

```bash
uv run pytest tests/
```

Smoke tests for optional CUDA / `torch_scatter` (from repo root, if AbstractBeam env is set up):

```bash
make test-abstractbeam-gpu
```

## License

See [LICENSE](LICENSE).
