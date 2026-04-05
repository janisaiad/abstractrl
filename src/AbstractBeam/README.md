# AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning

This repository contains the source code associated with the [paper](https://arxiv.org/abs/2405.17514) submited to NeurIPS 2024 :

In this research project, we aim to reduce the search space blowup in Program Synthesis. For this purpose, we train a neural model to learn a search
policy for bottom-up execution-guided program synthesis and extend it using DSL Enhancement.



## Setup
Install **one** PyTorch build (CUDA *or* CPU, not both) and matching `torch-scatter` — duplicate venvs or repeated `pip install torch` can cost **many GB** (wheels + cache). Prefer a single env (e.g. repo root `uv sync --extra abstractbeam`) and run `make clean-caches` from `abstractrl` to purge pip/uv caches after experiments.

Legacy pins: torch==2.2.1, torch-scatter==2.1.2, python=3.8.19. Or:
```
pip install -r requirements.txt
```
## File structure

The synthetic training data is saved to  `./neurips/abstractbeam/data` and  `./neurips/lambdabeam/data`.
Make sure to also create `./neurips/abstractbeam/models` and `./neurips/abstractbeam/results` directories. Same goes when you want to train the LambdaBeam benchmark.

## Train or eval the model
Navigate to `crossbeam/experiment/deepcoder` directory, and select the config you want to run.
Just adapt the config path to point to `./crossbeam/experiment/deepcoder/configs/` + [`train/abstractbeam.py`, `train/baseline.py`, `eval/abstractbeam_eval.py`, `eval/lambdabeam_eval.py`].
You can make any necessary edits to the selected config file including the data, model, and result directories.
Moreover, you can adapt the hyperparameters, e.g., the enumeration timeout, number of GPUs to use, ... .
To start run below from the project's root (the number of GPUs set in the config file must align with the number selected in below script):

```
./crossbeam/experiment/deepcoder/run_deepcoder.sh
```

## TSP (ATSP 3 villes) sur GPU

Domaine `tsp` : longueur de tournée minimale pour 3 villes (deux permutations depuis la ville 0), même DSL que DeepCoder (`Min`, `Add`, `Access`).

Depuis la racine de ce dépôt AbstractBeam, avec PyTorch CUDA et `torch-scatter` installés :

```
./run_tsp_gpu.sh
```

Cela écrit `neurips/tsp/data/train-weight-3-00000.pkl`, puis lance l’entraînement avec `crossbeam/experiment/deepcoder/configs/train/tsp_gpu.py` (`num_proc` / `gpu_list` : ajuster pour plusieurs GPU). Évaluation : config `eval/tsp_eval.py` (charger `model-best-valid.ckpt` dans `neurips/tsp/models`).
