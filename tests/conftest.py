import grmdil.amp  # noqa: F401 — runs silence_xla_gpu_probe_warning before jax is imported below

import jax

jax.config.update("jax_enable_x64", True)
