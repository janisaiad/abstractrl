#!/usr/bin/env python3
"""
Graph Coloring trace training + repair search in one file (v3 entrypoint).

v3 intentionally reuses the stabilized v2 core, which already includes:
  * teacher traces + solver-generated traces
  * TRM-style policy/value prior
  * explicit hard-k solving
  * anytime profiling hooks (--profile-every / --profile-out)
  * primitive_calls and anytime_trace in solve outputs
  * dense reward with slack / patchability features

This file exists so the rest of the v3 experimental stack can depend on a
versioned entrypoint without changing the underlying implementation.
"""

from gcp_trace_abstractbeam_v2 import *  # noqa: F401,F403
from gcp_trace_abstractbeam_v2 import main


if __name__ == "__main__":
    main()
