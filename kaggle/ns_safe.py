#!/usr/bin/env python
"""Run nerfstudio's ns-eval / ns-export / ns-render with PyTorch >= 2.6.

PyTorch 2.6 made torch.load(weights_only=True) the default; nerfstudio checkpoints contain numpy
scalars and fail with "Unsupported global: numpy.core.multiarray.scalar". This wrapper restores
weights_only=False for the checkpoint you trained yourself, then calls the nerfstudio entrypoint.

    python ns_safe.py eval   --load-config CFG --output-path eval.json --render-output-path renders
    python ns_safe.py export gaussian-splat --load-config CFG --output-dir export
    python ns_safe.py render camera-path --load-config CFG ...
"""
import functools
import sys

import torch

torch.load = functools.partial(torch.load, weights_only=False)

COMMANDS = {"eval": ("nerfstudio.scripts.eval", "ns-eval"),
            "export": ("nerfstudio.scripts.exporter", "ns-export"),
            "render": ("nerfstudio.scripts.render", "ns-render")}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    module, prog = COMMANDS[sys.argv[1]]
    sys.argv = [prog] + sys.argv[2:]
    import importlib
    importlib.import_module(module).entrypoint()
