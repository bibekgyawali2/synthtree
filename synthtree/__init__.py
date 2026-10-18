"""synthtree: procedural trees, exact procedural wind, camera paths and ground truth
for a paired calm/windy synthetic RGB-D foliage dataset.

Everything in this package is pure NumPy (no SciPy / YAML / OpenCV at import time)
so the same code runs inside Blender's bundled Python and in host Python.
Host-only helpers (EXR / PNG I/O) live in `synthtree.imageio` and import their
dependencies lazily.
"""

__version__ = "0.1.0"
