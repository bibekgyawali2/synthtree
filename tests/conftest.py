import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from synthtree.config import load_any  # noqa: E402
from synthtree.tree import generate_tree  # noqa: E402

SPECIES = load_any(os.path.join(ROOT, "configs", "species.yaml"))


@pytest.fixture(scope="session")
def species():
    return SPECIES


@pytest.fixture(scope="session")
def small_tree():
    return generate_tree(SPECIES["broad_sparse"], 1234)


@pytest.fixture(scope="session")
def needle_tree():
    return generate_tree(SPECIES["needle_sparse"], 99)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
