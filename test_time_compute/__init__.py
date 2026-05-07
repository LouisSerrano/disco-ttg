"""Test-time compute package.

Importing this package ensures the disco-ttg repo root is at sys.path[0], so that
`from train.train import ...` resolves to disco-ttg/train rather than to a
collision (e.g. an installed `geps` editable package that ships a top-level train.py).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path or sys.path.index(_REPO_ROOT) != 0:
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)
