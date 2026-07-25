"""Runtime-only compatibility for the frozen FH evaluator manifest."""

import builtins
from typing import Mapping

builtins.Mapping = Mapping
