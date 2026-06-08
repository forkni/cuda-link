# Stub for TouchDesigner ambient globals — checker-only, never imported at runtime.
#
# TD injects these names into the COMP namespace at interpreter startup.
# Files that reference them bare use:
#
#   from typing import TYPE_CHECKING
#   if TYPE_CHECKING:
#       from _td_builtins import op, run, CUDAMemoryShape  # noqa: F401
#
# This file lives on the pyrefly search-path (search-path = ["td_exporter"])
# so that import resolves without any runtime cost.
from typing import Any

op: Any
run: Any
CUDAMemoryShape: type[Any]
