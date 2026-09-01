"""Independent validation and measurement harness for Team C2.

The package deliberately contains no custom decode kernel.  Its baseline is a
minimal import-time adaptation of the vendored vLLM Triton implementation.
"""

from .data import DecodeProblem, make_decode_problem
from .reference import dense_sparse_attention_reference
from .triton_baseline import run_triton_baseline

__all__ = [
    "DecodeProblem",
    "dense_sparse_attention_reference",
    "make_decode_problem",
    "run_triton_baseline",
]
