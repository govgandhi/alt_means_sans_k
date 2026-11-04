"""Verify the Gensim BLAS wrapper used by this experiment handles a score of -1.

Input: the active environment's compiled Gensim extension. Output: JSON containing
the selected wrapper and exact dot-product checks; failure exits nonzero.
Run: python3 scripts/check_control_gensim.py
The C API signature is declared in Gensim 4.4.0 word2vec_inner.pxd. See upstream
issue https://github.com/piskvorky/gensim/issues/3665 for the original reproduction.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path

from gensim.models import word2vec_inner


def blas_check(strict: bool = True) -> dict[str, object]:
    wrappers = {0: "our_dot_double", 1: "our_dot_float", 2: "our_dot_noblas"}
    name = wrappers[word2vec_inner.FAST_VERSION]
    capsule = word2vec_inner.__pyx_capi__[name]
    get_name = ct.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = [ct.py_object], ct.c_char_p
    get_pointer = ct.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ct.py_object, ct.c_char_p]
    get_pointer.restype = ct.c_void_p
    address = get_pointer(capsule, get_name(capsule))
    integer, real = ct.POINTER(ct.c_int), ct.POINTER(ct.c_float)
    dot = ct.CFUNCTYPE(ct.c_float, integer, real, integer, real, integer)(address)
    n = ct.c_int(1)
    x = (ct.c_float * 1)(1)
    results = {}
    for expected in (-2.0, -1.0, -0.5, 0.0, 1.0):
        y = (ct.c_float * 1)(expected)
        actual = float(dot(ct.byref(n), x, ct.byref(n), y, ct.byref(n)))
        results[str(expected)] = actual
        if strict and actual != expected:
            raise RuntimeError(f"Gensim {name} returned {actual} instead of {expected}")
    return {
        "wrapper": name,
        "dot_products": results,
        "passed": all(
            float(expected) == actual for expected, actual in results.items()
        ),
        "extension_sha256": hashlib.sha256(
            Path(word2vec_inner.__file__).read_bytes()
        ).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(blas_check(), indent=2))
