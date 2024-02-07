import re
from typing import *


def check_tensor(data, pattern: str, allow_none: bool = False, **kwargs):
    if allow_none and data is None:
        return {}

    assert bool(
        re.match("^[a-zA-Z0-9 ]+$", pattern)
    ), "Invalid characters in pattern found! Only use [a-zA-Z0-9 ]."

    tokens = [t for t in pattern.split(" ") if t != ""]

    assert len(data.shape) == len(
        tokens
    ), "Input tensor has an invalid number of dimensions!"

    assignment = {}
    for dim, (token, size) in enumerate(zip(tokens, data.shape)):
        if token[0].isdigit():
            assert (
                int(token) == size
            ), f"Tensor mismatch in dimension {dim}: expected {size}, found {int(token)}!"
        else:
            if token in assignment:
                assert (
                    assignment[token] == size
                ), f"Tensor mismatch in dimension {dim}: expected {size}, found {assignment[token]}!"
            else:
                assignment[token] = size

                if token in kwargs:
                    assert (
                        kwargs[token] == size
                    ), f"Tensor mismatch in dimension {dim}: expected {kwargs[token]}, found {size}!"

    return assignment
