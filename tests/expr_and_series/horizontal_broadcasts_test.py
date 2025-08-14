from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_sumh_broadcasting(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(
        any=nw.any_horizontal(nw.sum("a", "b").cast(nw.Boolean), ignore_nulls=True),
        all=nw.all_horizontal(nw.sum("a", "b").cast(nw.Boolean), ignore_nulls=True),
        max=nw.max_horizontal(nw.sum("a"), nw.sum("b")),
        min=nw.min_horizontal(nw.sum("a"), nw.sum("b")),
        sum=nw.sum_horizontal(nw.sum("a"), nw.sum("b")),
        mean=nw.mean_horizontal(nw.sum("a"), nw.sum("b")),
    ).sort("i")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "i": [0, 1, 2],
        "any": [True, True, True],
        "all": [True, True, True],
        "max": [15, 15, 15],
        "min": [6, 6, 6],
        "sum": [21, 21, 21],
        "mean": [10.5, 10.5, 10.5],
    }
    assert_equal_data(result, expected)
