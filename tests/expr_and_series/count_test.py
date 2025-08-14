from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager, Data

data: Data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}


def test_count(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b", "z").count())
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)


def test_count_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": [df["a"].count()], "b": [df["b"].count()], "z": [df["z"].count()]}
    expected = {"a": [3], "b": [2], "z": [1]}
    assert_equal_data(result, expected)
