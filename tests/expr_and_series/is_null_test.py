from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager, Data

data_na: Data = {"a": [None, 3, 2], "z": [7.0, None, None]}


def test_null(constructor: Constructor) -> None:
    expected = {"a": [True, False, False], "z": [True, False, False]}
    df = nw.from_native(constructor(data_na))
    result = df.select(nw.col("a").is_null(), ~nw.col("z").is_null())

    assert_equal_data(result, expected)


def test_null_series(constructor_eager: ConstructorEager) -> None:
    expected = {"a": [True, False, False], "z": [True, False, False]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = {"a": df["a"].is_null(), "z": ~df["z"].is_null()}

    assert_equal_data(result, expected)
