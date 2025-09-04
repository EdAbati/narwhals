from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests._constructors import Constructor, ConstructorLazyBase, ConstructorName, Data

if TYPE_CHECKING:
    import ibis
    import polars as pl

else:
    ibis = pytest.importorskip("ibis")
    pl = pytest.importorskip("polars")


@pytest.fixture
def ibis_constructor() -> Constructor:
    class IbisConstructor(ConstructorLazyBase):
        name = ConstructorName.IBIS

        def __call__(self, data: Data) -> ibis.Table:
            df = pl.DataFrame(data)
            return ibis.memtable(df)

    return IbisConstructor()


def test_from_native(ibis_constructor: Constructor) -> None:
    df = nw.from_native(ibis_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.columns == ["a", "b"]
