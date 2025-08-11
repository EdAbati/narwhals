from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import Frame
    from tests.utils import Constructor


data = {"a": [1, 2, 3]}


def _get_expected_namespace(constructor: Constructor) -> Any | None:  # noqa: PLR0911
    """Get expected namespace module for a given constructor."""
    if constructor.name.is_pandas:
        import pandas as pd

        return pd
    elif "polars" in constructor.name.value:
        import polars as pl

        return pl
    elif constructor.name == "pyarrow":
        import pyarrow as pa

        return pa
    elif constructor.name == "duckdb":
        import duckdb

        return duckdb
    elif constructor.name == "cudf":  # pragma: no cover
        import cudf

        return cudf
    elif "modin" in constructor.name.value:
        import modin.pandas as mpd

        return mpd
    elif "dask" in constructor.name.value:
        import dask.dataframe as dd

        return dd
    elif constructor.name == "ibis":
        import ibis

        return ibis
    elif constructor.name == "sqlframe":
        import sqlframe

        return sqlframe
    return None  # pragma: no cover


def test_native_namespace_frame(constructor: Constructor) -> None:
    if "pyspark" in constructor.name.value:
        pytest.skip(reason="Requires special handling for spark local vs spark connect")

    expected_namespace = _get_expected_namespace(constructor=constructor)

    df: Frame = nw.from_native(constructor(data))
    assert nw.get_native_namespace(df) is expected_namespace
    assert nw.get_native_namespace(df.to_native()) is expected_namespace
    assert nw.get_native_namespace(df.lazy().to_native()) is expected_namespace


def test_native_namespace_series(constructor_eager: Constructor) -> None:
    expected_namespace = _get_expected_namespace(constructor=constructor_eager)

    df: Frame = nw.from_native(constructor_eager(data), eager_only=True)

    assert nw.get_native_namespace(df["a"].to_native()) is expected_namespace
    assert nw.get_native_namespace(df, df["a"].to_native()) is expected_namespace


def test_get_native_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="Could not get native namespace"):
        nw.get_native_namespace(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one object"):
        nw.get_native_namespace()


def test_get_native_namespace_invalid_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    with pytest.raises(ValueError, match="Found objects with different"):
        nw.get_native_namespace(pd.Series([1]), pl.Series([2]))
