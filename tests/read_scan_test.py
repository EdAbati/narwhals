from __future__ import annotations

from typing import Any
from typing import Mapping

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.utils import Implementation
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

pytest.importorskip("polars")
import polars as pl

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}
TEST_EAGER_BACKENDS = [
    Implementation.POLARS,
    Implementation.PANDAS,
    Implementation.PYARROW,
    "polars",
    "pandas",
    "pyarrow",
]


@pytest.mark.parametrize(
    "backend",
    TEST_EAGER_BACKENDS,
)
def test_read_csv(
    tmpdir: pytest.TempdirFactory,
    backend: Implementation | str,
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    result = nw.read_csv(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


@pytest.mark.parametrize(
    "backend",
    TEST_EAGER_BACKENDS,
)
def test_read_csv_v1(
    tmpdir: pytest.TempdirFactory, backend: Implementation | str
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    result = nw_v1.read_csv(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw_v1.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_csv_kwargs(tmpdir: pytest.TempdirFactory) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    result = nw.read_csv(filepath, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


def test_scan_csv(
    tmpdir: pytest.TempdirFactory,
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw.from_native(constructor(data))
    backend = nw.get_native_namespace(df)
    result = nw.scan_csv(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


def test_scan_csv_v1(
    tmpdir: pytest.TempdirFactory,
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    df = nw_v1.from_native(constructor(data))
    backend = nw_v1.get_native_namespace(df)
    result = nw_v1.scan_csv(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw_v1.LazyFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_csv_kwargs(tmpdir: pytest.TempdirFactory) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    df_pl.write_csv(filepath)
    result = nw.scan_csv(filepath, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_parquet(
    tmpdir: pytest.TempdirFactory,
    constructor_eager: ConstructorEager,
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    result = nw.read_parquet(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_parquet_v1(
    tmpdir: pytest.TempdirFactory, constructor_eager: ConstructorEager
) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    df = nw_v1.from_native(constructor_eager(data))
    backend = nw_v1.get_native_namespace(df)
    result = nw_v1.read_parquet(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw_v1.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_parquet_kwargs(tmpdir: pytest.TempdirFactory) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    result = nw.read_parquet(filepath, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_parquet(
    tmpdir: pytest.TempdirFactory,
    constructor: Constructor,
) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        from sqlframe.duckdb import DuckDBSession

        kwargs = {"session": DuckDBSession(), "inferSchema": True}
    elif "pyspark" in str(constructor):
        from pyspark.sql import SparkSession

        kwargs = {
            "session": (
                SparkSession.builder.appName("unit-tests")  # pyright: ignore[reportAttributeAccessIssue]
                .master("local[1]")
                .config("spark.ui.enabled", "false")
                .config("spark.default.parallelism", "1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate()
            ),
            "inferSchema": True,
        }
    else:
        kwargs = {}
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    df = nw.from_native(constructor(data))
    backend = nw.get_native_namespace(df)
    result = nw.scan_parquet(filepath, backend=backend, **kwargs)
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


@pytest.mark.parametrize("spark_like_backend", ["pyspark", "sqlframe"])
def test_scan_fail_spark_like_without_session(
    tmpdir: pytest.TempdirFactory, spark_like_backend: str
) -> None:
    _ = pytest.importorskip(spark_like_backend)

    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)

    with pytest.raises(
        ValueError,
        match="Spark like backends require a session object to be passed in `kwargs`.",
    ):
        nw.scan_parquet("file.parquet", backend=spark_like_backend)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_parquet_v1(
    tmpdir: pytest.TempdirFactory,
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    df = nw_v1.from_native(constructor(data))
    backend = nw_v1.get_native_namespace(df)
    result = nw_v1.scan_parquet(filepath, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw_v1.LazyFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_parquet_kwargs(tmpdir: pytest.TempdirFactory) -> None:
    df_pl = pl.DataFrame(data)
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    result = nw.scan_parquet(filepath, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)
