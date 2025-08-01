from __future__ import annotations

import os
import uuid
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, cast

import pytest

from narwhals._utils import generate_temporary_column_name

if TYPE_CHECKING:
    import duckdb
    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from pyspark.sql import DataFrame as PySparkDataFrame
    from typing_extensions import TypeAlias

    from narwhals._spark_like.dataframe import SQLFrameDataFrame
    from narwhals.typing import DataFrameLike, NativeFrame, NativeLazyFrame

    Data: TypeAlias = "dict[str, list[Any]]"

Constructor: TypeAlias = Callable[[Any], "NativeLazyFrame | NativeFrame | DataFrameLike"]
ConstructorEager: TypeAlias = Callable[[Any], "NativeFrame | DataFrameLike"]
ConstructorLazy: TypeAlias = Callable[[Any], "NativeLazyFrame"]


def pandas_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj)


def pandas_nullable_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")


def pandas_pyarrow_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")


def modin_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj))
    return cast("NativeFrame", df)


def modin_pyarrow_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")
    return cast("NativeFrame", df)


def cudf_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import cudf

    df = cudf.DataFrame(obj)
    return cast("NativeFrame", df)


def polars_eager_constructor(obj: Data) -> pl.DataFrame:
    import polars as pl

    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Data) -> pl.LazyFrame:
    import polars as pl

    return pl.LazyFrame(obj)


def duckdb_lazy_constructor(obj: Data) -> duckdb.DuckDBPyRelation:
    import duckdb
    import polars as pl

    duckdb.sql("""set timezone = 'UTC'""")

    _df = pl.LazyFrame(obj)
    return duckdb.table("_df")


def dask_lazy_p1_constructor(obj: Data) -> NativeLazyFrame:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeLazyFrame", dd.from_dict(obj, npartitions=1))


def dask_lazy_p2_constructor(obj: Data) -> NativeLazyFrame:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeLazyFrame", dd.from_dict(obj, npartitions=2))


def pyarrow_table_constructor(obj: dict[str, Any]) -> pa.Table:
    import pyarrow as pa

    return pa.table(obj)


def pyspark_lazy_constructor() -> Callable[[Data], PySparkDataFrame]:  # pragma: no cover
    pytest.importorskip("pyspark")
    import warnings
    from atexit import register

    is_spark_connect = bool(os.environ.get("SPARK_CONNECT", None))

    if TYPE_CHECKING:
        from pyspark.sql import SparkSession
    elif is_spark_connect:
        from pyspark.sql.connect.session import SparkSession
    else:
        from pyspark.sql import SparkSession

    with warnings.catch_warnings():
        # The spark session seems to trigger a polars warning.
        # Polars is imported in the tests, but not used in the spark operations
        warnings.filterwarnings(
            "ignore", r"Using fork\(\) can cause Polars", category=RuntimeWarning
        )
        builder = cast("SparkSession.Builder", SparkSession.builder).appName("unit-tests")

        session = (
            (
                builder.remote(f"sc://localhost:{os.environ.get('SPARK_PORT', '15002')}")
                if is_spark_connect
                else builder.master("local[1]").config("spark.ui.enabled", "false")
            )
            .config("spark.default.parallelism", "1")
            .config("spark.sql.shuffle.partitions", "2")
            # common timezone for all tests environments
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

        register(session.stop)

        def _constructor(obj: Data) -> PySparkDataFrame:
            _obj = deepcopy(obj)
            index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
            _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))

            return (
                session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()])
                .repartition(2)
                .orderBy(index_col_name)
                .drop(index_col_name)
            )

        return _constructor


def sqlframe_pyspark_lazy_constructor(obj: Data) -> SQLFrameDataFrame:  # pragma: no cover
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])


@lru_cache(maxsize=1)
def _ibis_backend() -> IbisDuckDBBackend:  # pragma: no cover
    """Cached (singleton) in-memory backend to ensure all tables exist within the same in-memory database."""
    import ibis

    return ibis.duckdb.connect()


def ibis_lazy_constructor(obj: Data) -> ibis.Table:  # pragma: no cover
    import polars as pl

    ldf = pl.from_dict(obj).lazy()
    table_name = str(uuid.uuid4())
    return _ibis_backend().create_table(table_name, ldf)


class ConstructorName(str, Enum):
    """Constructor names used in tests."""

    PANDAS = "pandas"
    PANDAS_PYARROW = "pandas[pyarrow]"
    PANDAS_NULLABLE = "pandas[nullable]"
    PYARROW = "pyarrow"
    MODIN = "modin"
    MODIN_PYARROW = "modin[pyarrow]"
    CUDF = "cudf"
    POLARS_EAGER = "polars[eager]"
    POLARS_LAZY = "polars[lazy]"
    DUCKDB = "duckdb"
    DASK = "dask"
    PYSPARK = "pyspark"
    PYSPARK_CONNECT = "pyspark[connect]"
    SQLFRAME = "sqlframe"
    IBIS = "ibis"

    def is_any_pyspark(self) -> bool:
        """Check if the constructor is any PySpark related."""
        return self in {ConstructorName.PYSPARK, ConstructorName.PYSPARK_CONNECT}

    def is_eager(self) -> bool:
        """Check if the constructor is an eager constructor."""
        return self in _EAGER_CONSTRUCTORS

    def is_lazy(self) -> bool:
        """Check if the constructor is a lazy constructor."""
        return self in _LAZY_CONSTRUCTORS

    def is_gpu(self) -> bool:
        """Check if the constructor is a GPU constructor."""
        return self in _GPU_CONSTRUCTORS

    def to_constructor(self) -> ConstructorBase:
        """Return the constructor function associated with this constructor name."""
        if self.is_any_pyspark():
            return PySparkConstructor()
        if self in _EAGER_CONSTRUCTORS:
            return _EAGER_CONSTRUCTORS[self]
        if self in _LAZY_CONSTRUCTORS:
            return _LAZY_CONSTRUCTORS[self]
        if self in _GPU_CONSTRUCTORS:
            return _GPU_CONSTRUCTORS[self]
        msg = f"Unknown constructor name: {self}"
        raise ValueError(msg)

    @classmethod
    def from_pytest_request(cls, request: pytest.FixtureRequest) -> ConstructorName:
        """Get the constructor name from the pytest request."""
        constructor_id = str(request.node.callspec.id)
        return cls(constructor_id)


def get_all_cpu_constructors() -> list[ConstructorName]:
    """Return all CPU constructors."""
    return [
        name
        for name in ConstructorName
        if name not in _GPU_CONSTRUCTORS
        and name
        not in {
            ConstructorName.MODIN,  # too slow
            ConstructorName.PYSPARK_CONNECT,  # complex local setup; can't run together with local spark
        }
    ]


def get_all_default_constructors() -> set[ConstructorName]:
    """Return all default constructors."""
    return {
        ConstructorName.PANDAS,
        ConstructorName.PANDAS_PYARROW,
        ConstructorName.POLARS_EAGER,
        ConstructorName.PYARROW,
        ConstructorName.DUCKDB,
        ConstructorName.SQLFRAME,
        ConstructorName.IBIS,
    }


_EAGER_CONSTRUCTORS: dict[ConstructorName, ConstructorEager] = {
    ConstructorName.PANDAS: pandas_constructor,
    ConstructorName.PANDAS_NULLABLE: pandas_nullable_constructor,
    ConstructorName.PANDAS_PYARROW: pandas_pyarrow_constructor,
    ConstructorName.PYARROW: pyarrow_table_constructor,
    ConstructorName.MODIN: modin_constructor,
    ConstructorName.MODIN_PYARROW: modin_pyarrow_constructor,
    ConstructorName.CUDF: cudf_constructor,
    ConstructorName.POLARS_EAGER: polars_eager_constructor,
}
_LAZY_CONSTRUCTORS: dict[ConstructorName, ConstructorLazy] = {
    ConstructorName.DASK: dask_lazy_p2_constructor,
    ConstructorName.POLARS_LAZY: polars_lazy_constructor,
    ConstructorName.DUCKDB: duckdb_lazy_constructor,
    ConstructorName.PYSPARK: pyspark_lazy_constructor,  # type: ignore[dict-item]
    ConstructorName.SQLFRAME: sqlframe_pyspark_lazy_constructor,
    ConstructorName.IBIS: ibis_lazy_constructor,
}
_GPU_CONSTRUCTORS: dict[ConstructorName, ConstructorEager] = {
    ConstructorName.CUDF: cudf_constructor
}


class ConstructorBase:
    name: ConstructorName
    needs_gpu: bool = False

    def __call__(self, obj: Data) -> NativeLazyFrame | NativeFrame | DataFrameLike: ...


class ConstructorEagerBase(ConstructorBase):
    def __call__(self, obj: Data) -> NativeFrame | DataFrameLike: ...


class ConstructorLazyBase(ConstructorBase):
    def __call__(self, obj: Data) -> NativeLazyFrame: ...


class PandasConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS

    def __call__(self, obj: Data) -> pd.DataFrame:
        return pandas_constructor(obj)


class PolarsEagerConstructor(ConstructorEagerBase):
    name = ConstructorName.POLARS_EAGER

    def __call__(self, obj: Data) -> pl.DataFrame:
        return polars_eager_constructor(obj)


class PySparkConstructor(ConstructorLazyBase):
    name = ConstructorName.PYSPARK

    def __call__(self, obj: Data) -> PySparkDataFrame:
        return pyspark_lazy_constructor()(obj)
