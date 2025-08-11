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


def _get_pyspark_lazy_constructor() -> Callable[
    [Data], PySparkDataFrame
]:  # pragma: no cover
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


@lru_cache(maxsize=1)
def _ibis_backend() -> IbisDuckDBBackend:  # pragma: no cover
    """Cached (singleton) in-memory backend to ensure all tables exist within the same in-memory database."""
    import ibis

    return ibis.duckdb.connect()


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

    @property
    def is_pandas(self) -> bool:
        """Check if the constructor is a Pandas constructor."""
        return self in {
            ConstructorName.PANDAS,
            ConstructorName.PANDAS_PYARROW,
            ConstructorName.PANDAS_NULLABLE,
        }

    @property
    def needs_pyarrow(self) -> bool:
        """Check if the constructor requires PyArrow."""
        return self in {
            ConstructorName.PYARROW,
            ConstructorName.PANDAS_PYARROW,
            ConstructorName.MODIN_PYARROW,
        }

    @classmethod
    def from_pytest_request(cls, request: pytest.FixtureRequest) -> ConstructorName:
        """Get the constructor name from the pytest request."""
        constructor_id = str(request.node.callspec.id)
        return cls(constructor_id)


DEFAULT_CONSTRUCTORS = {
    ConstructorName.PANDAS,
    ConstructorName.PANDAS_PYARROW,
    ConstructorName.POLARS_EAGER,
    ConstructorName.PYARROW,
    ConstructorName.DUCKDB,
    ConstructorName.SQLFRAME,
    ConstructorName.IBIS,
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
        import pandas as pd

        return pd.DataFrame(obj)


class PandasNullableConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS_NULLABLE

    def __call__(self, obj: Data) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")


class PandasPyArrowConstructor(ConstructorEagerBase):
    name = ConstructorName.PANDAS_PYARROW

    def __call__(self, obj: Data) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")


class ModinConstructor(ConstructorEagerBase):  # pragma: no cover
    name = ConstructorName.MODIN

    def __call__(self, obj: Data) -> NativeFrame:
        import modin.pandas as mpd
        import pandas as pd

        df = mpd.DataFrame(pd.DataFrame(obj))
        return cast("NativeFrame", df)


class ModinPyArrowConstructor(ConstructorEagerBase):  # pragma: no cover
    name = ConstructorName.MODIN_PYARROW

    def __call__(self, obj: Data) -> NativeFrame:
        import modin.pandas as mpd
        import pandas as pd

        df = mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")
        return cast("NativeFrame", df)


class CudfConstructor(ConstructorEagerBase):  # pragma: no cover
    name = ConstructorName.CUDF
    needs_gpu = True

    def __call__(self, obj: Data) -> NativeFrame:
        import cudf

        df = cudf.DataFrame(obj)
        return cast("NativeFrame", df)


class PolarsEagerConstructor(ConstructorEagerBase):
    name = ConstructorName.POLARS_EAGER

    def __call__(self, obj: Data) -> pl.DataFrame:
        import polars as pl

        return pl.DataFrame(obj)


class PolarsLazyConstructor(ConstructorLazyBase):
    name = ConstructorName.POLARS_LAZY

    def __call__(self, obj: Data) -> pl.LazyFrame:
        import polars as pl

        return pl.LazyFrame(obj)


class DuckDBConstructor(ConstructorLazyBase):
    name = ConstructorName.DUCKDB

    def __call__(self, obj: Data) -> duckdb.DuckDBPyRelation:
        import duckdb
        import polars as pl

        duckdb.sql("""set timezone = 'UTC'""")

        _df = pl.LazyFrame(obj)
        return duckdb.table("_df")


class DaskLazyConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.DASK

    def __init__(self, npartitions: int = 2) -> None:
        self.npartitions = npartitions

    def __call__(self, obj: Data) -> NativeLazyFrame:
        import dask.dataframe as dd

        return cast("NativeLazyFrame", dd.from_dict(obj, npartitions=self.npartitions))


class PyArrowConstructor(ConstructorEagerBase):
    name = ConstructorName.PYARROW

    def __call__(self, obj: dict[str, Any]) -> pa.Table:
        import pyarrow as pa

        return pa.table(obj)


class PySparkConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.PYSPARK

    def __call__(self, obj: Data) -> PySparkDataFrame:
        constructor = _get_pyspark_lazy_constructor()
        return constructor(obj)


class SQLFramePySparkLazyConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.SQLFRAME

    def __call__(self, obj: Data) -> SQLFrameDataFrame:
        from sqlframe.duckdb import DuckDBSession

        session = DuckDBSession()
        return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])


class IbisConstructor(ConstructorLazyBase):  # pragma: no cover
    name = ConstructorName.IBIS

    def __call__(self, obj: Data) -> ibis.Table:
        import polars as pl

        ldf = pl.from_dict(obj).lazy()
        table_name = str(uuid.uuid4())
        return _ibis_backend().create_table(table_name, ldf)


ALL_CONSTRUCTORS: set[ConstructorBase] = {
    PandasConstructor(),
    PandasNullableConstructor(),
    PandasPyArrowConstructor(),
    ModinConstructor(),
    ModinPyArrowConstructor(),
    CudfConstructor(),
    PolarsEagerConstructor(),
    PolarsLazyConstructor(),
    DuckDBConstructor(),
    DaskLazyConstructor(),
    PyArrowConstructor(),
    PySparkConstructor(),
    SQLFramePySparkLazyConstructor(),
    IbisConstructor(),
}

ALL_CONSTRUCTORS_MAP: dict[ConstructorName, ConstructorBase] = {
    constructor.name: constructor for constructor in ALL_CONSTRUCTORS
}

ALL_CPU_CONSTRUCTORS = {
    constructor.name for constructor in ALL_CONSTRUCTORS if not constructor.needs_gpu
}


def get_constructor(constructor_name: ConstructorName) -> ConstructorBase:
    """Get a constructor by name."""
    if constructor_name not in ALL_CONSTRUCTORS_MAP:
        msg = f"Constructor {constructor_name} not found."
        raise ValueError(msg)
    return ALL_CONSTRUCTORS_MAP[constructor_name]


Constructor: TypeAlias = ConstructorBase
ConstructorEager: TypeAlias = ConstructorEagerBase
ConstructorLazy: TypeAlias = ConstructorLazyBase
