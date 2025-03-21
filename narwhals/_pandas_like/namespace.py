from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals._compliant import EagerNamespace
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.selectors import PandasSelectorNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import align_series_full_broadcast
from narwhals._pandas_like.utils import diagonal_concat
from narwhals._pandas_like.utils import extract_dataframe_comparand
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import vertical_concat
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import TypeAlias

    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version

    _Scalar: TypeAlias = Any


class PandasLikeNamespace(
    EagerNamespace[PandasLikeDataFrame, PandasLikeSeries, PandasLikeExpr]
):
    @property
    def _expr(self) -> type[PandasLikeExpr]:
        return PandasLikeExpr

    @property
    def _series(self) -> type[PandasLikeSeries]:
        return PandasLikeSeries

    @property
    def selectors(self: Self) -> PandasSelectorNamespace:
        return PandasSelectorNamespace(self)

    # --- not in spec ---
    def __init__(
        self: Self,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version

    # --- selection ---
    def lit(self: Self, value: Any, dtype: DType | None) -> PandasLikeExpr:
        def _lit_pandas_series(df: PandasLikeDataFrame) -> PandasLikeSeries:
            pandas_series = self._series.from_iterable(
                data=[value],
                name="literal",
                index=df._native_frame.index[0:1],
                context=self,
            )
            if dtype:
                return pandas_series.cast(dtype)
            return pandas_series

        return PandasLikeExpr(
            lambda df: [_lit_pandas_series(df)],
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> PandasLikeExpr:
        return PandasLikeExpr(
            lambda df: [
                self._series.from_iterable(
                    [len(df._native_frame)], name="len", index=[0], context=self
                )
            ],
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    # --- horizontal ---
    def sum_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)
            native_series = (s.fill_null(0, None, None) for s in series)
            return [reduce(operator.add, native_series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def all_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = align_series_full_broadcast(
                *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.and_, series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def any_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = align_series_full_broadcast(
                *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.or_, series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def mean_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(
                *(s.fill_null(0, strategy=None, limit=None) for s in expr_results)
            )
            non_na = align_series_full_broadcast(*(1 - s.is_null() for s in expr_results))
            return [reduce(operator.add, series) / reduce(operator.add, non_na)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def min_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)

            return [
                PandasLikeSeries(
                    self.concat(
                        (s.to_frame() for s in series), how="horizontal"
                    )._native_frame.min(axis=1),
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                    version=self._version,
                ).alias(series[0].name)
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def max_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)

            return [
                PandasLikeSeries(
                    self.concat(
                        (s.to_frame() for s in series), how="horizontal"
                    )._native_frame.max(axis=1),
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                    version=self._version,
                ).alias(series[0].name)
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def concat(
        self: Self,
        items: Iterable[PandasLikeDataFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> PandasLikeDataFrame:
        dfs: list[Any] = [item._native_frame for item in items]
        if how == "horizontal":
            return PandasLikeDataFrame(
                horizontal_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        if how == "vertical":
            return PandasLikeDataFrame(
                vertical_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )

        if how == "diagonal":
            return PandasLikeDataFrame(
                diagonal_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        raise NotImplementedError

    def when(self: Self, predicate: PandasLikeExpr) -> PandasWhen:
        return PandasWhen(
            predicate, self._implementation, self._backend_version, version=self._version
        )

    def concat_str(
        self: Self,
        *exprs: PandasLikeExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> PandasLikeExpr:
        dtypes = import_dtypes_module(self._version)

        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(
                *(s.cast(dtypes.String()) for s in expr_results)
            )
            null_mask = align_series_full_broadcast(*(s.is_null() for s in expr_results))

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = reduce(lambda x, y: x + separator + y, series).zip_with(
                    ~null_mask_result, None
                )
            else:
                init_value, *values = [
                    s.zip_with(~nm, "") for s, nm in zip(series, null_mask)
                ]

                sep_array = init_value.from_iterable(
                    data=[separator] * len(init_value),
                    name="sep",
                    index=init_value._native_series.index,
                    context=self,
                )
                separators = (sep_array.zip_with(~nm, "") for nm in null_mask[:-1])
                result = reduce(
                    operator.add,
                    (s + v for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )


class PandasWhen:
    def __init__(
        self: Self,
        condition: PandasLikeExpr,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        then_value: PandasLikeExpr | _Scalar = None,
        otherwise_value: PandasLikeExpr | _Scalar = None,
        *,
        version: Version,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._condition: PandasLikeExpr = condition
        self._then_value: PandasLikeExpr | _Scalar = then_value
        self._otherwise_value: PandasLikeExpr | _Scalar = otherwise_value
        self._version = version

    def __call__(self: Self, df: PandasLikeDataFrame) -> Sequence[PandasLikeSeries]:
        condition = self._condition(df)[0]
        condition_native = condition._native_series

        if isinstance(self._then_value, PandasLikeExpr):
            value_series = self._then_value(df)[0]
        else:
            value_series = condition.alias("literal")._from_scalar(self._then_value)
            value_series._broadcast = True
        value_series_native = extract_dataframe_comparand(
            df._native_frame.index, value_series
        )

        if self._otherwise_value is None:
            return [
                value_series._from_native_series(
                    value_series_native.where(condition_native)
                )
            ]

        if isinstance(self._otherwise_value, PandasLikeExpr):
            otherwise_series = self._otherwise_value(df)[0]
        else:
            native_result = value_series_native.where(
                condition_native, self._otherwise_value
            )
            return [value_series._from_native_series(native_result)]
        otherwise_series_native = extract_dataframe_comparand(
            df._native_frame.index, otherwise_series
        )
        return [
            value_series._from_native_series(
                value_series_native.where(condition_native, otherwise_series_native)
            )
        ]

    def then(
        self: Self, value: PandasLikeExpr | PandasLikeSeries | _Scalar
    ) -> PandasThen:
        self._then_value = value

        return PandasThen(
            self,
            depth=0,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )


class PandasThen(PandasLikeExpr):
    def __init__(
        self: Self,
        call: PandasWhen,
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[PandasLikeDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        call_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version
        self._call: PandasWhen = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._call_kwargs = call_kwargs or {}

    def otherwise(
        self: Self, value: PandasLikeExpr | PandasLikeSeries | _Scalar
    ) -> PandasLikeExpr:
        self._call._otherwise_value = value
        self._function_name = "whenotherwise"
        return self
