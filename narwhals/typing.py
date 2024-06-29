from typing import TYPE_CHECKING  # pragma: no cover
from typing import TypeVar  # pragma: no cover
from typing import Union  # pragma: no cover

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from narwhals._expression import Expr
    from narwhals._series import Series

    IntoExpr: TypeAlias = Union[Expr, str, int, float, Series]

    NativeDataFrame = TypeVar("NativeDataFrame")
    NativeSeries = TypeVar("NativeSeries")
