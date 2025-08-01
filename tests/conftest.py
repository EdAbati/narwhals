from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.constructors_utils import (
    ConstructorName,
    get_all_cpu_constructors,
    get_all_default_constructors,
)
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from tests.utils import Constructor, ConstructorEager

    Data: TypeAlias = "dict[str, list[Any]]"

MIN_PANDAS_NULLABLE_VERSION = (2,)

# When testing cudf.pandas in Kaggle, we get an error if we try to run
# python -m cudf.pandas -m pytest --constructors=pandas. This gives us
# a way to run `python -m cudf.pandas -m pytest` and control which constructors
# get tested.
if default_constructors := os.environ.get(
    "NARWHALS_DEFAULT_CONSTRUCTORS", None
):  # pragma: no cover
    DEFAULT_CONSTRUCTORS = default_constructors
else:
    DEFAULT_CONSTRUCTORS = ",".join(get_all_default_constructors())


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--all-cpu-constructors",
        action="store_true",
        default=False,
        help="run tests with all cpu constructors",
    )
    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="libraries to test",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Function]
) -> None:  # pragma: no cover
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("all_cpu_constructors"):  # pragma: no cover
        selected_constructors = get_all_cpu_constructors()
    else:  # pragma: no cover
        opt = cast("str", metafunc.config.getoption("constructors"))
        selected_constructors = [ConstructorName(c) for c in opt.split(",")]

    eager_constructors: list[ConstructorEager] = []
    eager_constructors_ids: list[str] = []
    constructors: list[Constructor] = []
    constructors_ids: list[str] = []

    for constructor_name in selected_constructors:
        if (
            constructor_name
            in {ConstructorName.PANDAS_NULLABLE, ConstructorName.PANDAS_PYARROW}
            and MIN_PANDAS_NULLABLE_VERSION > PANDAS_VERSION
        ):
            continue  # pragma: no cover

        if constructor_name.is_eager():
            eager_constructors.append(constructor_name.to_constructor())
            eager_constructors_ids.append(str(constructor_name))
        constructors.append(constructor_name.to_constructor())
        constructors_ids.append(str(constructor_name))

    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor_eager", eager_constructors, ids=eager_constructors_ids
        )
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", constructors, ids=constructors_ids)
