# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config):
    try:
        import funsor
    except ImportError:
        pass
    else:
        funsor.set_backend("torch")


def pytest_runtest_call(item):
    try:
        item.runtest()
    except NotImplementedError as e:
        pytest.xfail(str(e))
