# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

pytestmark = pytest.mark.filterwarnings("ignore::numpyro.compat.util.UnsupportedAPIWarning")

PACKAGE_NAME = {
    "pyro": "pyro",
    "minipyro": "pyro",
    "numpy": "numpyro",
    "funsor": "funsor",
}


@pytest.fixture(params=[
    "pyro",
    "minipyro",
    "numpy",
    pytest.param("funsor", marks=[pytest.mark.xfail(
        reason="temporarily blocked by https://github.com/pyro-ppl/funsor/pull/327")]),
])
def backend(request):
    pytest.importorskip(PACKAGE_NAME[request.param])
    with pyro_backend(request.param):
        yield
