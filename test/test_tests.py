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
}


@pytest.fixture(params=["pyro", "minipyro", "numpy"])
def backend(request):
    pytest.importorskip(PACKAGE_NAME[request.param])
    with pyro_backend(request.param):
        yield
