# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

pytestmark = pytest.mark.filterwarnings("ignore::numpyro.compat.util.UnsupportedAPIWarning")


@pytest.fixture(params=["pyro", "minipyro", "numpy", "funsor"])
def backend(request):
    with pyro_backend(request.param):
        yield
