# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

pytestmark = pytest.mark.filterwarnings(
    "ignore::numpyro.compat.util.UnsupportedAPIWarning",
    "ignore:.*loss does not support models with discrete latent variables:UserWarning",
    # The behavior of using // with negative numbers is changed in PyTorch.
    # But we don't need to worry about it. This UserWarning will be removed in
    # a future version of PyTorch.
    "ignore:.*floordiv.* is deprecated, and its behavior will change in a future version:UserWarning",
)

PACKAGE_NAME = {
    "pyro": "pyro",
    "minipyro": "pyro",
    "numpy": "numpyro",
    "funsor": "funsor",
}


@pytest.fixture(params=["pyro", "minipyro", "numpy", "funsor"])
def backend(request):
    pytest.importorskip(PACKAGE_NAME[request.param])
    with pyro_backend(request.param):
        yield


# TODO(fehiepsi): Remove the following when the test passes in numpyro.
_test_mcmc_run_ok = test_mcmc_run_ok  # noqa F405


@pytest.mark.parametrize("backend", [
    "pyro",
    pytest.param("numpy", marks=[
        pytest.mark.xfail(reason="numpyro signature for MCMC is not consistent.")])])
def test_mcmc_run_ok(backend):
    pytest.importorskip(PACKAGE_NAME[backend])
    with pyro_backend(backend):
        _test_mcmc_run_ok(backend)
