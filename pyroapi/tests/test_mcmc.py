# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyroapi.dispatch import distributions as dist
from pyroapi.dispatch import infer, pyro

# Note that the backend arg to these tests must be provided as a
# user-defined fixture that sets the pyro_backend. For demonstration,
# see test/conftest.py.


def assert_ok(model, *args, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.get_param_store().clear()
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, num_samples=2, warmup_steps=2)
    mcmc.run(*args, **kwargs)


def test_mcmc_run_ok(backend):
    if backend not in ["pyro", "numpy"]:
        return

    def model():
        pyro.sample("x", dist.Normal(0, 1))

    assert_ok(model)
