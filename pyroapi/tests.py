import pytest

from pyroapi.dispatch import distributions as dist
from pyroapi.dispatch import infer, ops, optim, pyro


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("optim_name", ["Adam", "ClippedAdam"])
def test_optimizer(backend, optim_name, jit):

    def model(data):
        p = pyro.param("p", ops.tensor(0.5))
        pyro.sample("x", dist.Bernoulli(p), obs=data)

    def guide(data):
        pass

    data = ops.tensor(0.)
    pyro.get_param_store().clear()
    Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
    elbo = Elbo(ignore_jit_warnings=True)
    optimizer = getattr(optim, optim_name)({"lr": 1e-6})
    inference = infer.SVI(model, guide, optimizer, elbo)
    for i in range(2):
        inference.step(data)
