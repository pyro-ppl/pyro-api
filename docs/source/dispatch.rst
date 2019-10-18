Dispatch
========

It's easiest to see how to use pyroapi by example:

.. code-block:: python

    from pyroapi import distributions as dist
    from pyroapi import infer, ops, optim, pyro, pyro_backend

    # These model and guide are backend-agnostic.
    def model():
        locs = pyro.param("locs", ops.tensor([0.2, 0.3, 0.5]))
        p = ops.tensor([0.2, 0.3, 0.5])
        with pyro.plate("plate", len(data), dim=-1):
            x = pyro.sample("x", dist.Categorical(p))
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    def guide():
        p = pyro.param("p", ops.tensor([0.5, 0.3, 0.2]))
        with pyro.plate("plate", len(data), dim=-1):
            pyro.sample("x", dist.Categorical(p))

    # We can now set a backend at inference time.
    with pyro_backend("numpyro"):
        elbo = infer.Trace_ELBO(ignore_jit_warnings=True)
        adam = optim.Adam({"lr": 1e-6})
        inference = infer.SVI(model, guide, adam, elbo)
        for step in range(10):
            loss = inference.step(*args, **kwargs)
            print(f"step {step} loss = {loss}")

.. automodule:: pyroapi.dispatch
.. autofunction:: pyroapi.dispatch.pyro_backend

Generic Modules
---------------
- pyro - The main pyro module.
- distributions - Includes distributions.transforms and distributions.constraints.
- handlers - Generalizing the original pyro.poutine.
- infer - Inference algorithms.
- optim - Optimization utilities.
- ops - Basic tensor operations (like numpy or torch).
