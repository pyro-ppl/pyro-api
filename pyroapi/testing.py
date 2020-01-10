# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Models for testing the generic interface.

For specifying the arguments to model functions, the convention followed is
that positional arguments are inputs to the model and keyword arguments denote
observed data.
"""

from collections import OrderedDict

from pyroapi.dispatch import distributions as dist, handlers, ops, pyro

MODELS = OrderedDict()


def register(rng_seed=None):
    def _register_fn(fn):
        MODELS[fn.__name__] = handlers.seed(fn, rng_seed)

    return _register_fn


@register(rng_seed=1)
def logistic_regression():
    N, dim = 3000, 3
    # generic way to sample from distributions
    data = pyro.sample('data', dist.Normal(0., 1.), sample_shape=(N, dim))
    true_coefs = ops.arange(1., dim + 1.)
    logits = ops.sum(true_coefs * data, axis=-1)
    labels = pyro.sample('labels', dist.Bernoulli(logits=logits))

    def model(x, y=None):
        coefs = pyro.sample('coefs', dist.Normal(ops.zeros(dim), ops.ones(dim)))
        intercept = pyro.sample('intercept', dist.Normal(0., 1.))
        logits = ops.sum(coefs * x, axis=-1) + intercept
        return pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

    return {'model': model, 'model_args': (data,), 'model_kwargs': {'y': labels}}


@register(rng_seed=1)
def neals_funnel():
    def model(dim):
        y = pyro.sample('y', dist.Normal(0, 3))
        pyro.sample('x', dist.TransformedDistribution(
            dist.Normal(ops.zeros(dim - 1), 1), dist.transforms.AffineTransform(0, ops.exp(y / 2))))

    return {'model': model, 'model_args': (10,)}


@register(rng_seed=1)
def eight_schools():
    J = 8
    y = ops.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = ops.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def model(J, sigma, y=None):
        mu = pyro.sample('mu', dist.Normal(0, 5))
        tau = pyro.sample('tau', dist.HalfCauchy(5))
        with pyro.plate('J', J):
            theta = pyro.sample('theta', dist.Normal(mu, tau))
            pyro.sample('obs', dist.Normal(theta, sigma), obs=y)

    return {'model': model, 'model_args': (J, sigma), 'model_kwargs': {'y': y}}


@register(rng_seed=1)
def beta_binomial():
    N, D1, D2 = 10, 2, 2
    true_probs = ops.tensor([[0.7, 0.4], [0.6, 0.4]])
    total_count = ops.tensor([[1000, 600], [400, 800]])

    data = pyro.sample('data', dist.Binomial(total_count=total_count, probs=true_probs),
                       sample_shape=(N,))

    def model(N, D1, D2, data=None):
        with pyro.plate("plate_0", D1):
            alpha = pyro.sample("alpha", dist.HalfCauchy(1.))
            beta = pyro.sample("beta", dist.HalfCauchy(1.))
            with pyro.plate("plate_1", D2):
                probs = pyro.sample("probs", dist.Beta(alpha, beta))
                with pyro.plate("data", N):
                    pyro.sample("binomial", dist.Binomial(probs=probs, total_count=total_count), obs=data)

    return {'model': model, 'model_args': (N, D1, D2), 'model_kwargs': {'data': data}}
