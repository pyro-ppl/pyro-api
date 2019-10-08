[![Build Status](https://travis-ci.com/pyro-ppl/pyro-api.svg?branch=master)](https://travis-ci.com/pyro-ppl/pyro-api)

# Pyro API

Generic API for modeling and inference for dispatch to different Pyro backends.
----------------------------------------------------------------------------------------------------

## Testing

For testing API compatibility on different backends, install pytest and other test dependencies that includes backends like [funsor](https://github.com/pyro-ppl/funsor) and [numpyro](https://github.com/pyro-ppl/numpyro) and run the test suite:

```
pip install -e .[test]
pytest -vs
```

This library has no dependencies and can easily be installed for testing your particular Pyro backend
implementation. You can use the following pattern and test your backend on models in the `pyroapi.testing`
module.

```python
from pyro_api.dispatch import pyro_backend
from pyro_api.testing import MODELS


# Register backend
with pyro_backend(handlers='my_backend.handlers', 
                  distributions='my_backend.distributions',
                  ...):
                  
    # Test on models in pyro_api.testing
    for model in MODELS:
        f = MODELS[model]()
        model, model_args = f['model'], f.get('model_args', ())
        model(*model_args) 
        ... # further testing
``` 
