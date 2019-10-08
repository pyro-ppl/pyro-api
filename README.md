[![Build Status](https://travis-ci.com/pyro-ppl/pyro-api.svg?branch=master)](https://travis-ci.com/pyro-ppl/pyro-api)

----------------------------------------------------------------------------------------------------
# Pyro API

Pyro API for modeling and inference for generic backend dispatch.


## Testing

For testing different backends, install pytest and other dependencies and run the test suite:

```
pip install -e .[test]
pytest
```

For using these models to test your own backend implementation, you can use the following pattern:

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
        model, model_args, model_kwargs = f['model'], f.get('model_args', ()), f.get('model_kwargs', {})
        model(*model_args) 
        ... # further testing
``` 
