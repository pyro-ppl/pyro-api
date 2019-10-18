Testing
=======

The pyroapi package includes tests to ensure new backends conform to the standard API, indeed these tests serve as the formal API description.
To add tests to your new backend say in ``project/test/`` follow these steps (or see the example_ in funsor):

.. _example: https://github.com/pyro-ppl/funsor/tree/master/test/pyroapi

1. Create a new directory ``project/test/pyroapi/``.

2. Create a file ``project/test/pyroapi/conftest.py`` and a hook to treat missing features as xfail:

.. code-block:: python

    import pytest


    def pytest_runtest_call(item):
        try:
            item.runtest()
        except NotImplementedError as e:
            pytest.xfail(str(e))

3. Create a file ``project/test/pyroapi/test_pyroapi.py`` and define a ``backend`` fixture:

.. code-block:: python

    import pytest
    from pyroapi import pyro_backend
    from pyroapi.tests import *  # noqa F401

    @pytest.yield_fixture
    def backend():
        with pyro_backend("my_backend"):
            yield

4. Test your backend with pytest

.. code-block:: bash

    pytest -vx test/pyroapi
