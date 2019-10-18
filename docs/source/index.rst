.. pyroapi documentation master file, created by
   sphinx-quickstart on Fri Oct 18 13:54:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pyro API
========

The ``pyroapi`` package dynamically dispatches among multiple Pyro backends, including standard Pyro_, NumPyro_, Funsor_, and custom user-defined backends.
This package includes both **dispatch** mechanisms for use in model and inference code, and **testing** utilities to help develop and test new Pyro backends.

.. _Pyro: https://pyro.ai
.. _NumPyro: https://num.pyro.ai
.. _Funsor: https://funsor.pyro.ai

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   dispatch
   testing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
