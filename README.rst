spiketools
==========

|ProjectStatus|_ |Version|_ |BuildStatus|_ |Coverage|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/spiketools.svg
.. _Version: https://pypi.python.org/pypi/spiketools/

.. |BuildStatus| image:: https://github.com/spiketools/SpikeTools/actions/workflows/build.yml/badge.svg
.. _BuildStatus: https://github.com/spiketools/SpikeTools/actions/workflows/build.yml

.. |Coverage| image:: https://codecov.io/gh/spiketools/spiketools/branch/main/graph/badge.svg
.. _Coverage: https://codecov.io/gh/spiketools/spiketools

.. |License| image:: https://img.shields.io/pypi/l/spiketools.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/spiketools.svg
.. _PythonVersions: https://pypi.python.org/pypi/spiketools/

.. |Publication| image:: https://joss.theoj.org/papers/10.21105/joss.05268/status.svg
.. _Publication: https://doi.org/10.21105/joss.05268

``spiketools`` is a collection of tools and utilities for analyzing spiking data.

Overview
--------

``spiketools`` is an open-source module for processing and analyzing single-unit activity from neuro-electrophysiological recordings.

Available sub-modules in ``spiketools`` include:

- ``measures`` : measures and conversions that can be applied to spiking data
- ``objects`` : objects that can be used to manage spiking data
- ``spatial`` : space related functionality and measures
- ``stats`` : statistical measures for analyzing spiking data
- ``sim`` : simulations of spiking activity and related functionality
- ``plts`` : plotting functions for visualizing spike data and related measures
- ``utils`` : additional utilities for working with spiking data

Scope
-----

``spiketools`` is currently organized around analyses of single cell activity.

The current scope does not include population measures, though this may be extended in the future.

Note that ``spiketools`` does *not* cover spike sorting.
Check out `spikeinterface <https://github.com/SpikeInterface/>`_ for spike sorting.

Documentation
-------------

Documentation for ``spiketools`` is available
`here <https://spiketools.github.io/>`_.

The documentation includes:

- `Tutorials <https://spiketools.github.io/spiketools/auto_tutorials/index.html>`_: which describe and provide examples for each sub-module
- `API List <https://spiketools.github.io/spiketools/api.html>`_: which lists and describes everything available in the module
- `Glossary <https://spiketools.github.io/spiketools/glossary.html>`_: which defines key terms used in the module

If you have a question about using ``spiketools`` that doesn't seem to be covered by the documentation, feel free to
open an `issue <https://github.com/spiketools/spiketools/issues>`_ and ask!

Dependencies
------------

``spiketools`` is written in Python, and requires Python >= 3.6 to run.

It has the following required dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `pandas <https://github.com/pandas-dev/pandas>`_
- `scipy <https://github.com/scipy/scipy>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_

There are also optional dependencies, that offer extra functionality:

- `statsmodels <https://github.com/statsmodels/statsmodels>`_ is needed for some statistical measures, for example ANOVAs
- `pytest <https://github.com/pytest-dev/pytest>`_ is needed to run the test suite locally

We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ distribution to manage these requirements.

Installation
------------

The current release version of `spiketools` is the 0.X.X series.

See the `changelog <https://spiketools.github.io/spiketools/changelog.html>`_ for notes on major version releases.

**Stable Release Version**

To install the latest stable release, you can use pip:

.. code-block:: shell

    $ pip install spiketools

Optionally, to include dependencies required for the `stats` module:

.. code-block:: shell

    $ pip install spiketools[stats]

**Development Version**

To get the current development version, first clone this repository:

.. code-block:: shell

    $ git clone https://github.com/spiketools/spiketools

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

**Editable Version**

To install an editable version, download the development version as above, and run:

.. code-block:: shell

    $ pip install -e .

Reference
---------

If you use this code in your project, please cite:

.. code-block:: text

    Donoghue, T., Maesta-Pereira, S., Han C. Z., Qasim, S. E., & Jacobs, J. (2023) spiketools: A Python package for 
    analyzing single-unit neural activity. Journal of Open Source Software, 8(91), 5268. DOI: 10.21105/joss.05268

Direct Link: https://doi.org/10.21105/joss.05268

For bibtex citation information, see the
`citation file <https://github.com/spiketools/spiketools/blob/main/CITATION.cff>`_.

Contribute
----------

This project welcomes and encourages contributions from the community!

To file bug reports and/or ask questions about this project, please use the
`Github issue tracker <https://github.com/spiketools/spiketools/issues>`_.

To see and get involved in discussions about the module, check out:

- the `issues board <https://github.com/spiketools/spiketools/issues>`_ for topics relating to code updates, bugs, and fixes
- the `development page <https://github.com/spiketools/Development>`_ for discussion of potential major updates to the module

When interacting with this project, please use the
`contribution guidelines <https://github.com/spiketools/spiketools/blob/main/CONTRIBUTING.md>`_
and follow the
`code of conduct <https://github.com/spiketools/spiketools/blob/main/CODE_OF_CONDUCT.md>`_.
