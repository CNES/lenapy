Contributing
============

Thanks for helping to build lenapy!

Retrieve the code: forking and cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make a fork of the `lenapy repo <https://github.com/CNES/lenapy>`__ and clone
the fork.

A documentation is available on GitHub to help platform users create a fork: `https://docs.github.com/fork-a-repo <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`__

.. code-block:: none

   git clone https://github.com/<your-github-username>/lenapy.git
   cd lenapy

You may want to add ``https://github.com/CNES/lenapy`` as an upstream remote
repository.

.. code-block:: none

   git remote add upstream https://github.com/CNES/lenapy

Creating a Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have conda environment YAML file with all the necessary dependencies.

.. code-block:: none

   conda env create -f environment.yml --name=lenapy-dev

to create a conda environment and install all the dependencies.

Building lenapy
~~~~~~~~~~~~~~~

Lenapy is a pure-python repository. Development installation should be as simple as
cloning the repository and running the following in the cloned directory:

.. code-block:: none

  conda activate lenapy-dev
  python -m pip install --no-deps -e .

If you have any trouble, please open an issue on the
`lenapy issue tracker <https://github.com/CNES/lenapy/issues>`_.

Style
~~~~~

Lenapy uses `black <http://black.readthedocs.io/en/stable/>`_ and `isort <https://isort.readthedocs.io/en/latest/>`_
for formatting. If you installed lenapy with ``python -m pip install -e ".[dev]"`` these tools will already be
installed.

Running tests
~~~~~~~~~~~~~

Lenapy uses `pytest <https://docs.pytest.org/en/latest/>`_ for testing. You
can run tests from the main lenapy directory as follows:

.. code-block:: none

    pytest tests

Alternatively you may choose to run only a subset of the full test suite. For
example to test only the `writers` submodule we would run tests as follows:

.. code-block:: none

    pytest tests/writers

Coverage
~~~~~~~~

It is possible to check code coverage

.. code-block:: none

   pytest --cov=lenapy --cov-report=html

You can still use all the usual pytest command-line options in addition to those.

Pre-Commit Hooks
~~~~~~~~~~~~~~~~

Install and build the `pre commit <https://github.com/pre-commit/pre-commit>`_ tool as:

.. code-block:: none

    python -m pip install -e ".[dev]"
    pre-commit install

to install a few plugins like black, isort, and pylint. These tools will automatically
be run on each commit. You can skip the checks with ``git commit --no-verify``.

Documentation
~~~~~~~~~~~~~

We use `numpydoc <http://numpydoc.readthedocs.io/en/latest/format.html>`_ for our docstrings.

Building the docs is possible with

.. code-block:: none

   $ conda env create -f environment.yml --name=lenapy-dev
   $ conda activate lenapy-dev
   $ python -m pip install -e ".[dev]"
   $ cd doc
   $ sphinx-build -b html doc doc/build
