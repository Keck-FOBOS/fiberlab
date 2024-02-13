
.. include:: include/links.rst

.. _install:

Installation
============

Install Python 3
----------------

The FOBOS FiberLab package is supported for Python 3 only. To install Python,
you can do so along with a full package manager, like `Anaconda`_, or you can
install python 3 directly from `python.org`_.

Python environment
------------------

We strongly recommend you setup a fresh python environment for installation and
use of the FOBOS producer.  See, for example:

 - `Anaconda`_, specifically `Conda Environments`_
 - `virtualenv`_
 - `pyenv`_

Clone the repo
--------------

To download the software and associated data, clone the GitHub repo by
executing:

.. code-block:: console

    git clone https://github.com/Keck-FOBOS/fiberlab.git

This will create a ``fiberlab`` directory in the directory where the command is
executed.

Install from source
-------------------

The preferred method to install ``fiberlab`` and ensure its dependencies are
met is to, from the top-level directory, run:

.. code-block:: console

    pip install -e .

Importantly, note that this will upgrade/install the relevant python package
dependencies, which is why it's useful to isolate the code to its own
environment.  To include the development dependencies, run:

.. code-block:: console

    pip install -e ".[dev]"

Note that use of the quotes is shell dependent; e.g., you need them for
``zshell`` (the current default Mac shell), but they'll cause a fault in
``bash``.

Installation in this way should also mean that changes made to the code
should take effect immediately when re-running code.

Uninstall
---------

Installation via pip is preferred because it eases uninstalling the code:

.. code-block:: console
    
    pip uninstall fobos-fiberlab

----

Test your installation
----------------------

.. warning::

    Tests do not currently exist!  Text here is a placeholder.

If you've installed with the developer dependencies, you can test the
installation using ``pytest``:

.. code-block:: console

    cd fiberlab/tests
    pytest . -W ignore


Problems?
---------

If you have problems, particularly those that you think may be a more general
problem, please `Submit an issue`_.


