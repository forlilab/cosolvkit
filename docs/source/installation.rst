.. _installation:

Installing cosolvkit
###############

Installation (from PyPI)
*******************
Please note that CosolvKit requires Python 3.10.

.. code-block:: bash

    $ pip install cosolvkit

If using conda, ``pip`` installs the package in the active environment.
This installation doesn't take care of dependencies too since some of them would take too long to be resolved in the conda envinroment.
To install cosolvkit dependencies:

.. code-block:: bash

    $ conda create -n cosolvkit --file cosolvkit_env.yml && conda activate cosolvkit


Installation from source code
*******************

.. code-block:: bash

    $ git clone git@github.com:forlilab/cosolvkit.git
    $ cd cosolvkit
    $ conda create -n cosolvkit --file cosolvkit_env.yml
    $ pip install .


If you wish to make the code for CosolvKit **editable** without having to re-run ``pip install .``, instead use

.. code-block:: bash

    $ pip install --editable .

Test installation
*******************

.. code-block:: python
    import cosolvkit
    from cosolvkit.cosolvent_system import CosolventSystem