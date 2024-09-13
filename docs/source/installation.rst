.. _installation:

Installing cosolvkit
####################

Installation from conda/mamba
*Please note that Apple M1 chips are not supported by some of CosolvKit's dependencies.
we recommend macOS users of Apple Silicon install the x86_64 version of MiniForge and run CosolvKit through Rosetta.*

.. code-block:: bash

    $ conda install --channel cosolvkit


Installation (from PyPI)
************************
Please note that CosolvKit requires Python >=3.10.

.. code-block:: bash

    $ pip install cosolvkit

If using conda, ``pip`` installs the package in the active environment.
This installation doesn't take care of dependencies too since some of them would take too long to be resolved in the conda envinroment.
To install cosolvkit dependencies:

.. code-block:: bash

    $ conda create -n cosolvkit --file environment.yml && conda activate cosolvkit


Installation from source code
*****************************

.. code-block:: bash

    $ git clone git@github.com:forlilab/cosolvkit.git
    $ cd cosolvkit
    $ conda create -n cosolvkit --file environment.yml && conda activate cosolvkit
    $ pip install .


If you wish to make the code for CosolvKit **editable** without having to re-run ``pip install .``, instead use

.. code-block:: bash

    $ pip install --editable .

Test installation
*******************

.. code-block:: python
    
    import cosolvkit
    from cosolvkit.cosolvent_system import CosolventSystem