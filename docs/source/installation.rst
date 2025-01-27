.. _installation:

We recommend using micromamba to manage Python environments and install Meeko.
Other similar package managers also work, like mamba, conda, or miniconda.
We prefer micromamba because it uses conda-forge as its default channel.
If you use other package managers, please use the ``-c conda-forge`` option.

To get micromamba, visit https://mamba.readthedocs.io/en/latest/

From conda-forge
****************
*Please note that Apple M1 chips are not supported by some of CosolvKit's dependencies.
we recommend macOS users of Apple Silicon install the x86_64 version of MiniForge and run CosolvKit through Rosetta.*

.. code-block:: bash

    micromamba install cosolvkit


From PyPI
*********
Please note that CosolvKit requires Python >=3.10.

.. code-block:: bash

    pip install cosolvkit

If using conda, micromamba, or similar, ``pip`` installs the package in the active environment.
This installation doesn't take care of dependencies too since some of them would take too long to be resolved in the conda envinroment.
To install cosolvkit dependencies:

.. code-block:: bash

    micromamba create -n cosolvkit --file environment.yml
    micromamba activate cosolvkit


From source code
****************

.. code-block:: bash

    git clone git@github.com:forlilab/cosolvkit.git
    cd cosolvkit
    micromamba create -n cosolvkit --file environment.yml
    micromamba activate cosolvkit
    pip install .


If you wish to make the code for CosolvKit **editable** without having to re-run ``pip install .``, instead use

.. code-block:: bash

    pip install --editable .

Test installation
*****************

.. code-block:: python
    
    import cosolvkit
    from cosolvkit.cosolvent_system import CosolventSystem
