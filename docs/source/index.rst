.. cosolvkit documentation master file, created by
   sphinx-quickstart on Fri May  3 14:26:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
CosolvKit: a versatile tool for cosolvent MD preparation and analysis
========================================

**CosolvKit** is one of the most **generalizable** and **flexible** **open-source** cosolvent systems preparation tool. It is developed on top of the **OpenMM** echosystem and allows the user to specify every kind of cosolvent molecule(s) in the form of SMILES strings at desired concentrations (M) - or specific number of copies.
It exposes a user friendly interface that ensures reproducibility through the files config.json, cosolvents.json and forcefields.json that define all the setup options. At the same time, CosolvKit has also a powerful python API that ensures high **flexibility** and ease of use.

Availability
------------
CosolvKit can be easily installed with all its dependencies using `pip` or `conda` package managers.

All **source code** is available under the `LGPL License, version 2.1+ <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_ from `github.com/forlilab/cosolvkit <https://github.com/forlilab/cosolvkit>`_ and the Python Package index `https://pypi.org/project/cosolvkit <pypi.org/project/cosolvkit>`_.

Participating
-------------
Please report bugs or enhancement requests through the `Issue Tracker <https://github.com/forlilab/cosolvkit/issues>`_.

CosolvKit is **open source** and welcomes your contributions. `Fork the repository on GitHub <https://github.com/forlilab/cosolvkit>`_ and submit a pull request.


Welcome to cosolvkit's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Manual

   installation
   citations
   changes

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   get_started
   cmdline
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Python Documentation

   cosolvkit