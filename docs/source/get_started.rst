.. _get_started:

Get started!
============

**CosolvKit** is a tool entirely developed in Python that allows the user to specify any type of cosolvent molecule starting from SMILES strings and a concentration (M) or the number of copies desired. It populates a cubic box with the required concentration (or number of copies) for each molecule of cosolvent and fill the remaining volume with water or any other type of solvent specified as SMILES string.

Besides its **flexibility**, the strength of CosolvKit is the **reproducibility**, in fact by using "recipe" files to configure the system preparation it is highly reproducible as well as easy to setup and run.

Furthermore, CosolvKit provides the user a standard **MD simulation** protocol to simulate the cosolvent systems exploiting the **OpenMM** echosystem, but it also allows the user to run custom simulations with all the major MD engines (Amber, GROMACS, NAMD). 

Finally, CosolvKit's post-processing pipeline is useful to create and visualize PyMol sessions to highlight cosolvent molecules densities during the simulation. Radial Distribution Function (RDF) plots are produced for each atom of cosolvent molecule with respect to every other atom of the same cosolvent type and with the oxygen atoms of the water solvent.