:: _api:

cosolvkit API
###############################

The CosolvKit API allows for a more advanced, flexible use of CosolvKit where the user can create their own scripts. 

Config class and config.json
###############################

CosolvKit implements a `Config` class to handle the list of setup options.  
In the data folder a template of the `config.json` file used to setup the system building is provided.  
It is also possible to overwrite some of the options from the API:

.. code-block:: python

    from cosolvkit.config import Config
    config = Config.from_config('config.json')

    # Modify the receptor option and the radius value
    config.receptor = FALSE
    config.radius = 8.5

Creating CosolventMolecules without json files
###############################

It is possible to avoid to create a `cosolvents.json` recipe file (although highly recommended).  

.. code-block:: python

    from cosolvkit.cosolvent_system import CosolventMolecule 

    cosolvent_molecules = list()
    cosolvent_molecules.append(CosolventMolecule(name="benzene",
                                                 smiles="C1=CC=CC=C1",
                                                 resname="BEN",
                                                 concentration=0.25))
    cosolvent_molecules.append(CosolventMolecule(name="methanol",
                                                 smiles="CO",
                                                 # MET is reserved for residues
                                                 resname="MNL",
                                                 copies=58))



Building a CosolventSystem
###############################

Thanks to the flexible API CosolvKit allows the user to instantiate custom CosolventSystem classes with user prepared proteins:

.. code-block:: python

    import json
    from cosolvkit.cosolvent_system import CosolventSystem
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    with open('cosovlents.json') as fi:
        cosolvents = json.load(fi)
    with open('forcefields.json') as fi:
        forcefields = json.load(fi)
    
    cosolvent_system = CosolventSystem(cosolvents=cosolvents,
                                       forcefields=forcefields,
                                       modeller=modeller)
    cosolvent_system.build()

Building a CosolventMembraneSystem
###############################

A cosolvent system with membrane can be easily built:

.. code-block:: python

    import json
    from cosolvkit.cosolvent_system import CosolventSystem
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    with open('cosovlents.json') as fi:
        cosolvents = json.load(fi)
    with open('forcefields.json') as fi:
        forcefields = json.load(fi)
    
    cosolvent_system = CosolventMembraneSystem(cosolvents=cosolvents,
                                               forcefields=forcefields,
                                               modeller=modeller,
                                               lipid_type="POPC")

    # Or if want to pass a different type of lipids (pre-equilibrated patch needed)
    cosolvent_system = CosolventMembraneSystem(cosolvents=cosolvents,
                                               forcefields=forcefields,
                                               modeller=modeller,
                                               lipid_patch_path="path/to/the/patch")

    cosolvent_system.build()


Adding repulsive forces in case of aggregation events
###############################

Aggregation events can be common for some types of cosolvents, if in doubt, we suggest to run a simulation without custom repulsive forces and inspect the RDF profiles (please refer to the original paper for more details).  
If aggregation is observed, CosolvKit offers the possibility to add a custom repulsive force between specified residues.

.. code-block:: python

    import json
    from cosolvkit.cosolvent_system import CosolventSystem, CosolventMolecule
    from openmm.app import Modeller
    modeller = Modeller(protein_topology, protein_positions)

    cosolvent_molecules = list()
    cosolvent_molecules.append(CosolventMolecule(name="benzene",
                                                 smiles="C1=CC=CC=C1",
                                                 resname="BEN",
                                                 concentration=0.25))
    
    with open('forcefields.json') as fi:
        forcefields = json.load(fi)
    
    cosolvent_system = CosolventSystem(cosolvents=cosolvents,
                                       forcefields=forcefields,
                                       modeller=modeller)
    cosolvent_system.build()
    cosolvent_system.add_repulsive_forces(["BEN"])

    # or you can specify epsilon and sigma parameters of the LJ potential
    e = 0.05
    s = 9

    cosolvent_system.add_repulsive_forces(["BEN"], epsilon=e, sigma=s)


Use custom solvent
###############################

CosolvKit offers the possibility of using solvents different from water. In case of water the solvation is done by OpenMM, while for custom cosolvents CosolvKit exploits the same method used to place cosolvent molecules to place solvent molecules (if filling the box with solvent can be pretty slow).
This feature of CosolvKit is meant to offer flexibility for different advanced tasks.  

The solvent can be specified as SMILES string and the number of molecules requested can be specified optionally.

.. code-block:: python

    #... Previous code to create cosolvent system
    cosolvent_system.build(solvent_smiles="CO", n_solvent_molecules=350)


Saving topologies and the system
###############################

Once the cosolvent system is created and parametrized, it has to be saved for the next steps (likely MD simulation).
Depending on what MD engine was selected the format of the topology files can change.  

.. code-block:: python

    #... Previous code to create and parametrize the cosolvent system
    cosolvent_system.save_topology(topology=cosolvent_system.modeller.topology,
                                   positions=cosolvent_system.modeller.positions,
                                   system=cosolvent_system.system,
                                   # Gather the md_format from the config file
                                   simulation_format=config.md_format,
                                   forcefield=cosolvent_system.forcefield)


Run MD simulations with CosolvKit
###############################

CosolvKit offers a general and standard protocol to run MD simulations that can be used for the majority of the use cases.  
The flags `run_cosolvent_system` and `run_md` in the `Config` class take care of building the cosolvent system and using the standard MD protocol to run a simulation.

.. code-block:: python

    from cosolvkit.simulation import run_simulation

    if config.md_format.upper() != "OPENMM":
            # Change the next two lines depending on the simulation_format you chose
            topo = os.path.join(config.output, "system.prmtop")
            pos = os.path.join(config.output, "system.rst7")
            # This is for openmm
            pdb = None
            system = None
        else:
            topo = None
            pos = None
            # This is for openmm
            pdb = os.path.join(config.output, "system.pdb")
            system = os.path.join(config.output, "system.xml")
        
        if config.md_format.upper() == "OPENMM":
            print(f"Starting MD simulation from the files: {pdb}, {system}")
        else:
            print(f"Starting MD simulation from the files: {topo}, {pos}")
        
        run_simulation(
                        simulation_format = config.md_format,
                        topology = topo,
                        positions = pos,
                        pdb = pdb,
                        system = system,
                        warming_steps = 100000,
                        simulation_steps = 6250000, # 25ns
                        results_path = config.output, # This should be the name of system being simulated
                        seed=None
        )

Post processing analysis
###############################

CosolvKit offers a very basic package to analyze the results of the MD simulations.  
In particualr, Radial Distribution Functions (RDFs) of the cosolvent atoms and waters are generated with the respective autocorrelation functions.  
Furthermore, densities of the specified cosolvent molecules are depicted during the simulation and saved as a PyMol session for further analysis (check the pre-print for more examples of the use of cosolvent densities).  

.. code-block:: python

    # The whole analysis module relies on the Report class
    from cosolvkit.analysis import Report
    
    report = Report(log_file="statistics.csv",
                    traj_file="trajectory.dcd",
                    top_file="system.prmtop",
                    cosolvents_path="cosolvents.json")
    # Generate RDF and autocorrelation plots
    report.generate_report(out_path="results")

    # Generate density files
    # analysis_selection_string is a string in MDAnalysis format
    # to select specific cosolvents for the densities
    report.generate_density_maps(out_path="densities",
                                 analysis_selection_string="")

    report.generate_pymol_reports(topology="system.prmtop",
                                  trajectory="trajectory.dcd",
                                  density_files=["map_density_BEN.dx"],
                                  # It's possible to specify PyMol selection string to highlight
                                  # specific residues for that particular density
                                  selection_string="",
                                  out_path="results")                

