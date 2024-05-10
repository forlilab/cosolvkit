import os
import io
import json
import time
import argparse
from collections import defaultdict
from cosolvkit.config import Config
from cosolvkit.utils import fix_pdb, add_variants
from cosolvkit.cosolvent_system import CosolventSystem, CosolventMembraneSystem
from cosolvkit.simulation import run_simulation
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit and MD simulation afterwards.",
                                     epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2023 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""")
    
    parser.add_argument('-c', '--config', dest='config', required=True,
                        action='store', help='path to the json config file')
    parser.add_argument(
        "--num_simulation_steps", type=int, default=6250000
    )
    parser.add_argument(
        "--traj_write_freq", type=int, default=25000
    )
    parser.add_argument(
        "--time_step", type=float, default=.004
    )
    parser.add_argument(
        "--iteratively_adjust_copies", action="store_true", default=False
    )
 
    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_lineparser()
    config_file = args.config
    config = Config.from_config(config_file)
    # Start setting up the pipeline
    os.makedirs(config.output, exist_ok=True)
    
    if config.run_cosovlent_system:
        if (config.receptor and config.radius is not None) or (not config.receptor and config.radius is None):
            raise SystemExit("Error! If the config file specifies a receptor, the radius should be set to null and vice versa.")
        
        # Check if need to clean the protein and add variants of reisudes
        if config.receptor:
            print("Protein present")
            with open(config.protein_path) as f:
                pdb_string = f.read()
            pdb_string = io.StringIO(pdb_string)
            if config.clean_protein:
                pdbfile = None
                pdbxfile = None
                if config.protein_path.endswith(".pdb"):
                    pdbfile = pdb_string
                else:
                    pdbxfile = pdb_string
                protein_topology, protein_positions = fix_pdb(pdbfile=pdbfile,
                                                            pdbxfile=pdbxfile, 
                                                            keep_heterogens=config.keep_heterogens, )
            else:
                if not config.protein_path.endswith(".pdb"):
                    pdb = PDBxFile(pdb_string)
                else:
                    pdb = PDBFile(pdb_string)
                protein_topology, protein_positions = pdb.topology, pdb.positions
            
            # Call add_variants fucntion
            if len(config.variants_d.keys()) > 0:
                variants = list()
                residues = list(protein_topology.residues())
                mapping = defaultdict(list)
                for r in residues:
                    mapping[r.chain.id].append(int(r.id))
                
                for chain in mapping:
                    for res_number in mapping[chain]:
                        key = f"{chain}:{res_number}"
                        if key in config.variants_d:
                            variants.append(config.variants_d[key])
                        else:
                            variants.append(None)
                protein_topology, protein_positions = add_variants(protein_topology, protein_positions, variants)
                
        else:
            assert config.radius is not None, "radius is None in the config"
            # Create empty modeller since there's nothing in the system yet
            config.radius = config.radius * openmmunit.angstrom
            protein_topology, protein_positions = Topology(), None

        protein_modeller = Modeller(protein_topology, protein_positions)

        # Load cosolvents and forcefields dictionaries
        with open(config.cosolvents) as fi:
            cosolvents = json.load(fi)

        with open(config.forcefields) as fi:
            forcefields = json.load(fi)

        print("Building cosolvent system")
        if config.membrane:
            cosolv_system = CosolventMembraneSystem(cosolvents=cosolvents,
                                                    forcefields=forcefields,
                                                    simulation_format=config.md_format,
                                                    modeller=protein_modeller,
                                                    lipid_type=config.lipid_type,
                                                    lipid_patch_path=config.lipid_patch_path)
            cosolv_system.add_membrane(cosolvent_placement=config.cosolvent_placement,
                                    waters_to_keep=config.waters_to_keep)
            cosolv_system.build(iteratively_adjust_copies=args.iteratively_adjust_copies)
        else:
            cosolv_system = CosolventSystem(cosolvents=cosolvents,
                                            forcefields=forcefields,
                                            simulation_format=config.md_format,
                                            modeller=protein_modeller,
                                            radius=config.radius)
            cosolv_system.build(solvent_smiles=config.solvent_smiles,
                                n_solvent_molecules=config.solvent_copies,
                                iteratively_adjust_copies=args.iteratively_adjust_copies)
            
        if config.add_repulsive:
            cosolv_system.add_repulsive_forces(config.repulsive_residues, epsilon=config.epsilon, sigma=config.sigma)

        print("Saving topology file")
        cosolv_system.save_topology(topology=cosolv_system.modeller.topology, 
                                    positions=cosolv_system.modeller.positions,
                                    system=cosolv_system.system,
                                    simulation_format=config.md_format,
                                    forcefield=cosolv_system.forcefield,
                                    out_path=config.output)
    
    if config.run_md:
        print("Running MD simulation")
        start = time.time()
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
                        traj_write_freq = args.traj_write_freq,
                        time_step = args.time_step,
                        warming_steps = 100000,
                        simulation_steps = args.num_simulation_steps, 
                        results_path = config.output, # This should be the name of system being simulated
                        seed=None
        )
        print(f"Simulation finished after {(time.time() - start)/60:.2f} min.")
