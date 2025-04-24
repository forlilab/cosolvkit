import os
import io
import json
import time
import sys
import argparse
from collections import defaultdict
from cosolvkit.config import Config
from cosolvkit.utils import setup_logging, fix_pdb, add_variants, MD_FORMAT_EXTENSIONS
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
        "--num_simulation_steps", type=int, default=25000000
    )
    parser.add_argument(
        "--traj_write_freq", type=int, default=25000
    )
    parser.add_argument(
        "--time_step", type=float, default=.004
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0
    )
    parser.add_argument(
        "--iteratively_adjust_copies", action="store_true", default=False
    )
 
    return parser.parse_args()

def main():

    # Parse command line arguments
    args = cmd_lineparser()
    config_file = args.config

    # Load config file
    config = Config.from_config(config_file)
    os.makedirs(config.output_dir, exist_ok=True)

    # Set up logging
    logger=setup_logging(level="INFO", filepath=f"{config.output_dir}/cosolvkit.log")
    
    start = time.time()
    if config.run_cosovlent_system:
        if (config.protein_path is not None and config.box_size is not None) or (config.protein_path is None and config.box_size is None):
            logger.error("Error! If the config file specifies a receptor, the box_size should be set to null and vice versa.")
            raise SystemExit("Error! If the config file specifies a receptor, the box_size should be set to null and vice versa.")
        
        if config.protein_path is not None:
            logger.info(f"Loading receptor file {config.protein_path}")
            try:
                with open(config.protein_path) as f:
                    pdb_string = io.StringIO(f.read())
            except FileNotFoundError:
                logger.error(f"Error! File {config.protein_path} not found.")
                raise SystemExit(f"Error! File {config.protein_path} not found.")
        
            # Check if we need to clean the protein and add variants of residues
            if config.clean_protein:
                pdbfile = None
                pdbxfile = None
                if config.protein_path.endswith(".pdb"):
                    pdbfile = pdb_string
                else:
                    pdbxfile = pdb_string
                logger.info("Cleaning protein structure")
                protein_topology, protein_positions = fix_pdb(pdbfile=pdbfile,
                                                            pdbxfile=pdbxfile, 
                                                            keep_heterogens=config.keep_heterogens)
            else:
                if not config.protein_path.endswith(".pdb"):
                    pdb = PDBxFile(pdb_string)
                else:
                    pdb = PDBFile(pdb_string)
                protein_topology, protein_positions = pdb.topology, pdb.positions
                
            # Call add_variants funtion to assing variants to the protein
            if len(config.variants.keys()) > 0:
                logger.info("Adding variants to the protein")
                variants_list = list()
                residues = list(protein_topology.residues())
                mapping = defaultdict(list)
                for r in residues:
                    mapping[r.chain.id].append(int(r.id))
                
                for chain in mapping:
                    for res_number in mapping[chain]:
                        key = f"{chain}:{res_number}"
                        if key in config.variants:
                            variants_list.append(config.variants[key])
                        else:
                            variants_list.append(None)
                protein_topology, protein_positions = add_variants(protein_topology, protein_positions, variants_list)
                
        else:
            assert config.box_size is not None, "box_size is None in the config"
            # Create empty modeller since there's nothing in the system yet
            config.box_size = config.box_size * openmmunit.angstrom
            protein_topology, protein_positions = Topology(), None

        protein_modeller = Modeller(protein_topology, protein_positions)

        # Check repulsive forces and md engine consistency        
        if (config.md_format.upper() != "OPENMM") and len(config.repulsive_residues) > 0:
            logger.warning("Custom repulsive forces will only work if the MD engine is OpenMM!")
            raise Warning("Custom repulsive forces will only work if the MD engine is OpenMM!")
    
        # Load cosolvents and forcefields dictionaries
        with open(config.cosolvents) as fi:
            cosolvents = json.load(fi)

        with open(config.forcefields) as fi:
            forcefields = json.load(fi)

        if config.membrane:
            logger.info("Building a membrane-cosolvent system")
            cosolv_system = CosolventMembraneSystem(cosolvents=cosolvents,
                                                    forcefields=forcefields,
                                                    ligands=config.ligands,
                                                    simulation_format=config.md_format,
                                                    modeller=protein_modeller,
                                                    padding=config.padding,
                                                    box_size=config.box_size,
                                                    lipid_type=config.lipid_type,
                                                    lipid_patch_path=config.lipid_patch_path)
            cosolv_system.add_membrane(cosolvent_placement=config.memb_cosolv_placement,
                                    waters_to_keep=config.waters_to_keep)
            cosolv_system.build(iteratively_adjust_copies=args.iteratively_adjust_copies)
        else:
            logger.info("Building cosolvent system")
            cosolv_system = CosolventSystem(cosolvents=cosolvents,
                                            forcefields=forcefields,
                                            ligands=config.ligands,
                                            simulation_format=config.md_format,
                                            modeller=protein_modeller,
                                            padding=config.padding,
                                            box_size=config.box_size)
            cosolv_system.build(solvent_smiles=config.solvent_smiles,
                                n_solvent_molecules=config.solvent_copies,
                                iteratively_adjust_copies=args.iteratively_adjust_copies)
            
        if len(config.repulsive_residues) > 0:
            logger.info("Adding custom repulsive forces to the system")
            cosolv_system.add_repulsive_forces(config.repulsive_residues, epsilon=config.repulsive_epsilon, sigma=config.repulsive_sigma)

        logger.info("Saving topology file")
        cosolv_system.save_topology(topology=cosolv_system.modeller.topology, 
                                    positions=cosolv_system.modeller.positions,
                                    system=cosolv_system.system,
                                    simulation_format=config.md_format,
                                    forcefield=cosolv_system.forcefield,
                                    out_path=config.output_dir)
    logger.info(f"All done! System building took {(time.time() - start)/60:.2f} min.")    

    if config.run_md:
        if config.md_format.upper() != "OPENMM":
            logger.error(f"MD format {config.md_format} is not supported for running simulations. Please use OpenMM instead.")
            raise ValueError(f"MD format {config.md_format} is not supported for running simulations. Please use OpenMM instead.")
        
        logger.info("Running MD simulation")
        start = time.time()
        pdb_fname = os.path.join(config.output_dir, "system.pdb")
        system_fname = os.path.join(config.output_dir, "system.xml")
        run_simulation(
                        pdb_fname = pdb_fname,
                        system_fname = system_fname,
                        membrane_protein = config.membrane,
                        traj_write_freq = args.traj_write_freq,
                        time_step = args.time_step,
                        temperature=args.temperature,
                        simulation_steps = args.num_simulation_steps, 
                        results_path = config.output_dir,
                        seed=None
        )
        logger.info(f"Simulation finished after {(time.time() - start)/60:.2f} min.")
    return


if __name__ == "__main__":
    sys.exit(main())