import os
import time
import argparse
from sys import stdout
from cosolvkit.cosolvent_system import CosolventSystem, CosolventMembraneSystem
from cosolvkit.simulation import run_simulation
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit


def build_cosolvent_box(receptor_path: str, cosolvents: str, forcefields: str, simulation_format: str, results_path: str, radius: float) -> CosolventSystem:
    os.makedirs(results_path, exist_ok=True)
    
    if radius is not None:
        radius = radius * openmmunit.angstrom
        
    # If starting from PDB file path
    # cosolv = CosolventSystem.from_filename(cosolvents, forcefields, simulation_format, receptor_path, clean_protein=False)

    # If starting from a pdb string or without receptor
    # cosolv = CosolventSystem(cosolvents, forcefields, simulation_format, receptor_path, radius=radius)
    # cosolv.build()
    
    # Membranes
    cosolv = CosolventMembraneSystem.from_filename(cosolvents, 
                                                   forcefields, 
                                                   simulation_format, 
                                                   receptor_path, 
                                                   clean_protein=True, 
                                                   lipid_type="POPC")
    cosolv.add_membrane_and_cosolvents(cosolvent_placement=0, neutralize=True, waters_to_keep=[])
    cosolv.build(neutralize=True)
    return cosolv

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit and MD simulation afterwards.")
    parser.add_argument('-c', '--cosolvents_list', dest='cosolvents', required=True,
                        action='store', help='path to the json file defining the cosolvents to add')
    parser.add_argument('-f', '--forcefields', dest='ffs', required=True,
                        action='store', help='path to the json file defining the forcefields to add')
    parser.add_argument('-mdout', '--mdoutputformat', dest='output_format', required=True,
                        action='store', help='MD output formats <AMBER [prmtop, inpcrd], GROMACS [top, gro], CHARMM [psf, crd], OPENMM [xml]>')
    parser.add_argument('-o', '--results_path', dest='outpath', required=True,
                        action='store', help='path where to store output of the MD simulation')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--receptor', dest='receptor_path',
                        action='store', help='path to the receptor file')
    group.add_argument('-r', '--radius', dest='radius', action='store',
                       help='radius (in Angstrom) to build the box if receptor not specified.')

    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_lineparser()
    cosolvents = args.cosolvents
    forcefields = args.ffs
    simulation_format = args.output_format
    results_path = args.outpath

    if args.receptor_path:
        receptor_path = args.receptor_path
    else:
        receptor_path = None
    if args.radius:
        radius = float(args.radius)
    else:
        radius = None

    print("Building cosolvent box")
    cosolv_system = build_cosolvent_box(receptor_path, cosolvents, forcefields, simulation_format, results_path, radius)

    print("Saving topology file")
    cosolv_system.save_topology(cosolv_system.modeller.topology, 
                                cosolv_system.modeller.positions,
                                cosolv_system.system,
                                simulation_format,
                                cosolv_system.forcefield,
                                results_path)
    
    print("Running MD simulation")
    start = time.time()
    # topology = os.path.join(results_path, "system.prmtop"),
    # positions = os.path.join(results_path, "system.inpcrd")
    # run_simulation(
    #                 simulation_format = simulation_format,
    #                 topology = os.path.join(results_path, "system.prmtop"),
    #                 positions = os.path.join(results_path, "system.inpcrd"),
    #                 pdb = None,
    #                 system = None,
    #                 warming_steps = 100000,
    #                 simulation_steps = 6250000, # 25ns
    #                 results_path = results_path, # This should be the name of system being simulated
    #                 seed=None
    # )

    print(f"Simulation finished after {(time.time() - start)/60:.2f} min.")
