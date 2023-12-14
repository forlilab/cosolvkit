import os
import json
import argparse

from cosolvkit import CoSolventBox
from cosolvkit.parametrize import parametrize_system 
from openmm.app import *
from openmm import *
from openmm.unit import *
from mdtraj.reporters import DCDReporter, NetCDFReporter


def build_cosolvent_box(receptor_path: str, cosolvents: list, output_path: str) -> str:
    receptor = receptor_path.split('/')[-1].split(".")[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cosolv = CoSolventBox(cutoff=12, box='cubic')
    cosolv.add_receptor(receptor_path)
    print("Adding cosolvents...")
    for cosolvent in cosolvents:
        cosolv.add_cosolvent(**cosolvent)
    cosolv.build()
    cosolv.export_pdb(filename=f"cosolv_system_{receptor}.pdb")
    parametrize_system(f"protein.pdb", "cosolvents.json", "forcefields.json", "Amber", output_path)
    # parametrize_system(f"cosolv_system_{receptor}.pdb", "cosolvents.json", "forcefields.json", "Amber", output_path)
    return

def run_simulation(out_path, prmtop_file, inpcrd_file, output_file_name, simulation_time=None, simulation_engine="Amber"):
    # results_path = os.path.join(out_path, "results")
    if simulation_time is None:
        simulation_time = 25000000

    results_path = out_path
    if simulation_engine == "Amber":
        topology = AmberPrmtopFile(prmtop_file)
        inpcrd = AmberInpcrdFile(inpcrd_file)

    system = topology.createSystem(nonbondedMethod=PME, 
                                nonbondedCutoff=10 * angstrom,
                                constraints=HBonds,
                                hydrogenMass=1.5 * amu,
                                rigidWater=False)
    try:
        platform = Platform.getPlatformByName("OpenCL")
        properties = {"Precision": "mixed"}
        print("Using GPU.")
    except: 
        properties = {}
        platform = Platform.getPlatformByName("CPU")
        print("Switching to CPU, no GPU available.")
        
    integrator = LangevinMiddleIntegrator(300 * kelvin,
                                            1 / picosecond,
                                            1 * femtoseconds)

    if len(properties) > 0:
        simulation = Simulation(topology.topology, system, integrator, platform, properties)
    else:
        simulation = Simulation(topology.topology, system, integrator, platform)

    print("Setting positions for the simulation")
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(300 * kelvin)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    # MD simulations - equilibration (1ns)
    print("Equilibrating system")
    simulation.step(250000)

    system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))
    simulation.context.reinitialize(preserveState=True)
    # cosolvkit.utils.update_harmonic_restraints(simulation, 0.1)

    simulation.reporters.append(NetCDFReporter(os.path.join(results_path, output_file_name + ".nc"), 250))
    simulation.reporters.append(DCDReporter(os.path.join(results_path, output_file_name + ".dcd"), 250))
    simulation.reporters.append(CheckpointReporter(os.path.join(results_path, output_file_name + ".chk"), 250))
    simulation.reporters.append(StateDataReporter(os.path.join(results_path, output_file_name + ".log"), 250, step=True, time=True,
                                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                temperature=True, volume=True, density=True, speed=True))

    #100 ns = 25000000
    print("Running simulation")
    simulation.step(simulation_time)
    return


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit and MD simulation afterwards.")
    parser.add_argument('-c', '--cosolvents_list', dest='cosolvents', required=True,
                        action='store', help='path to the json file defining the cosolvents to add')
    parser.add_argument('-r', '--receptor', dest='receptor_path', required=True,
                        action='store', help='path to the receptor file')
    parser.add_argument('-o', '--output_path', dest='outpath', required=True,
                        action='store', help='path where to store output of the MD simulation')
    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_lineparser()

    with open(args.cosolvents) as fi:
        cosolvents = json.load(fi)

    receptor_path = args.receptor_path
    output_path = args.outpath
    print("Building cosolvent box")
    build_cosolvent_box(receptor_path, cosolvents, output_path)
    # print("Starting simulation")
    # run_simulation(output_path, f"{output_path}/system.prmtop", f"{output_path}/system.inpcrd", "simulation", simulation_time=None)
    # print("Simulation finished! Time to analyse the results!")