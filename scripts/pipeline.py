import os
import time
import argparse

from cosolvkit.openmm_cosolventbox import CosolventSystem
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit
from mdtraj.reporters import DCDReporter, NetCDFReporter


def build_cosolvent_box(receptor_path: str, cosolvents: str, forcefields: str, simulation_engine: str, output_path: str) -> CosolventSystem:
    receptor = receptor_path.split('/')[-1].split(".")[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cosolv = CosolventSystem(cosolvents, forcefields, simulation_engine, receptor_path, padding=50*openmmunit.angstrom)
    return cosolv

def run_simulation(out_path, cosolv_system, simulation_time=None, simulation_engine="Amber", output_filename="simulation"):
    # results_path = os.path.join(out_path, "results")
    if simulation_time is None:
        simulation_time = 25000000

    results_path = out_path

    try:
        platform = Platform.getPlatformByName("OpenCL")
        properties = {"Precision": "mixed"}
        print("Using GPU.")
    except: 
        properties = {}
        platform = Platform.getPlatformByName("CPU")
        print("Switching to CPU, no GPU available.")

    cosolv_system.system.addForce(MonteCarloBarostat(1 * openmmunit.bar, 300 * openmmunit.kelvin))
    integrator = LangevinMiddleIntegrator(300 * openmmunit.kelvin,
                                            1 / openmmunit.picosecond,
                                            0.004 * openmmunit.picosecond)

    if len(properties) > 0:
        simulation = Simulation(cosolv_system.modeller.topology, cosolv_system.system, integrator, platform, properties)
    else:
        simulation = Simulation(cosolv_system.modeller.topology, cosolv_system.system, integrator, platform)

    print("Setting positions for the simulation")
    simulation.context.setPositions(cosolv_system.modeller.positions)
    # simulation.context.setVelocitiesToTemperature(300 * openmmunit.kelvin)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    # MD simulations - equilibration (1ns)
    # print("Equilibrating system")
    # simulation.step(25000)

    # cosolv_system.system.addForce(MonteCarloBarostat(1 * openmmunit.bar, 300 * openmmunit.kelvin))
    # simulation.context.reinitialize(preserveState=True)
    # cosolvkit.utils.update_harmonic_restraints(simulation, 0.1)

    simulation.reporters.append(NetCDFReporter(os.path.join(results_path, output_filename + ".nc"), 25000))
    simulation.reporters.append(DCDReporter(os.path.join(results_path, output_filename + ".dcd"), 25000))
    simulation.reporters.append(CheckpointReporter(os.path.join(results_path, output_filename + ".chk"), 250))
    simulation.reporters.append(StateDataReporter(os.path.join(results_path, output_filename + ".log"), 250, step=True, time=True,
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
    parser.add_argument('-f', '--forcefields', dest='ffs', required=True,
                        action='store', help='path to the json file defining the forcefields to add')
    parser.add_argument('-e', '--engine', dest='engine', required=True,
                        action='store', help='Simulation engine [AMBER, GROMACS, CHARMM]')
    parser.add_argument('-r', '--receptor', dest='receptor_path', required=True,
                        action='store', help='path to the receptor file')
    parser.add_argument('-o', '--output_path', dest='outpath', required=True,
                        action='store', help='path where to store output of the MD simulation')
    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_lineparser()
    cosolvents = args.cosolvents
    forcefields = args.ffs
    simulation_engine = args.engine
    receptor_path = args.receptor_path
    output_path = args.outpath
    print("Building cosolvent box")
    cosolv_system = build_cosolvent_box(receptor_path, cosolvents, forcefields, simulation_engine, output_path)
    # print("Saving PDB file")
    # cosolv_system.save_pdb(cosolv_system.modeller.topology, 
    #                        cosolv_system.modeller.positions,
    #                        f"{output_path}/system.pdb")
    # Good habit to save the topology files
    # print("Saving topology file")
    # cosolv_system.save_topology(cosolv_system.modeller.topology, 
    #                             cosolv_system.modeller.positions,
    #                             cosolv_system.system,
    #                             simulation_engine,
    #                             output_path)
    # If you want to save the system as well
    # cosolv_system.save_system(output_path, cosolv_system.system)
    
    # print("Starting simulation")
    # start = time.time()
    # run_simulation(output_path, cosolv_system, simulation_time=250000, simulation_engine=simulation_engine)
    # print(f"Simulation finished - simulation time: {time.time() - start}.")