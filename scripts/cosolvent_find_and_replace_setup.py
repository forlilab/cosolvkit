import os
import json
import time
import argparse

from cosolvkit import CoSolventBox
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit
from mdtraj.reporters import DCDReporter, NetCDFReporter


def build_cosolvent_box(receptor_path: str, cosolvents: list, output_path: str) -> str:
    receptor = receptor_path.split('/')[-1].split(".")[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cosolv = CoSolventBox(cutoff=10, box='cubic')
    cosolv.add_receptor(receptor_path)
    print("Adding cosolvents...")
    for cosolvent in cosolvents:
        cosolv.add_cosolvent(**cosolvent)
    cosolv.build()
    pdb_filename = os.path.join(output_path, f"cosolv_system_{receptor}.pdb")
    cosolv.export_pdb(filename=pdb_filename)

    print("Generating tleap files")
    tleap_filename = os.path.join(output_path, "tleap.cmd")
    prmtop_filename = os.path.join(output_path, "cosolv_system.prmtop")
    inpcrd_filename = os.path.join(output_path, "cosolv_system.inpcrd")
    cosolv.prepare_system_for_amber(filename=tleap_filename,
                                    pdb_filename=pdb_filename,
                                    prmtop_filename=prmtop_filename,
                                    inpcrd_filename=inpcrd_filename,
                                    run_tleap=True)
    return prmtop_filename, inpcrd_filename

def run_simulation(out_path, prmtop_file, inpcrd_file, output_file_name, simulation_time=None):
    # results_path = os.path.join(out_path, "results")
    if simulation_time is None:
        simulation_time = 25000000

    results_path = out_path

    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)

    system = prmtop.createSystem(nonbondedMethod=PME, 
                                nonbondedCutoff=10 * openmmunit.angstrom,
                                constraints=HBonds,
                                hydrogenMass=1.5 * openmmunit.amu)
    try:
        platform = Platform.getPlatformByName("OpenCL")
        properties = {"Precision": "mixed"}
        print("Using GPU.")
    except:
        platform = Platform.getPlatformByName("CPU")
        print("Switching to CPU, no GPU available.")
        
    system.addForce(MonteCarloBarostat(1 * openmmunit.bar, 300 * openmmunit.kelvin))
    integrator = LangevinIntegrator(300 * openmmunit.kelvin,
                                    1 / openmmunit.picosecond,
                                    0.004 * openmmunit.picosecond)

    if len(properties) > 0:
        simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
    else:
        simulation = Simulation(prmtop.topology, system, integrator, platform)

    print("Setting positions for the simulation")
    simulation.context.setPositions(inpcrd.positions)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    simulation.reporters.append(NetCDFReporter(os.path.join(results_path, output_file_name + ".nc"), 250))
    simulation.reporters.append(DCDReporter(os.path.join(results_path, output_file_name + ".dcd"), 250))
    simulation.reporters.append(CheckpointReporter(os.path.join(results_path, output_file_name + ".chk"), 250))
    simulation.reporters.append(StateDataReporter(os.path.join(results_path, output_file_name + ".log"), 250, step=True, time=True,
                                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                temperature=True, volume=True, density=True, speed=True))

    #100 ns
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
    start = time.time()
    args = cmd_lineparser()

    with open(args.cosolvents) as fi:
        cosolvents = json.load(fi)

    receptor_path = args.receptor_path
    output_path = args.outpath
    print("Building cosolvent box")
    prmtop_file, inpcrd_file = build_cosolvent_box(receptor_path, cosolvents, output_path)
    print("Starting simulation")
    # 10 ps
    run_simulation(output_path, prmtop_file, inpcrd_file, "simulation", simulation_time=25000)
    print("Simulation finished! Time to analyse the results!")
    print(f"Simulation finished - simulation time: {time.time() - start}.")