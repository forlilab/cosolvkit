import os
from sys import stdout
import openmm.app as app
import openmm
import openmm.unit as openmmunit
from mdtraj.reporters import NetCDFReporter, DCDReporter

def run_simulation( simulation_format: str = 'OPENMM',
                    results_path: str = "output",
                    topology=None, 
                    positions=None,
                    pdb: str = 'output/system.pdb',
                    system: str = 'output/system.xml', 
                    warming_steps: int = 100000,
                    simulation_steps: int = 25000000,
                    seed: int = None
                    ):

    # Temperature annealing
    Tstart = 5
    Tend = 300
    Tstep = 5
    
    openmm_flag = simulation_format == "OPENMM"
    total_steps = warming_steps + simulation_steps

    if simulation_format.upper() not in ['OPENMM', 'AMBER', 'GROMACS', 'CHARMM']:
        raise ValueError(f"Unknown simulation_format {simulation_format}. It must be one of 'OPENMM', 'AMBER', 'GROMACS', or 'CHARMM'.")
    
    if not openmm_flag:
        assert topology is not None and positions is not None, "If the simulation format specified is not OpenMM be sure to pass both topology and positions files"
        if simulation_format.upper() == "AMBER":
            positions = app.AmberInpcrdFile(positions)
            topology = app.AmberPrmtopFile(topology, periodicBoxVectors=positions.boxVectors)
        elif simulation_format.upper() == "GROMACS":
            positions = app.GromacsGroFile(positions)
            topology = app.GromacsTopFile(topology, periodicBoxVectors=positions.getPeriodicBoxVectors())
        elif simulation_format.upper() == "CHARMM":
            topology = app.CharmmPsfFile(topology)
            positions = app.CharmmCrdFile(positions)
        system = topology.createSystem(nonbondedMethod=app.PME,
                                       nonbondedCutoff=10*openmmunit.angstrom,
                                       constraints=app.HBonds,
                                       hydrogenMass=1.5*openmmunit.amu)
    else:
        assert pdb is not None and system is not None, "If the simulation format specified is OpenMM be sure to pass both pdb file and system.xml"
        pdb = app.PDBFile(f'{results_path}/system.pdb')
        topology = pdb.topology
        positions = pdb.positions
        system = openmm.XmlSerializer.deserialize(open(f'{results_path}/system.xml').read())

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("GPU")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = openmm.Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")

    integrator = openmm.LangevinMiddleIntegrator(Tstart * openmmunit.kelvin,
                                          1 / openmmunit.picosecond,
                                          0.001 * openmmunit.picosecond)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    
    if not openmm_flag:
        simulation = app.Simulation(topology.topology, system, integrator, platform)
    else:
        simulation = app.Simulation(topology, system, integrator, platform)
        
    print('Adding reporters to the simulation')
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(os.path.join(results_path, "statistics.csv"), 25000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps))
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(stdout, 25000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
    
    #every 0.1ns
    simulation.reporters.append(app.DCDReporter(os.path.join(results_path, "trajectory.dcd"),
                                            reportInterval=25000, enforcePeriodicBox=None))

    
    #every 1ns
    simulation.reporters.append(app.CheckpointReporter(os.path.join(results_path,"simualtion.chk"), 250000)) 

    print("Setting positions for the simulation")
    if not openmm_flag:
        simulation.context.setPositions(positions.positions.value_in_unit(openmmunit.nanometer))
    else:
        simulation.context.setPositions(positions)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    print(f'Heating system in NVT ensemble for {warming_steps*0.001/1000} ns')
    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)
    
    # Warm up the system gradually
    for i in range(nT):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        print(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    # Increase the timestep for production simulations
    integrator.setStepSize(0.004 * openmmunit.picoseconds)

    print(f'Adding a Montecarlo Barostat to the system')
    system.addForce(openmm.MonteCarloBarostat(1 * openmmunit.bar, Tend * openmmunit.kelvin))
    simulation.context.reinitialize(preserveState=True)

    print(f"Running simulation in NPT ensemble for {simulation_steps*0.004/1000} ns")
    simulation.step(simulation_steps) #25000000 = 100ns

    return