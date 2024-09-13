import os
from sys import stdout
import openmm.app as app
import openmm
import openmm.unit as openmmunit
from mdtraj.reporters import NetCDFReporter, DCDReporter

def run_simulation( simulation_format: str = 'OPENMM',
                    results_path: str = "output",
                    topology: str=None, 
                    positions: str=None,
                    pdb: str = 'output/system.pdb',
                    system: str = 'output/system.xml',
                    membrane_protein: bool = False, 
                    traj_write_freq: int = 25000,
                    time_step: float = 0.004, 
                    warming_steps: int = 100000,
                    simulation_steps: int = 25000000,
                    seed: int = None
                    ):
    """_summary_

    :param simulation_format: determines what MD engine will be used, defaults to 'OPENMM'. Available engines are: [AMBER, GROMACS, CHARMM, OPENMM]
    :type simulation_format: str, optional
    :param results_path: path to where to save the results, defaults to "output"
    :type results_path: str, optional
    :param topology: path to the topology file if using simulation_format different from OPENMM, defaults to None
    :type topology: str, optional
    :param positions: path to the positions file if using simulation_format different from OPENMM, defaults to None
    :type positions: str, optional
    :param pdb: path to the pdb file if using simulation_format OPENMM, defaults to 'output/system.pdb'
    :type pdb: str, optional
    :param system: path to the system.xml file if using simulation_format OPENMM, defaults to 'output/system.xml'
    :type system: str, optional
    :param membrane_protein: True if using a membrane in the system, False otherwise
    :type membrane_protein: bool, optional
    :param warming_steps: number of warming steps, defaults to 100000
    :type warming_steps: int, optional
    :param simulation_steps: number of simulation steps, defaults to 25000000
    :type simulation_steps: int, optional
    :param seed: random seed for reproducibility, defaults to None
    :type seed: int, optional
    :raises ValueError: different checks are performed and expections are raised if some of the fail.
    """

    # Temperature annealing
    Tstart = 5
    Tend = 300
    Tstep = 5
    
    openmm_flag = simulation_format == "OPENMM"
    total_steps = warming_steps + simulation_steps

    if simulation_format not in ['OPENMM', 'AMBER', 'GROMACS', 'CHARMM']:
        raise ValueError(f"Unknown simulation_format {simulation_format}. It must be one of 'OPENMM', 'AMBER', 'GROMACS', or 'CHARMM'.")
    
    # if not openmm_flag:
    #     assert topology is not None and positions is not None, "If the simulation format specified is not OpenMM be sure to pass both topology and positions files"
    #     if simulation_format == "AMBER":
    #         positions = app.AmberInpcrdFile(positions)
    #         topology = app.AmberPrmtopFile(topology, periodicBoxVectors=positions.boxVectors)
    #     elif simulation_format == "GROMACS":
    #         positions = app.GromacsGroFile(positions)
    #         topology = app.GromacsTopFile(topology, periodicBoxVectors=positions.getPeriodicBoxVectors())
    #     elif simulation_format == "CHARMM":
    #         topology = app.CharmmPsfFile(topology)
    #         positions = app.CharmmCrdFile(positions)
    #     system = topology.createSystem(nonbondedMethod=app.PME,
    #                                    nonbondedCutoff=10*openmmunit.angstrom,
    #                                    switchDistance=9*openmmunit.angstrom,
    #                                    constraints=app.HBonds,
    #                                    removeCMMotion=True,
    #                                    hydrogenMass=3.0*openmmunit.amu)
    # else:
    assert pdb is not None and system is not None, "If the simulation format specified is OpenMM be sure to pass both pdb file and system.xml"
    pdb = app.PDBFile(f'{results_path}/system.pdb')
    topology = pdb.topology
    positions = pdb.positions
    system = openmm.XmlSerializer.deserialize(open(f'{results_path}/system.xml').read())

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
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
                                            reportInterval=traj_write_freq, enforcePeriodicBox=True))

    
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
    integrator.setStepSize(time_step * openmmunit.picoseconds)

    print(f'Adding a Montecarlo Barostat to the system')
    if membrane_protein:
        barostat = openmm.MonteCarloMembraneBarostat( 1.0 * openmmunit.bar,  # Pressure in bar 
                                                     0.0 * openmmunit.nanometers * openmmunit.bar,  # surface tension 
                                                     Tend * openmmunit.kelvin,  # Temperature in Kelvin 
                                                     openmm.MonteCarloMembraneBarostat.XYIsotropic,  # XY isotropic scaling 
                                                     openmm.MonteCarloMembraneBarostat.ZFree,  # Z dimension is free 
                                                     15  # Number of Monte Carlo steps 
                                                     ) 
        system.addForce(barostat)
    else:
        system.addForce(openmm.MonteCarloBarostat(1 * openmmunit.bar, Tend * openmmunit.kelvin))
    simulation.context.reinitialize(preserveState=True)

    print(f"Running simulation in NPT ensemble for {simulation_steps*0.004/1000} ns")
    simulation.step(simulation_steps) #25000000 = 100ns

    return
