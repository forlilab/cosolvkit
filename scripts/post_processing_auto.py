import os

base_path = "/mnt/bigdisk1/validation_cosolvkit/results"

for result in os.listdir(base_path):
    result_path = os.path.join(base_path, result)
    if "charged_" in result:
        cosolvent_file = "/mnt/bigdisk1/validation_cosolvkit/cosolvents_charged.json"
    elif "non_polar_" in result:
        cosolvent_file = "/mnt/bigdisk1/validation_cosolvkit/cosolvents_non_polar.json"
    else:
        cosolvent_file = "/mnt/bigdisk1/validation_cosolvkit/cosolvents_polar.json"

    if "_far" in result:
        top_file = os.path.join(result_path, "cosolv_system.prmtop")
    else:
        top_file = os.path.join(result_path, "system.pdb")
    
    cmd = f"python ~/phd/cosolvkit/scripts/post_simulation_processing.py -l {os.path.join(result_path, 'simulation.log')} -traj {os.path.join(result_path, 'simulation.dcd')} -top {top_file} -o {os.path.join(result_path, 'analysis_high')} -c {cosolvent_file}"
    print(f"running:\n\t{cmd}")
    os.system(cmd)
