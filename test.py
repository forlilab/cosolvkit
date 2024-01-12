import json
import random
import math
import numpy as np
import quaternion as quat
from itertools import product
from scipy.stats import qmc
from scipy import spatial
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from collections import defaultdict
from cosolvkit.cosolvent_system import CosolventSystem
import openmm.unit as openmmunit
from cosolvkit.cosolvent_system import CoSolvent

def check_coordinates_to_add(new_coords, cosolvent_kdtree, protein_kdtree):
    protein_radius = 3.5*openmmunit.angstrom
    cosolv_radius = 3.5*openmmunit.angstrom
    # radius = radius.value_in_unit(openmmunit.nanometer)
    if protein_kdtree is not None and not any(protein_kdtree.query_ball_point(new_coords, protein_radius.value_in_unit(openmmunit.angstrom))):
        if cosolvent_kdtree is not None:
            if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolv_radius.value_in_unit(openmmunit.nanometer))):
                return True
            else: return False
        else:
            return True
    elif protein_kdtree is None and cosolvent_kdtree is not None:
        if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolv_radius.value_in_unit(openmmunit.nanometer))):
            return True
        else: 
            return False
    else:
        return False

def accept_reject(xyz, halton, kdtree, used, lowerBound, upperBound, protein_kdtree):
    trial = 0
    accepted = False
    coords_to_return = 0
    moves = local_search()
    valid_ids = np.array(range(0, len(halton)))
    while not accepted:
        halton_idx = np.random.choice(valid_ids[~np.isin(valid_ids, np.array(used))])
        rotated_xyz = generate_rotation(xyz)
        cosolv_xyz = rotated_xyz + halton[halton_idx]
        if check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
            used.append(halton_idx)
            accepted = True
            coords_to_return = cosolv_xyz
        else:
            trial += 1
            # print("Trying local moves")
            for move in moves:
                # print("Trying local search")
                rotated_xyz = generate_rotation(xyz)
                cosolv_xyz = rotated_xyz + halton_idx + move
                if is_in_box(cosolv_xyz, lowerBound, upperBound):
                    if check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                        accepted = True
                        used.append(halton_idx)
                        coords_to_return = cosolv_xyz
                        break
                #     else:
                #         print("Too close to the receptro")
                # else:
                #     print("Points out of the box!")
                trial += 1
    return coords_to_return, used

def is_in_box(xyzs, lowerBound, upperBound):
    """Is in the box or not?
    """
    xyzs = np.atleast_2d(xyzs)
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    xmin, xmax = lowerBound[0], upperBound[0][0]
    ymin, ymax = lowerBound[1], upperBound[1][1]
    zmin, zmax = lowerBound[2], upperBound[2][2]

    x_in = np.logical_and(xmin <= x, x <= xmax)
    y_in = np.logical_and(ymin <= y, y <= ymax)
    z_in = np.logical_and(zmin <= z, z <= zmax)
    all_in = np.all((x_in, y_in, z_in), axis=0)

    return np.all(all_in)

def local_search():
    step = 1
    moves = filter(lambda point: not all(axis ==0 for axis in point), list(product([-step, 0, step], repeat=3)))
    return moves

def generate_rotation(coords):
    """
        Rotate a list of 3D [x,y,z] vectors about corresponding random uniformly
        distributed quaternion [w, x, y, z]
    
        Parameters
        ----------
        coords : numpy.ndarray with shape [n,3]
            list of [x,y,z] cartesian vector coordinates
    """
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    qrot = np.array([np.cos(t2) * r2,
                     np.sin(t1) * r1,
                     np.cos(t1) * r1,
                     np.sin(t2) * r2])
    rotation = spatial.transform.Rotation.from_quat(qrot)
    return rotation.apply(coords)

def add_cosolvents(cosolvents, vectors, lowerBound, upperBound, receptor_positions):
    import sys
    protein_radius = 0.1*openmmunit.angstrom
    prot_kdtree = None
    placed_atoms_positions = []
    if receptor_positions is not None and len(receptor_positions) > 0:
        prot_kdtree = spatial.cKDTree(receptor_positions)
    cosolv_xyzs = defaultdict(list)
    sampler = qmc.Halton(d=3)
    points = sampler.random(10000)
    points= qmc.scale(points, [lowerBound[0], lowerBound[0], lowerBound[0]], [upperBound[0], upperBound[1], upperBound[2]])
    used_halton_ids = set()
    # for protein_radius in [0.1*openmmunit.angstrom, 1.*openmmunit.angstrom, 2.5*openmmunit.angstrom, 3.5*openmmunit.angstrom]:
    if prot_kdtree is not None:
        banned_ids = prot_kdtree.query_ball_point(points, protein_radius.value_in_unit(openmmunit.nanometer))
    for x in banned_ids:
        if len(x) > 0:
            for y in x:
                used_halton_ids.add(y)
    print(f"Pruned {len(used_halton_ids)} with cutoff {protein_radius.value_in_unit(openmmunit.nanometer)}")
    used_halton_ids = list(used_halton_ids)
    for cosolvent in cosolvents:
        # centroid, radius = get_centroid(cosolvent.mol, cosolvents[cosolvent])
        c_xyz = cosolvents[cosolvent]
        for replicate in range(cosolvent.copies):
            print(f"Placing {cosolvent.name} copy {replicate}")
            counter = replicate
            if len(placed_atoms_positions) < 1:
                xyz = points[counter]
                cosolv_xyz = c_xyz + xyz
                [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                cosolv_xyzs[cosolvent].append(cosolv_xyz)
                used_halton_ids.append(counter)
                kdtree = spatial.cKDTree(placed_atoms_positions)
            else:
                kdtree = spatial.cKDTree(placed_atoms_positions)
                cosolv_xyz, used_halton_ids = accept_reject(c_xyz, points, kdtree, used_halton_ids, lowerBound, vectors, prot_kdtree)
                if isinstance(cosolv_xyz, int):
                    print("Could not place the cosolvent molecule!")
                else:
                    # cosolv_xyz = cosolv_xyz * openmmunit.angstrom
                    # cosolv_xyzs[cosolvent].append(cosolv_xyz.value_in_unit(openmmunit.nanometer))
                    cosolv_xyzs[cosolvent].append(cosolv_xyz)
                    [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
        print("Done!")
    print("Added cosolvents:")
    for cosolvent in cosolv_xyzs:
        print(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
    return cosolv_xyzs

def create_and_parametrize_system(receptor=None):
    if receptor is not None:
        radius = None
    else:
        radius=30*openmmunit.angstrom
    cosolv = CosolventSystem("cosolvents.json", "forcefields.json", "amber", receptor=receptor, radius=radius)
    vectors, lowerBound, upperBound = cosolv.get_box_origin_and_size(cosolv.modeller.positions, cosolv._padding, radius, system_unit=openmmunit.nanometer)
    print("Adding cosolvents")
    cosolv_xyzs = add_cosolvents(cosolv.cosolvents, vectors, lowerBound, upperBound, cosolv.modeller.positions.value_in_unit(openmmunit.nanometer))
    print("Done adding cosolvents, setting up new topology!")
    cosolv.modeller = cosolv._setup_new_topology(cosolv_xyzs,
                                             cosolv.modeller.topology,
                                             cosolv.modeller.positions)
    print("New topology set up!")
    cosolv.modeller.addSolvent(cosolv.forcefield)
    cosolv.save_pdb(cosolv.modeller.topology, cosolv.modeller.positions, "results/halton_random_scipy_quat_bigger_box.pdb")
    return

if __name__ == "__main__":
    create_and_parametrize_system("data/example/protein.pdb")