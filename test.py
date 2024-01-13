import os
import math
import numpy as np
import quaternion as quat
from itertools import product
from scipy.stats import qmc
from scipy import spatial
from collections import defaultdict
from cosolvkit.cosolvent_system import CosolventSystem
import openmm.unit as openmmunit
from cosolvkit.cosolvent_system import CoSolvent

def check_coordinates_to_add(new_coords, cosolvent_kdtree, protein_kdtree):
    protein_radius = 3.5*openmmunit.angstrom
    cosolv_radius = 2.5*openmmunit.angstrom
    # radius = radius.value_in_unit(openmmunit.nanometer)
    if protein_kdtree is not None and not any(protein_kdtree.query_ball_point(new_coords, protein_radius.value_in_unit(openmmunit.nanometer))):
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
    while not accepted and trial < 1000000:
        halton_idx = np.random.choice(valid_ids[~np.isin(valid_ids, np.array(used))])
        rotated_xyz = generate_rotation(xyz)
        cosolv_xyz = rotated_xyz + halton[halton_idx]
        if check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
            used.append(halton_idx)
            accepted = True
            coords_to_return = cosolv_xyz
        else:
            trial += 1
            for move in moves:
                rotated_xyz = generate_rotation(xyz)
                cosolv_xyz = rotated_xyz + halton_idx + move
                if is_in_box(cosolv_xyz, lowerBound, upperBound):
                    if check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                        accepted = True
                        used.append(halton_idx)
                        coords_to_return = cosolv_xyz
                        break
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
    protein_radius = 3.5*openmmunit.angstrom
    prot_kdtree = None
    placed_atoms_positions = []
    if receptor_positions is not None and len(receptor_positions) > 0:
        prot_kdtree = spatial.cKDTree(receptor_positions)
    cosolv_xyzs = defaultdict(list)
    sampler = qmc.Halton(d=3)
    points = sampler.random(1000000)
    points= qmc.scale(points, [lowerBound[0], lowerBound[0], lowerBound[0]], [upperBound[0], upperBound[1], upperBound[2]])
    used_halton_ids = set()
    if prot_kdtree is not None:
        banned_ids = prot_kdtree.query_ball_point(points, protein_radius.value_in_unit(openmmunit.nanometer))
    for x in banned_ids:
        if len(x) > 0:
            for y in x:
                used_halton_ids.add(y)
    print(f"Pruned {len(used_halton_ids)} with cutoff {protein_radius.value_in_unit(openmmunit.nanometer)}")
    used_halton_ids = list(used_halton_ids)
    for cosolvent in cosolvents:
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
                    cosolv_xyzs[cosolvent].append(cosolv_xyz)
                    [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
        print("Done!")
    print("Added cosolvents:")
    for cosolvent in cosolv_xyzs:
        print(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
    return cosolv_xyzs

def calculate_receptor_volume(receptor_positions):
    """Computes volume occupied by the receptor in nm**3"""
    padding = 3.5*openmmunit.angstrom
    offset = 1.5*openmmunit.angstrom
    mesh_step = 0.3*openmmunit.angstrom
    padding = padding.value_in_unit(openmmunit.nanometer)
    offset = offset.value_in_unit(openmmunit.nanometer)
    mesh_step = mesh_step.value_in_unit(openmmunit.nanometer)
    if isinstance(receptor_positions, openmmunit.Quantity):
        receptor_positions = receptor_positions.value_in_unit(openmmunit.nanometer)
    minRange = np.array([min((pos[i] for pos in receptor_positions)) for i in range(3)])
    maxRange = np.array([max((pos[i] for pos in receptor_positions)) for i in range(3)])
    x = np.arange(minRange[0]-padding, maxRange[0]+padding, mesh_step)
    y = np.arange(minRange[1]-padding, maxRange[1]+padding, mesh_step)
    z = np.arange(minRange[2]-padding, maxRange[2]+padding, mesh_step)
    X, Y, Z = np.meshgrid(x, y, z)
    center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    kdtree = spatial.cKDTree(center_xyzs)
    query = kdtree.query_ball_point(receptor_positions, offset)
    points = np.unique(np.hstack(query)).astype(int)
    return round(len(points)*mesh_step**3, 2)

# def save_kdtree(mol_positions, cosolv):
#     from openff.toolkit import Molecule, Topology
#     import openmm.app as app
#     molecules = []
#     molecule_positions = []
#     for cosolvent in mol_positions:
#         for i in range(len(mol_positions[cosolvent])):
#             mol = Molecule.from_smiles(cosolvent.smiles, name=cosolvent.name)
#             molecules.append(mol)
#             [molecule_positions.append(x + mol_positions[cosolvent][i]) for x in cosolvent.positions]
#     molecule_positions = np.array(molecule_positions)
#     new_top = Topology.from_molecules(molecules)
#     new_mod = app.Modeller(new_top.to_openmm(), molecule_positions)
#     cosolv.save_pdb(new_mod.topology, new_mod.positions, "results/kdtree.pdb")
#     return


def create_and_parametrize_system(receptor=None, out_file=None):
    if receptor is not None:
        radius = None
    else:
        radius=30*openmmunit.angstrom

    cosolv = CosolventSystem("cosolvents_small.json", "forcefields.json", "amber", receptor=receptor, radius=radius)
    cosolv = fitting_checks(cosolv)
    assert cosolv is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the box padding or radius."
    vectors, lowerBound, upperBound = cosolv.vectors, cosolv.lowerBound, cosolv.upperBound
    print("Adding cosolvents")
    cosolv_xyzs = add_cosolvents(cosolv.cosolvents, vectors, lowerBound, upperBound, cosolv.modeller.positions.value_in_unit(openmmunit.nanometer))
    print("Done adding cosolvents, setting up new topology!")
    cosolv.modeller = cosolv._setup_new_topology(cosolv_xyzs,
                                             cosolv.modeller.topology,
                                             cosolv.modeller.positions)
    print("New topology set up!")
    cosolv.modeller.addSolvent(cosolv.forcefield)
    cosolv.save_pdb(cosolv.modeller.topology, cosolv.modeller.positions, os.path.join("results", out_file))
    return

def fitting_checks(cosolv):
    prot_volume = calculate_receptor_volume(cosolv.modeller.positions)
    empty_volume = cubic_nanometers_to_liters(cosolv.box_volume - prot_volume)
    cosolv._copies_from_concentration(empty_volume)
    cosolvs_volume = defaultdict(float)
    for cosolvent in cosolv.cosolvents:
        cosolvs_volume[cosolvent] = calculate_receptor_volume(cosolv.cosolvents[cosolvent])*cosolvent.copies
    volume_occupied_by_cosolvent = round(sum(cosolvs_volume.values()), 3)
    empty_available_volume = round(liters_to_cubic_nanometers(empty_volume)/2., 3)
    print(f"Volume requested for cosolvents: {volume_occupied_by_cosolvent} nm**3")
    print(f"Volume available for cosolvents: {empty_available_volume} nm**3")
    if volume_occupied_by_cosolvent > empty_available_volume:
        return None
    return cosolv

# def test_volumes(receptor):
#     """Test to check the volumes calculations are okay"""
#     import openmm.app as app
#     cosolv = CosolventSystem("cosolvents.json", "forcefields.json", "amber", receptor=receptor, radius=None)
#     prot_volume = calculate_receptor_volume(cosolv.modeller.positions)
#     cosolv.modeller.addSolvent(cosolv.forcefield)
#     n_waters = cosolv._get_n_waters()
#     wat_volume = liters_to_cubic_nanometers(((n_waters/openmmunit.AVOGADRO_CONSTANT_NA)*(1*openmmunit.liters))/(55.4*openmmunit.moles))
#     print(f"Volume occupied by protein: {prot_volume} nm**3")
#     print(f"Volume occupied by water: {wat_volume} nm**3")
#     print(f"Combined volume: {wat_volume+prot_volume} nm**3")
#     print(f"Total volume: {cosolv._box_volume} nm**3")

#     empty_modeller = app.Modeller(app.Topology(), []) 
#     empty_modeller.topology.setPeriodicBoxVectors(cosolv._periodic_box_vectors)
#     empty_modeller.addSolvent(cosolv.forcefield)
#     res = [r.name for r in empty_modeller.topology.residues()]
#     n_waters_empty = res.count("HOH")
#     wat_volume_empty = liters_to_cubic_nanometers(((n_waters_empty/openmmunit.AVOGADRO_CONSTANT_NA)*(1*openmmunit.liters))/(55.4*openmmunit.moles))
#     vX, vY, vZ = empty_modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
#     box_volume = vX*vY*vZ 
#     print(f"\nVolume occupied by water: {wat_volume_empty} nm**3")
#     print(f"Total volume: {box_volume} nm**3")
#     return

def liters_to_cubic_nanometers(liters):
    if isinstance(liters, openmmunit.Quantity):
        liters = liters.value_in_unit(openmmunit.liters)
    value = liters * 1e+24
    return value

def cubic_nanometers_to_liters(vol):
    value = vol * 1e-24
    return value

if __name__ == "__main__":
    # test_volumes("data/example/protein.pdb")
    create_and_parametrize_system("data/example/protein.pdb", "try_concentration.pdb")