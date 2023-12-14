import json
import numpy as np
from collections import defaultdict
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit
from openff.toolkit import Molecule
from cosolvkit.cosolvent import CoSolvent
from scipy import spatial
import itertools
from cosolvkit.utils import fix_pdb



class OpenmmCosolventBox:
    # Need to update CDKtree every time I add a new cosolvent and consider it for the next time
    # Need to find a way to add the new topologies once I'm done
    # After that there is to solvate and do the real paramterization with forcefields and then 
    # create the system and run the simulation

    # TODO:
    # Add random rotation of the cosolvents so that their orientation is not always the same 
    #   -> Could bring to a bias in short MD simulations
    # Add random selection and placement of different cosolvents

    def __init__(self, cosolvents, receptor=None, padding=12*openmmunit.angstrom, radius=None):
        self.kdtree = None
        self.receptor_cutoff = 6.5*openmmunit.angstrom
        self.cosolvents_cutoff = 3.5*openmmunit.angstrom
        self.receptor = receptor
        self.cosolvents = dict()
        self.cosolvent_positions = defaultdict(list)
        self.box = None
        self.modeller = None
        for cosolvent in cosolvents:
            with open("benzene.json") as fi:
                c = json.load(fi)
            cosolvent = CoSolvent(**c)
            cosolvent_xyz = cosolvent.positions*openmmunit.angstrom
            self.cosolvents[cosolvent.name] = cosolvent_xyz.value_in_unit(openmmunit.nanometer)
        if receptor is not None:
            top, pos = fix_pdb(receptor)
            self.modeller = Modeller(top, pos)
            self.kdtree = spatial.cKDTree(receptor.positions.value_in_unit(openmmunit.nanometer))
            if self.modeller.getTopology() is not None:
                self.modeller.deleteWater()
        
        if self.receptor is None:
            self.box = self._build_box(None, padding, radius=radius)
        else:
            self.box = self._build_box(self.modeller.positions, padding, radius=None)
        
        # Setting up the box
        self.modeller.topology.setPeriodicBoxVectors(self.box[0])
        return

    def build(self, cosolvents, cosolvent_positions):
        # Still need to account for the concentration
        for cosolvent in cosolvents:
            cosolvent_xyz = cosolvents[cosolvent]
            sizeX, sizeY, sizeZ = cosolvent_xyz.max(axis=0) - cosolvent_xyz.min(axis=0)
            center_xyzs = self._build_mesh(self.modeller, sizeX, sizeY, sizeZ, cutoff=self.cosolvent_cutoff)
            new_coords = cosolvent_xyz + center_xyzs[i]
            for i in range(len(center_xyzs)):
                new_coords = cosolvent_xyz + center_xyzs[i]
                if self.kdtree is not None:
                    if not any(self.kdtree.query_ball_point(new_coords, self.cutoff_receptor.value_in_unit(openmmunit.nanometer))):
                        cosolvent_positions[cosolvent].append(new_coords)
        return cosolvent_positions
    
    def _setup_new_topology(self, cosolvents_positions, receptor_molecules=None, receptor_positions=None):
        # Adding the cosolvent molecules
        molecules = []
        molecules_positions = []
        for cosolvent in cosolvents_positions:
            for i in range(len(cosolvents_positions[cosolvent])):
                molecules.append(Molecule.from_smiles(cosolvent.smiles))
                [molecules_positions.append(x) for x in cosolvents_positions[cosolvent][i]]

        # Here I need to add the original receptor (iterate over molecules and positions)

        molecules_positions = np.array(molecules_positions)
        new_top = Topology.from_molecules(molecules)
        new_mod = Modeller(new_top.to_openmm(), molecules_positions)
        return new_mod

    def _build_box(self, positions, padding, radius=None):
        padding = padding.value_in_unit(openmmunit.nanometer)
        if positions is not None:
            positions = positions.value_in_unit(openmmunit.nanometer)
            minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
            maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
            center = 0.5*(minRange+maxRange)
            radius = max(unit.norm(center-pos) for pos in positions)
            print(radius)
        else:
            radius = radius.value_in_unit(openmmunit.nanometer)
        width = max(2*radius+padding, 2*padding)
        vectors = (Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width))
        box = Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        return vectors, box
    
    def _build_mesh(self, modeller, sizeX, sizeY, sizeZ, cutoff):
        vX, vY, vZ = modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
        positions = modeller.positions.value_in_unit(openmmunit.nanometer)
        center = [(max((pos[i] for pos in positions))+min((pos[i] for pos in positions)))/2 for i in range(3)]
        origin = center - (np.ceil(np.array([vX, vY, vZ])).astype(int)/2)
        xmin, xmax = origin[0], origin[0] + vX
        ymin, ymax = origin[1], origin[1] + vY
        zmin, zmax = origin[2], origin[2] + vZ

        cutoff = cutoff.value_in_unit(openmmunit.nanometer)
        x = np.arange(xmin, xmax, sizeX+cutoff) + cutoff
        y = np.arange(ymin, ymax, sizeY+cutoff) + cutoff
        z = np.arange(zmin, zmax, sizeZ+cutoff) + cutoff

        X, Y, Z = np.meshgrid(x, y, z)
        center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        return center_xyzs

    
