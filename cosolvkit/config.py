import os
import json
from inspect import signature

class Config(object):
    def __init__(self, 
                 cosolvents=None,
                 forcefields=None,
                 md_format=None,
                 receptor=True,
                 protein_path=None,
                 clean_protein=True,
                 keep_heterogens=False,
                 variants_d=dict(),
                 add_repulsive=False,
                 repulsive_residues=list(),
                 solvent_smiles=None,
                 solvent_copies=None,
                 membrane=False,
                 lipid_type=None,
                 lipid_patch_path=None,
                 cosolvent_placement=0,
                 waters_to_keep=list(),
                 radius=None,
                 output=None,
                 run_cosolvent_system=True,
                 run_md=False):
        
        self.cosolvents = cosolvents
        self.forcefields = forcefields
        self.md_format = md_format
        self.receptor = receptor
        self.protein_path = protein_path
        self.clean_protein = clean_protein
        self.keep_heterogens = keep_heterogens
        self.variants_d = variants_d
        self.add_repulsive = add_repulsive
        self.repulsive_residues = repulsive_residues
        self.solvent_smiles = solvent_smiles
        self.solvent_copies = solvent_copies
        self.membrane = membrane
        self.lipid_type = lipid_type
        self.lipid_patch_path = lipid_patch_path
        self.cosolvent_placement = cosolvent_placement
        self.waters_to_keep = waters_to_keep
        self.radius = radius
        self.output = output
        self.run_cosovlent_system = run_cosolvent_system
        self.run_md = run_md
    
    @classmethod
    def get_defaults_dict(cls):
        defaults = {}
        sig = signature(cls)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default 
        return defaults

        
    @classmethod
    def from_config(cls, config):
        expected_keys = cls.get_defaults_dict().keys()
        with open(config) as f:
            config = json.load(f)
        bad_keys = [k for k in config if k not in expected_keys]
        if len(bad_keys) > 0:
            err_msg = "unexpected keys in MoleculePreparation.from_config():" + os.linesep
            for key in bad_keys:
                err_msg += "  - %s" % key + os.linesep
            raise ValueError(err_msg)
        p = cls(**config)
        return p