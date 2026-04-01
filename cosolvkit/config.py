import os
import json
from inspect import signature

class Config(object):
    """This class handles the config.json options and the whole Cosolvent setup.

    :param object: inherits from the base class
    :type object: object
    """
    def __init__(self, 
                 cosolvents=None,
                 forcefields=None,
                 md_format=None,
                 protein_path=None,
                 clean_protein=True,
                 keep_heterogens=False,
                 variants=dict(),
                 repulsive_residues=list(),
                 repulsive_epsilon=None,
                 repulsive_sigma=None,
                 solvent_smiles=None,
                 solvent_copies=None,
                 positive_ion="Na+",
                 negative_ion="Cl-",
                 ligands= dict(),
                 padding=None,
                 box_size=None,
                 membrane=False,
                 lipid_type=None,
                 lipid_patch_path=None,
                 memb_cosolv_placement='both',
                 waters_to_keep=list(),
                 output_dir=None,
                 run_cosolvent_system=True,
                 run_md=False):
        
        self.cosolvents = cosolvents
        self.forcefields = forcefields
        self.md_format = md_format.upper()
        self.protein_path = protein_path
        self.clean_protein = clean_protein
        self.keep_heterogens = keep_heterogens
        self.variants = variants
        self.repulsive_residues = repulsive_residues
        self.repulsive_epsilon = repulsive_epsilon
        self.repulsive_sigma = repulsive_sigma
        self.solvent_smiles = solvent_smiles
        self.solvent_copies = solvent_copies
        self.positive_ion = positive_ion
        self.negative_ion = negative_ion
        self.ligands = ligands
        self.padding = padding
        self.box_size = box_size
        self.membrane = membrane
        self.lipid_type = lipid_type
        self.lipid_patch_path = lipid_patch_path
        self.memb_cosolv_placement = memb_cosolv_placement
        self.waters_to_keep = waters_to_keep
        self.output_dir = output_dir
        self.run_cosovlent_system = run_cosolvent_system
        self.run_md = run_md
        self.check_validity()
    
    @classmethod
    def get_defaults_dict(cls):
        """Returns a dictionary of all the class attributes

        :return: dictionary of class attributes
        :rtype: dict
        """
        defaults = {}
        sig = signature(cls)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default 
        return defaults

        
    @classmethod
    def from_config(cls, config):
        """Sets up the parameters to run cosolvent from the config.json file supplied.

        :param config: loads the config.json file and populates the class attributes
        :type config: str
        :raises ValueError: raises an error if some attributes are not recognized
        :return: instance of the Config class
        :rtype: Config
        """
        expected_keys = cls.get_defaults_dict().keys()
        with open(config) as f:
            config = json.load(f)
        bad_keys = [k for k in config if k not in expected_keys]
        if len(bad_keys) > 0:
            err_msg = "unexpected keys in Config.from_config():" + os.linesep
            for key in bad_keys:
                err_msg += "  - %s" % key + os.linesep
            raise ValueError(err_msg)
        p = cls(**config)
        return p
    
    def check_validity(self):
        if self.run_md:
            assert self.md_format == "OPENMM", f"{self.md_format} is not supported with the parameter run_md set to {self.run_md}. Only OPENMM is available with this option."
        return
    