File `protein_parts.json` contains molecules that resemble protein sidechains
and N-methylacetamide as a representative of protein backbone.

Here, N-methylacetamide is named `NML` but in `cosolvents.json` in the main
directory it is named `ACE`. Here `ACE` is acetamide.

All molecules use the PDB 3-letter code to identify them,
except isobutane (IBT) and ethylammonium (NCC).

We don't know whether or not it is a good idea to simulate all of these at
the same time. The concentrations of charged cosolvents differ from other
molecules so that the net cosolvent charge is zero.
