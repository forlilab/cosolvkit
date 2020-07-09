from cosolvkit import CoSolventBox

cosolv = CoSolventBox(concentration=0.25, cutoff=12, box="orthorombic")
cosolv.add_receptor("protein.pdb")
cosolv.add_cosolvent(name='benzene', smiles='c1ccccc1')
cosolv.add_cosolvent(name='methanol', smiles='CO', resname="MEH")
cosolv.add_cosolvent(name='propane', smiles='CCC', resname="PRP")
cosolv.add_cosolvent(name='imidazole', smiles='C1=CN=CN1')
cosolv.add_cosolvent(name='acetamide', smiles='CC(=O)NC', resname="ACT")
cosolv.build()
cosolv.export(prefix="cosolv")

#cosolv = CoSolventBox(concentration=0.15, cutoff=12, box="orthorombic")
#cosolv.add_receptor("protein.pdb")
#cosolv.build()
#cosolv.export(prefix="gist")