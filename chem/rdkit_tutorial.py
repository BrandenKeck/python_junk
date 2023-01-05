import numpy as np
from rdkit import Chem

# Creating a simple molecule
toluene = Chem.MolFromSmiles('C1C=CC=CC1C')
hydrogen_cyanide = Chem.MolFromSmiles('C#N')
hydrogen_cyanide = Chem.MolFromSmiles('C#N')

# Build bond array
molecule = toluene
molecule_data = []
for i in range(toluene.GetNumAtoms()):
    molecule_data.append([])
    for j in range(toluene.GetNumAtoms()):
        if i == j or toluene.GetBondBetweenAtoms(i,j) == None:
            molecule_data[i].append(0)
        elif toluene.GetBondBetweenAtoms(i,j).GetBondType() == Chem.rdchem.BondType.SINGLE:
            molecule_data[i].append(1)
        elif toluene.GetBondBetweenAtoms(i,j).GetBondType() == Chem.rdchem.BondType.DOUBLE:
            molecule_data[i].append(2)
        elif toluene.GetBondBetweenAtoms(i, j).GetBondType() == Chem.rdchem.BondType.TRIPLE:
            molecule_data[i].append(3)
        else:
            molecule_data[i].append(0)

print(np.array(molecule_data))
