import torch

from .constant import ALANINE_HEAVY_ATOM_IDX

def coordinate2distance(molecule, position):
    '''
        Calculates distance between heavy atoms for Deep LDA
        input
            - molecule (str)
            - coordinates (torch.Tensor)
        output
            - distance (torch.Tensor)
    '''
    
    if molecule == "alanine":
        position = position.reshape(-1, 3)
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
        num_heavy_atoms = len(heavy_atom_position)
        distance = []
        for i in range(num_heavy_atoms):
            for j in range(i+1, num_heavy_atoms):
                distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
        distance = torch.stack(distance)
    
    elif molecule == "chignolin":
        raise NotImplementedError("Will be implemented later")
    
    else:
        raise ValueError(f"Heavy atom distance for molecule {molecule} not supported")
    
    return distance
