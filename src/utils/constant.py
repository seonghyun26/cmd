# Methods
MLCOLVAR_METHODS = [
    "deeplda",
    "deeptda",
    "deeptica",
    "autoencoder",
    "timelagged-autoencoder"
]

CLCV_METHODS = [
    "cvmlp",
    "cvmlp-bn",
    "cvmlp-test",
]


# Alanine Dipeptide
ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]
ALDP_THETA_ANGLE = [1, 4, 6, 8]
ALDP_OMEGA_ANGLE = [8, 14, 16, 18]
ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]
ALANINE_BACKBONE_ATOM_IDX = [1, 4, 6, 8, 10, 14, 16, 18]
ALANINE_HEAVY_ATOM_EDGE_INDEX = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
]
ALANINE_HEAVY_ATOM_ATTRS =[
    [1., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.],
    [1., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.]
]


# Chignolin
# TBA