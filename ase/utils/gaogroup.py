from ase import *
import numpy as np
from numpy import transpose,inv,dot


def rotate_cell(atom, target_cell):
    orig_cell=atom.get_cell()
    deform_grad= dot(transpose(target_cell),inv(transpose(orig_cell)))
    J_def_grad=np.linalg.det(deform_grad)
    
    orig_coord=atom.get_positions()
    

    
