import numpy as np
import math
import random
import warnings
import scipy as sp

from scipy.stats import unitary_group
from quspin.operators import hamiltonian, quantum_operator, exp_op
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

from src.haar_tools import *
from src.utils import indx, modx, mody

NO_CHECKS = dict(check_symm=False, check_pcon=False, check_herm=False)

#------------------------------------------------------------------------------ #
# Build Single Plaquette Rotations (only for Z2)                                #
#------------------------------------------------------------------------------ #


def build_ZI_rotation_ising(i,j,theta,parity,basis):
    '''
    Build the ZI rotation generator with ising operators
    args:
        i (int) - index of the first qubit

        j (int) - index of the second qubit

        theta (float) - The angle we rotate by

        parity (int) - This specifies whether the gate acts on the even

        or odd parity sector. Can only be \pm 1

        basis (quspin user basis) - The user basis we use

    returns:
        R_theta (quspin expmultiplyparallel) - Rotation operator.
        Use R_theta.dot(psi) to apply this to a states

    notes:
        We don't store this as a matrix for efficiency
    '''

    static, dynamic = [], []
    
    h_zi = [[1,i]]
    h_zj = [[parity,j]]
    
    static.append(["z",h_zi])
    static.append(["z",h_zj])
    
    H_rot = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS).tocsc()
    
    R_theta = expm_multiply_parallel(H_rot,a=1j*theta/2)
    
    return R_theta

def build_YX_rotation_ising(i,j,theta,parity,basis):
    '''
    Build the YX rotation generator with ising operators

    args:
        i (int) - index of the first qubit

        j (int) - index of the second qubit

        size_args (list) - Gives the size of the lattice in each direction

        theta (float) - The angle we rotate by

        parity (int) - This specifies whether the gate acts on the even
        or odd parity sector. Can only be \pm 1

        basis (quspin user basis) - The user basis we use

    returns:
        R_theta (quspin expmultiplyparallel) - Rotation operator.
        Use R_theta.dot(psi) to apply this to a states

    notes:
        We don't store this as a matrix for efficiency
    '''

    static, dynamic = [], []
    
    h_yx = [[1,i,j]]
    h_xy = [[parity,i,j]]
    
    static.append(["yx",h_yx])
    static.append(["xy",h_xy])
    
    H_rot = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS).tocsc()
    
    R_theta = expm_multiply_parallel(H_rot,a=1j*theta/2)
    
    return R_theta
    
def apply_U_ij_conj_ising(psi,i,j,angles,basis):
    '''
    Build the unitary (in the 2 qubit CUE parameterization) with Z2 operators

    args:
        psi (array) - State vector that we will rotate

        i (int) - index of the first qubit

        j (int) - index of the second qubit

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits

    notes:
        Since we invert the channel on the ising side this is for constructing the conjugate of the unitary

        Also for the z2 operators we described each qubit by x and y coordinates.
        Here we just use the index (so 0<= i,j < Nx*Ny)

    '''

    a,b,g,ap,bp,gp = [-1*theta for theta in angles] #reverse the angles
    
    R_a = build_ZI_rotation_ising(i,j,a,-1,basis)
    R_b = build_YX_rotation_ising(i,j,b,-1,basis)
    R_g = build_ZI_rotation_ising(i,j,g,-1,basis)

    R_ap = build_ZI_rotation_ising(i,j,ap,1,basis)
    R_bp = build_YX_rotation_ising(i,j,bp,1,basis)
    R_gp = build_ZI_rotation_ising(i,j,gp,1,basis)
    
    psi_rot=psi.copy().astype(np.complex128)
    
    try:
        state_size = len(psi_rot)
    except TypeError:
        state_size = psi_rot.shape[0]
    
    work_array=np.zeros((2*state_size,), dtype=psi.dtype)
    
    #Note that the operations are reversed compared to the Z2 case
    U_odd_list = [R_g,R_b,R_a]
    for rot in U_odd_list:
        rot.dot(psi_rot,work_array=work_array,overwrite_v=True)
    
    U_even_list = [R_gp,R_bp,R_ap]
    for rot in U_even_list:
        rot.dot(psi_rot,work_array=work_array,overwrite_v=True)
    
    return psi_rot
    
def apply_U_ij_ising_obc(psi,i,j,decomp_conj,basis):
    
    static, dynamic = [], []
    
    for key,val in decomp_conj.items():
        hij = [[val,i,j]]
        
        static.append([key,hij])
    
    H_rot = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
    return H_rot.dot(psi)
    
def get_ising_yz(index,basis):
    staticy,staticz,dynamic = [],[],[]
    
    staticy.append(['y',[[1,index]]])
    staticz.append(['z',[[1,index]]])
    
    hy = hamiltonian(staticy, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    hz = hamiltonian(staticz, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
    return hy,hz

################ Pure Ising rotations (Old) ################

def apply_random_pairs_ising(psi,pairs,basis):
    '''
    Takes pairs of indices and applies pairwise random unitaries based on the indices
    Forward measurements on the ising side

    args:
        psi (array) - state vector
        pairs (list) - list of pairs we apply unitaries to
    
    returns:
        psi (array) - pairwise randomized state
        angle_list (list) - random list of angles
    '''
    angle_list = []
    
    for i,j in pairs:
        angles = generate_CUE_angles(2)
        angle_list.append(angles)
        
        psi = apply_U_ij_ising(psi,i,j,angles,basis)
        
    return [psi, angle_list]

def apply_U_ij_ising(psi,i_coords,j_coords,angles,basis):
    '''
    Build the unitary (in the 2 qubit CUE parameterization) with ising operators
    Forward transformation on the ising side

    args:
        psi (array) - State vector that we will rotate

        i_coords (int) - index of the first qubit

        j_coords (int) - index of the second qubit

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits
    '''
    
    i,j = i_coords,j_coords
    a,b,g,ap,bp,gp = angles
    
    R_a = build_ZI_rotation_ising(i,j,a,-1,basis)
    R_b = build_YX_rotation_ising(i,j,b,-1,basis)
    R_g = build_ZI_rotation_ising(i,j,g,-1,basis)
    
    R_ap = build_ZI_rotation_ising(i,j,ap,1,basis)
    R_bp = build_YX_rotation_ising(i,j,bp,1,basis)
    R_gp = build_ZI_rotation_ising(i,j,gp,1,basis)
    
    #We first apply the even parity unitaries, then the odd
    #The angle ordering is gp,bp,ap,g,b,a
    
    psi_rot=psi.copy().astype(np.complex128)
    work_array=np.zeros((2*(psi_rot.shape[0]),), dtype=psi.dtype)
    
    U_even_list = [R_ap,R_bp,R_gp]
    for rot in U_even_list:
        rot.dot(psi_rot,work_array=work_array,overwrite_v=True)
        
    U_odd_list = [R_a,R_b,R_g]
    for rot in U_odd_list:
        rot.dot(psi_rot,work_array=work_array,overwrite_v=True)
    
    return psi_rot
    
def single_shot_ising(psi,size_args,basis):
    '''
    Apply the gates in a single round of the experiment
    for the pure ising side

    args:
        psi (array) - state vector that we randomize over

        size_args (list) - Contains the size of the system

        basis (quspin user basis) - Gauge fixed basis

    returns:
        psi_random (array) - randomized state vector and associated angles

        pairs (list) - pairs of indices chosen at random
    '''
    
    Nx,Ny = size_args
    
    dual_ind_list = np.arange(0,Nx*Ny)
    random.shuffle(dual_ind_list)

    pairs = [[dual_ind_list[i],dual_ind_list[i+1]] for i in np.arange(0,len(dual_ind_list)-1,2)]
    
    psi_random = apply_random_pairs_ising(psi,pairs,basis)
    
    return [psi_random,pairs]
