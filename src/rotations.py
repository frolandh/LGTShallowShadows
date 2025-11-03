import numpy as np
import random

from quspin.tools.evolution import expm_multiply_parallel

from src.haar_tools import *
from src.rotation_tools import *
from src.rotations_ising import apply_U_ij_conj_ising,apply_U_ij_ising_obc,get_ising_yz
from src.utils import dual2lgt

import sys

#------------------------------------------------------ #
# Pairs Protocols                                       #
#------------------------------------------------------ #
    
def single_shot(psi,size_args,basis,bc="PBC"):
    '''
    Apply the gates in a single round of the experiment for all pairs

    args:
        psi (array) - state vector that we randomize over

        size_args (list) - Contains the size of the system

        basis (quspin user basis) - Gauge fixed basis
        
        bc (str) - specifies what the boundary conditons are

    returns:
        psi_random (array) - randomized state vector
        
        unitary_data (list/dict) - If bc = PBC, returns a list of angles associated with each pair
        If bc = OBC, returns a dictionary that specifies the decomposition of the unitary

        pairs (list) - pairs of indices chosen at random
    '''
    
    Nx,Ny = size_args
    
    if bc == "PBC":
        dual_ind_list = np.arange(0,Nx*Ny)
    elif bc == "OBC":
        dual_ind_list = np.arange(0,(Nx-1)*(Ny-1))
        
    random.shuffle(dual_ind_list)

    pairs = [[dual_ind_list[i],dual_ind_list[i+1]] for i in np.arange(0,len(dual_ind_list)-1,2)]
    
    psi_random,unitary_data = apply_random_pairs(psi,pairs,size_args,basis,bc=bc)
    
    return psi_random,unitary_data,pairs
    
def apply_random_pairs(psi,pairs,size_args,basis,bc="PBC"):
    '''
    Takes pairs of indices and applies pairwise random unitaries based on the indices

    args:
        psi (array) - state vector
        
        pairs (list) - list of pairs we apply unitaries to
        
        size_args (list) - Size of lattice
        
        basis (quspin spinbasis1d or user) - basis we do the rotations in
    
    returns:
        psi (array) - pairwise randomized state
        
        angle_list (list) - random list of angles
    '''
    angle_list = []
    su4_decompositions = []
    
    for i,j in pairs:
        i = dual2lgt(i,size_args,bc=bc)
        j = dual2lgt(j,size_args,bc=bc)
        
        if bc == "PBC":
            angles = generate_CUE_angles(2)
            angle_list.append(angles)
            
            psi = apply_U_ij_Z2_pbc(psi,i,j,size_args,angles,basis)
            
        elif bc == "OBC":
            ##Here we generate a decomposition of SU(4) unitry as a dict
            su4_decomposition = create_2q_U()
            su4_decompositions.append(su4_decomposition)
            
            psi = apply_U_ij_Z2_obc(psi,i,j,size_args,su4_decomposition,basis)
            
    if bc == "PBC":
        return psi, angle_list
    elif bc == "OBC":
        return psi, su4_decompositions
    
def apply_U_ij_Z2_pbc(psi,i_coords,j_coords,size_args,angles,basis):
    '''
    Build the unitary (in the 2 qubit CUE parameterization) with Z2 operators

    args:
        psi (array) - State vector that we will rotate

        i_coords (list) - x and y coordinates of the first qubit

        j_coords (list) - x and y coordinates of the second qubit

        size_args (list) - Gives the size of the lattice in each direction

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits
    '''

    positions = [i_coords,j_coords]
    a,b,g,ap,bp,gp = angles
    
    R_a = build_ZI_rotation_Z2(positions,size_args,a,-1,basis,"PBC")
    R_b = build_YX_rotation_Z2(positions,size_args,b,-1,basis,"PBC")
    R_g = build_ZI_rotation_Z2(positions,size_args,g,-1,basis,"PBC")
    
    R_ap = build_ZI_rotation_Z2(positions,size_args,ap,1,basis,"PBC")
    R_bp = build_YX_rotation_Z2(positions,size_args,bp,1,basis,"PBC")
    R_gp = build_ZI_rotation_Z2(positions,size_args,gp,1,basis,"PBC")
    
    try:
        state_size = len(psi)
    except TypeError:
        state_size = psi.shape[0]
        
    work_array=np.zeros((2*state_size,), dtype=psi.dtype)
    
    U_even_list = [R_ap,R_bp,R_gp]
    for rot in U_even_list:
        rot.dot(psi,work_array=work_array,overwrite_v=True)
        
    U_odd_list = [R_a,R_b,R_g]
    for rot in U_odd_list:
        rot.dot(psi,work_array=work_array,overwrite_v=True)
    
    return psi
    
def apply_U_ij_Z2_obc(psi,i_coords,j_coords,size_args,su4_decomposition,basis):
    '''
    Build the unitary with Z2 operators

    args:
        psi (array) - State vector that we will rotate

        i_coords (list) - x and y coordinates of the first qubit

        j_coords (list) - x and y coordinates of the second qubit
        
        size_args (list) - Gives the size of the lattice in each direction

        su4_decomposition (dict) - This dictionary describes how the SU(4) unitary decomposes into
        2qubit paulis

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits
    '''

    positions = [i_coords,j_coords]
    
    #Create the rotations here
    su4_unitary = create_SU4_unitary(su4_decomposition,positions,size_args,basis)
    
    return su4_unitary.dot(psi)

def rotate_back(psi,angles,pairs,basis):
    '''
    Rotate a state psi by the inverse of the unitary in the shadow table

    args:
        psi (array) - State vector that we will rotate

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        pairs (list) - A list of pairs that parameterize the full unitary

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits

    notes:
        This is essentialy the inverse of the apply random pairs functions
        This is a post-processing method
    '''

    for i_p, (i,j) in enumerate(pairs):
        psi = apply_U_ij_conj_ising(psi,i,j,angles[i_p],basis)
        
    return psi
    
def rotate_back_obc(psi,su4_decomps,pairs,basis):
    '''
    Rotate a state psi by the inverse of the unitary in the shadow table

    args:
        psi (array) - State vector that we will rotate

        su4_decomps (dict of length 6) - The coefficients of the unitary.

        pairs (list) - A list of pairs that parameterize the full unitary

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits

    notes:
        This is essentialy the inverse of the apply random pairs functions
        Due to the way we do the unitaries we just need to conjugate the coefficients
        This is a post-processing method
    '''

    for i_p, (i,j) in enumerate(pairs):
        decomp = su4_decomps[i_p]
        decomp_conj = {key:np.conj(val) for key,val in decomp.items()}
        
        psi = apply_U_ij_ising_obc(psi,i,j,decomp_conj,basis)
        
    return psi

#------------------------------------------------------ #
# Product Protocols                                     #
#------------------------------------------------------ #

def apply_product(psi,product_gates,size_args,basis,bc="PBC"):
    '''
    Takes pairs of indices and applies pairwise random unitaries based on the indices

    args:
        psi (array) - state vector
        
        product_gates (array (2*NxNy,2)) - list of precomputed y,z matrices for the rotations
        
        size_args (list) - Size of lattice
        
        basis (quspin spinbasis1d or user) - basis we do the rotations in
    
    returns:
        psi (array) - pairwise randomized state
        
        angle_list (list) - random list of angles
    '''
    angle_list = []
    
    Nx,Ny = size_args
    
    for i_qubit in range(2*Nx*Ny):
    
        angles = generate_CUE_angles(1)
        angle_list.append(angles)
        
        #Build the rotation matrices
        h_y,h_z = product_gates[i_qubit,:]
        a,b,g = angles
    
        R_a = expm_multiply_parallel(h_z.tocsc(),a=1j*a)
        R_b = expm_multiply_parallel(h_y.tocsc(),a=1j*b)
        R_g = expm_multiply_parallel(h_z.tocsc(),a=1j*g)
        
        try:
            state_size = len(psi)
        except TypeError:
            state_size = psi.shape[0]
            
        work_array=np.zeros((2*state_size,), dtype=psi.dtype)
        
        R_a.dot(psi,work_array=work_array,overwrite_v=True)
        R_b.dot(psi,work_array=work_array,overwrite_v=True)
        R_g.dot(psi,work_array=work_array,overwrite_v=True)
        
    return [psi, angle_list]
    
def rotate_back_product(psi,angles,product_gates,basis):
    '''
    Rotate a state psi by the inverse of the unitary in the shadow table

    args:
        psi (array) - State vector that we will rotate

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        pairs (list) - A list of pairs that parameterize the full unitary

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits

    notes:
        This is essentialy the inverse of the apply random pairs functions
    '''
    
    N_qubits,_ = product_gates.shape

    for i_qubit in range(N_qubits):
        #Build the rotation matrices
        h_y,h_z = product_gates[i_qubit,:]
        a,b,g = angles[i_qubit]

        R_a = expm_multiply_parallel(h_z.tocsc(),a=-1j*a)
        R_b = expm_multiply_parallel(h_y.tocsc(),a=-1j*b)
        R_g = expm_multiply_parallel(h_z.tocsc(),a=-1j*g)
        
        try:
            state_size = len(psi)
        except TypeError:
            state_size = psi.shape[0]
            
        work_array=np.zeros((2*state_size,), dtype=psi.dtype)
    
        R_g.dot(psi,work_array=work_array,overwrite_v=True)
        R_b.dot(psi,work_array=work_array,overwrite_v=True)
        R_a.dot(psi,work_array=work_array,overwrite_v=True)
        
    return psi

def apply_dual_product(psi,product_gates,size_args):
    '''
    args:
        psi (array) - state vector
        
        product_gates (array (NxNy,2)) - list of precomputed y,z matrices for the rotations
        
        size_args (list) - Size of lattice
        
        basis (quspin spinbasis1d or user) - basis we do the rotations in
    
    returns:
        psi (array) - pairwise randomized state
        
        angle_list (list) - random list of angles
    '''
    angle_list = []
    
    Nx,Ny = size_args
    num_ising_spins = Nx * Ny
    
    for i_q in range(num_ising_spins):
    
        angles = generate_CUE_angles(1)
        angle_list.append(angles)
        
        #Build the rotation matrices
        h_y,h_z = product_gates[i_q,:]
        a,b,g = angles
    
        R_a = expm_multiply_parallel(h_z.tocsc(),a=1j*a)
        R_b = expm_multiply_parallel(h_y.tocsc(),a=1j*b)
        R_g = expm_multiply_parallel(h_z.tocsc(),a=1j*g)
        
        try:
            state_size = len(psi)
        except TypeError:
            state_size = psi.shape[0]
            
        work_array=np.zeros((2*state_size,), dtype=psi.dtype)
        
        if not isinstance(psi, np.ndarray):
            psi = psi.toarray()

        
        R_a.dot(psi,work_array=work_array,overwrite_v=True)
        R_b.dot(psi,work_array=work_array,overwrite_v=True)
        R_g.dot(psi,work_array=work_array,overwrite_v=True)
        
    return [psi, angle_list]

def rotate_back_dual_product(psi,angles,basis):
    '''
    Rotate a state psi by the inverse of the unitary in the shadow table

    args:
        psi (array) - State vector that we will rotate

        angles (list of length 6) - The angles used in the gates.
        There are 6, 3 for each qubit

        pairs (list) - A list of pairs that parameterize the full unitary

        basis (quspin user basis) - The user basis we use

    returns:
        psi_rot (array) - The rotated state after we have applied unitaries
        on all qubits

    notes:
        This is essentialy the inverse of the apply random pairs functions
    '''
    
    N_qubits = len(angles)

    for i_q in range(N_qubits):
        #Build the rotation matrices
        
        a,b,g = angles[i_q]
        
        h_y,h_z = get_ising_yz(i_q,basis)

        R_a = expm_multiply_parallel(h_z.tocsc(),a=-1j*a)
        R_b = expm_multiply_parallel(h_y.tocsc(),a=-1j*b)
        R_g = expm_multiply_parallel(h_z.tocsc(),a=-1j*g)
        
        try:
            state_size = len(psi)
        except TypeError:
            state_size = psi.shape[0]
            
        work_array=np.zeros((2*state_size,), dtype=psi.dtype)
    
        R_g.dot(psi,work_array=work_array,overwrite_v=True)
        R_b.dot(psi,work_array=work_array,overwrite_v=True)
        R_a.dot(psi,work_array=work_array,overwrite_v=True)
        
    return psi
