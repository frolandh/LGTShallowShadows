import numpy as np

from quspin.operators import hamiltonian
from math import comb, prod, factorial
from scipy import sparse
from itertools import combinations, permutations

from src.rotations import rotate_back,rotate_back_product,rotate_back_obc,rotate_back_dual_product
from src.measurements import state_from_amp
from src.utils import indx_d

NO_CHECKS=dict(check_symm=False, check_pcon=False, check_herm=False)

def generate_swaps(op_str,d):
    '''
    Return a list of all Z strings that are a swap distance d from op_str

    args:
        op_str (str) - Bit string of length sites that corresponds to the observable we want to measure
        There will be a 1 if there is a Z in the string and a 0 if there is identity
        
        d (int) - swap distance
        
    returns:
        swapped_strs (list) - list of all strings at some swap distance.

    notes:
        A string will have a 1 if there is a Z there and a 0 if it is identity
    '''
    
    swapped_strs = []
    
    n = len(op_str)
    site_indices = range(n)
    nonzero_indxs = [indx for indx, element in enumerate(op_str) if element != 0]
    zero_indxs = [indx for indx in site_indices if not(indx in nonzero_indxs)]

    swap_indxs_a = list(permutations(nonzero_indxs,d)) # why do I need to turn these into lists
    swap_indxs_b = list(permutations(zero_indxs,d))
    
    if d == 0:
        return [op_str]
    else:
        for indxs_a in swap_indxs_a:
            for indxs_b in swap_indxs_b:
                new_str = op_str.copy()
                
                for i_d in range(d):
                    indx_a = indxs_a[i_d]
                    indx_b = indxs_b[i_d]
                    
                    new_str[indx_a],new_str[indx_b] = 0,1

                swapped_strs.append(new_str)

                del new_str
            
    return swapped_strs
      
def get_z(z_str,basis):
    '''
    Return the quspin operator in the ising basis that correspond to a specified z string

    args:
        z_str (list) - z string we want to construct. There will be a 1 if the site hosts a z
        0 if identity

        basis (quspin user basis) - This is the user basis we defined on the ising side

    returns:
        H_z (quspin hamiltonian) - Operator that is a string of Zs
    '''
    static, dynamic = [], []
    h_z = []
    
    nonzero_indxs = [i for i,s in enumerate(z_str) if s == 1]
    num_z = sum(z_str)
    
    if num_z == 0:
        return sparse.csc_matrix(np.eye(basis.Ns))
    else:
        h_z.append([1,*nonzero_indxs])

        static.append(["z"*num_z,h_z])

        H_z = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

        return H_z
        
def get_xy(x_str,y_str,basis):
    '''
    Return the quspin operator in the ising basis that correspond to a specified z string

    args:
        x_str (list) - x string we want to construct. There will be a 1 if the site hosts a x
		0 if identity
                
		y_str (list) - y string we want to construct. There will be a 1 if the site hosts a y
        0 if identity

        basis (quspin user basis) - This is the user basis we defined on the ising side

    returns:
        H_xy (quspin hamiltonian) - Operator that is a string of XYs
    '''
    static, dynamic = [], []
    h_xy = []
    
    nonzero_indxs = [i for i,s in enumerate(x_str) if s == 1] + [i for i,s in enumerate(y_str) if s == 1]
    num_x = sum(x_str)
    num_y = sum(y_str)
    
    if num_x == 0 and num_y == 0:
        return sparse.csc_matrix(np.eye(basis.Ns))
    else:
        h_xy.append([1,*nonzero_indxs])

        static.append(["x"*num_x+"y"*num_y,h_xy])

        H_xy = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

        return H_xy
        
def get_xyz(x_str,y_str,z_str,basis):
    '''
    Return the quspin operator in the ising basis that correspond to a specified z string

    args:
        x_str (list) - x string we want to construct. There will be a 1 if the site hosts a x
        0 if identity
                
        y_str (list) - y string we want to construct. There will be a 1 if the site hosts a y
        0 if identity
        
        z_str (list) - z string we want to construct. There will be a 1 if the site hosts a y
        0 if identity

        basis (quspin user basis) - This is the user basis we defined on the ising side

    returns:
        H_xyz (quspin hamiltonian) - Operator that is a string of XYs
    '''
    static, dynamic = [], []
    h_xyz = []
    
    nonzero_indxs = ([i for i,s in enumerate(x_str) if s == 1]
                    + [i for i,s in enumerate(y_str) if s == 1]
                    + [i for i,s in enumerate(z_str) if s == 1]
                    )
                    
    num_x = sum(x_str)
    num_y = sum(y_str)
    num_z = sum(z_str)
    
    if num_x + num_y + num_z == 0:
        return sparse.csc_matrix(np.eye(basis.Ns))
    else:
        h_xyz.append([1,*nonzero_indxs])

        static.append(["x"*num_x+"y"*num_y+"z"*num_z,h_xyz])

        H_xyz = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

        return H_xyz
    
def compute_sample_pairs(x_str,y_str,z_str,b_str,angles,pairs,size_args,basis,basis_full,basis_z2_full):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the gauge theory side.

    args:
        x_str (list) - list of integers describing x operator we want
        
        y_str (list) - list of integers describing y operator we want
        
        z_str (list) - list of integers describing z operator we want

        b_str (string) - measurement results from shadow table (|b> in the paper)

        pairs (list) - pairs from shadow table

        angles (list) - angles from shadow table
        
        size_args (list) - Nx,Ny size of the system

        basis (qupin user basis) - user basis for the ising model

        basis_full (quspin spin basis 1d) - full spin basis to be used for getting the ising state
    
    returns:
        total (float) - estimate for the observable we want

    notes:
        This currently only does estimates for zs! Needs to be updated for x and y
    '''
    
    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    psi_reduced = state_from_amp(b_str,size_args,basis,basis_full,basis_z2_full)
    orig_psi = rotate_back(psi_reduced,angles,pairs,basis)
    Nx,Ny = size_args
    V = Nx*Ny
    nz = sum(z_str)
    j_slots = [1 if x_str[i] != y_str[i] else 0 for i in range(len(x_str))]
    nj = sum(j_slots)
    
    expt_z,expt_xy = 1,1
    if nz != 0:
        deflation = deflation_factor(V,nz)
        inflation = 1 / deflation
        
        z_op = get_z(z_str,basis)
        expt_z *= inflation * z_op.expt_value(orig_psi)
        
        return expt_z
    if nj != 0:
        f = P(nj) * P(V - nj) / P(V)
        
        #Inverting channel corresponds to multiplying by 3^number of pairs JpJm, etc
        xy_op = get_xy(x_str,y_str,basis)
        
        expt_xy *= ((3)**(nj//2) / f) * xy_op.expt_value(orig_psi)
        

        return expt_xy
        
def compute_sample_pairs_obc(x_str,y_str,z_str,b_str,su4_decomp,pairs,size_args,basis,basis_full,basis_z2_full):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the gauge theory side.

    args:
        x_str (np.ndarrays) - list of integers describing x operator we want
        
        y_str (np.ndarrays) - list of integers describing y operator we want
        
        z_str (np.ndarrays) - list of integers describing z operator we want

        b_str (string) - measurement results from shadow table (|b> in the paper)

        pairs (list) - pairs from shadow table

        su4_decomp (dict) - Dictionary describing the rotations
        
        size_args (list) - Nx,Ny size of the system

        basis (qupin user basis) - user basis for the ising model

        basis_full (quspin spin basis 1d) - full spin basis to be used for getting the ising state
    
    returns:
        total (float) - estimate for the observable we want

    notes:
        This currently only does estimates for zs! Needs to be updated for x and y
    '''
    if not isinstance(x_str, np.ndarray):
        x_str = np.array(x_str)
    if not isinstance(y_str, np.ndarray):
        y_str = np.array(y_str)
    if not isinstance(z_str, np.ndarray):
        z_str = np.array(z_str)
    
    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    psi_reduced = state_from_amp(b_str,size_args,basis,basis_full,basis_z2_full,bc="OBC")
    orig_psi = rotate_back_obc(psi_reduced,su4_decomp,pairs,basis)
    Nx,Ny = size_args
    V = (Nx-1)*(Ny-1)
    k_dual = sum(x_str)+sum(y_str)+sum(z_str)
    
    op = get_xyz(x_str,y_str,z_str,basis)
    
    atilde = alpha_obc(V,k_dual)
    
    return op.expt_value(orig_psi) / atilde
    
def compute_sample_product(x_str,y_str,z_str,b_str,angles,size_args,basis_full,product_gates):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the gauge theory side.

    args:
        x_str (list) - list of integers describing x operator we want
        
        y_str (list) - list of integers describing y operator we want
        
        z_str (list) - list of integers describing z operator we want

        b_str (string) - measurement results from shadow table (|b> in the paper)

        angles (list) - angles from shadow table
        
        size_args (list) - Nx,Ny size of the system

        basis_full (quspin spin basis 1d) - full spin basis to be used for getting the Z2 state
    
    returns:
        total (float) - estimate for the observable we want
        
    '''
    
    Nx,Ny = size_args
    V = Nx*Ny
    nx = sum(x_str)
    nz = sum(z_str)
    ny = sum(y_str)
    weight = nx+ny+nz

    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    b_indx = 2**(2*V) - basis_full.state_to_int(b_str) - 1
    psi = np.zeros(2**(2*V),dtype=complex)
    psi[b_indx] = 1
    psi = np.matrix(psi).T
    orig_psi = rotate_back_product(psi,angles,product_gates,basis_full)
    
    xyz_op = get_xyz(x_str,y_str,z_str,basis_full)
    expt = 3**weight * xyz_op.expt_value(orig_psi)
        
    return expt
    
def compute_sample_dual_product(x_str,y_str,z_str,b_str,angles,size_args,basis,basis_full,basis_z2_full):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the gauge theory side.

    args:
        x_str (list) - list of integers describing x operator we want
        
        y_str (list) - list of integers describing y operator we want
        
        z_str (list) - list of integers describing z operator we want

        b_str (string) - measurement results from shadow table (|b> in the paper)

        angles (list) - angles from shadow table
        
        size_args (list) - Nx,Ny size of the system

        basis_full (quspin spin basis 1d) - full spin basis to be used for getting the Z2 state
    
    returns:
        total (float) - estimate for the observable we want
        
    '''
    
    Nx,Ny = size_args
    V = Nx*Ny
    nx = sum(x_str)
    nz = sum(z_str)
    ny = sum(y_str)
    weight = nx+ny+nz

    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    psi = state_from_amp(b_str,size_args,basis,basis_full,basis_z2_full,ancilla=True)
    orig_psi = rotate_back_dual_product(psi,angles,basis_full)
    
    xyz_op = get_xyz(x_str,y_str,z_str,basis_full)
    expt = 3**(weight) * xyz_op.expt_value(orig_psi)
        
    return expt

def compute_sample_ising_z(op_str,b_str,angles,pairs,beta,size_args,basis):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the Ising side.

    args:
        op_str (list) - list of integers describing z operator we want

        b_str (string) - measurement results from shadow table (|b> in the paper)

        pairs (list) - pairs from shadow table

        angles (list) - angles from shadow table

        beta (list) - precomputed array of channel coefficients

        size_args (list) - Nx,Ny size of the system

        basis (qupin user basis) - user basis for the ising model
    
    returns:
        total (float) - estimate for the observable we want

    notes:
        This currently only does estimates for zs! Needs to be updated for x and y
    '''
    
    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    num_states = len(basis.states)
    Nx,Ny = size_args
    psi = np.zeros((num_states),dtype=np.complex128)
    ind = basis.states.tolist().index(basis.state_to_int(b_str))
    psi[ind] = 1
    orig_psi = rotate_back(psi,angles,pairs,basis)
    
    nz = sum(op_str)
    
    V = Nx*Ny
    deflation = deflation_factor(V,nz)
    inflation = 1 / deflation
    
    #Inverting channel corresponds to multiplying by 3^number of pairs JpJm, etc
    z_op = get_z(op_str,basis)
    expt = inflation * z_op.expt_value(orig_psi)

    return expt
    
def compute_sample_ising_xy(x_str,y_str,b_str,angles,pairs,size_args,basis):
    '''
    Take in a shadow table entry and compute sum_{d=0}^{nx} beta_d sum_{S_d} Tr[U^{dag}|b><b|U S_d]
    This is for data that was measured on the Ising side.

    args:
        x_str (list) - list of integers describing x positions
        
        y_str (list) - list of integers describing y positions

        b_str (string) - measurement results from shadow table (|b> in the paper)

        pairs (list) - pairs from shadow table

        angles (list) - angles from shadow table

        size_args (list) - Nx,Ny size of the system

        basis (qupin user basis) - user basis for the ising model
    
    returns:
        total (float) - estimate for the observable we want
    '''
    
    #Take the measurement and compute the unrotated state (Udag|b><b|U)
    num_states = len(basis.states)
    Nx,Ny = size_args
    V = Nx*Ny
    psi = np.zeros((num_states),dtype=np.complex128)
    ind = basis.states.tolist().index(basis.state_to_int(b_str))
    psi[ind] = 1
    orig_psi = rotate_back(psi,angles,pairs,basis)
    
    #This counts the qbuits that have only an X or a Y. If there are none or both
    #then it is identity or Z
    j_slots = [1 if x_str[i] != y_str[i] else 0 for i in range(len(x_str))]
    
    nj = sum(j_slots)
    f = P(nj) * P(V - nj) / P(V)
    
    #Inverting channel corresponds to multiplying by 3^number of pairs JpJm, etc
    xy_op = get_xy(x_str,y_str,basis)
    
    if type(xy_op) != sparse._csc.csc_matrix:
        expt = (3**(nj//2) / f) * xy_op.expt_value(orig_psi)
    else:
        expt = (3**(nj//2) / f) * np.conj(orig_psi) @ xy_op @ orig_psi

    return expt

#------------------------------------------------------------------------------ #
# This is for precomputing the beta coefficients                                #
#------------------------------------------------------------------------------ #

def P(V):
    '''
    Number of pairings for system of size V

    args:
        V (int) - system size

    returns:
        number of pairings
    '''
    V = V-1

    return prod(range(V, 0, -2))

def a(V,nz):
    '''
    Compute the alpha_d coefficients as specified in the paper

    args:
        V (int) - system size

        nz (int) - nz sector of the observable we want to estimate

    returns:
        total (float) - the value of alpha
    '''

    total = 0
    
    for m in range(nz+1):
        if np.mod(m,2) == np.mod(nz,2):
            total += (2/3)**m * comb(nz,m) * comb(V-nz,m) * factorial(m) * P(nz-m) * P(V-nz-m)
            
    return total / P(V)
    
def deflation_factor(V,nz):
    '''
    Compute the factor M[So]=deflation * So, where S0 is a Z string
    This is actually just the d=0 swap amplitude with (2/3)^m->(1/3)^m.

    args:
        V (int) - system size

        nz (int) - nz sector of the observable we want to estimate

    returns:
        total (float) - the value of the channel
    '''

    total = 0
    
    for m in range(nz+1):
        if np.mod(m,2) == np.mod(nz,2):
            total += (1/3)**m * comb(nz,m) * comb(V-nz,m) * factorial(m) * P(nz-m) * P(V-nz-m)
            
    return total / P(V)

def G(V,nz,l,d):
    '''
    Compute the change of basis between different representations of the channel

    args:
        V (int) - system size

        nz (int) - nz sector of the string we want to estimate

        l (int) - permutation irrep label

        d (int) - swap distance

    returns:
        total (float) - Matrix element G_{ld}
    '''
    total = 0
    
    for x in range(d+1):
        y = d-x
        
        total += (-1)**x * comb(l,x) * comb(nz-l,y) * comb(V-nz-l,y)
        
    return total

def c(V,l):
    '''
    Compute the eigenvalues of the channel

    args:
        V (int) - system size

        l (int) - permutation irrep label

    returns:
        total (float) - eigenvalue c_l
    '''

    total = 0
    
    for d in range(l+1):
        for m in range(l-d+1):
            if np.mod(m,2) == np.mod(l - d,2):
                total += (-1/3)**d * (2/3)**m * (factorial(l) / factorial(l-d-m)) * comb(V-d-l,m) * (P(l-d-m) * P(V-d-l-m) / P(V))
                
    return total

def get_betas(V,nz):
    '''
    Compute the inverse channel

    args:
        V (int) - system size

        nz (int) - number of Zs in the string we want to compute

    returns:
        beta (list) - list of beta values for each swap distancein the string
    '''
    betas = np.zeros(V,dtype=object)

    l_max = min(nz,V-nz)

    G_mat = np.zeros((l_max+1,l_max+1),dtype=float)
    c_vec = np.zeros(l_max+1,dtype=float)

    for l in range(l_max+1):
        c_vec[l] = c(V,l)

        for d in range(l_max+1):
            G_mat[l,d] = G(V,nz,l,d)

    c_inv = 1.0 / c_vec
    G_inv = np.linalg.inv(G_mat)
    beta = G_inv @ c_inv

    return beta

# OBC Code

def alpha_obc(V,k_dual):
    '''
    Channel inversion factor for OBC case
    
    args:
        V - size of the lattice
        
        k_dul - The weight of the dual operator
        
    returns:
        Channel inverion factor \tilde{\alpha}
    '''
    
    sum = 0
    
    for m in range(0,int(k_dual/2)+1):
        sum += comb(k_dual, 2*m) * P(2*m) * comb(V-k_dual,k_dual-2*m) * factorial(k_dual-2*m) / 5**(k_dual-m)
    
    return sum / P(V)
