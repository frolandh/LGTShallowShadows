import numpy as np

from quspin.tools.evolution import expm_multiply_parallel
from quspin.operators import hamiltonian
from tqdm import tqdm

from src.rotations import single_shot, apply_product, apply_dual_product
from src.rotations_ising import single_shot_ising
from src.rotation_tools import create_dual_yz
from src.utils import to_binary, indx_d, indx, dual2lgt

from numba import njit
import multiprocessing as mp

NO_CHECKS=dict(check_symm=False, check_pcon=False, check_herm=False)

#------------------------------------------------------------------------------ #
# Simulate measurements                                                         #
#------------------------------------------------------------------------------ #
    
def run_experiment(psi,num_shots,size_args,basis,basis_full,method):
    '''
    Run many rounds of measurements and collect the results. Assumes PBC

    args:
        psi (array) - state that is being measured

        num_shots (int) - number of measurement rounds

        size_args (list) - list of system size parameters

        basis (quspin user basis) - user defined basis for Z2 gauge
        theory
        
        basis_full (quspin spin basis 1d) - The full spin basis
        
        method (string) - either "pairs" or "product"
    
    returns:
        shadow table (array, size=(num_shots,2/3)) - each row is a
        measurement. The first column is angles, the second is pairs (if method == "pairs"),
        the third is a bitstring result. Given the form of rotations
        angles and pairs are sufficient to characterize a unitary
    '''

    #We need to map amplitudes to bitstrings
    Nx,Ny = size_args
    Nspins = 2*Nx*Ny
    z2_configs = [to_binary(s,Nspins) for s in basis_full.states]

    if method == "pairs":
        shadow_table = np.zeros((num_shots,3),dtype=object)
        
        for i_s in tqdm(range(num_shots),desc="Taking Measurements"):
            #Single shot modifies psi so we need a new copy
            psi_shot = psi.copy()
            psi_result,angles,pairs = single_shot(psi_shot,size_args,basis,bc="PBC")

            #Perform measurement
            psi_full = basis.project_from(psi_result).copy().toarray()
            psi_result_x = fast_walsh(psi_full)
            b = projective_measurement(psi_result_x,z2_configs)

            shadow_table[i_s,:] = [angles,pairs,b]
            
        return shadow_table
            
    elif method == "product":
        shadow_table = np.zeros((num_shots,2),dtype=object)
        
        #Build the single qubit rotation matrices
        product_gates = np.zeros((Nspins,2),dtype=object)
        
        for i in range(Nspins):
            static_y,static_z,dynamic = [],[],[]
            
            h = [[1,i]]
            
            static_y.append(["y",h])
            static_z.append(["z",h])
            
            product_gates[i,0] = hamiltonian(static_y,
                                            dynamic,
                                            basis=basis_full,
                                            dtype=np.complex128,
                                            **NO_CHECKS)
                                            
            product_gates[i,1] = hamiltonian(static_z,
                                dynamic,
                                basis=basis_full,
                                dtype=np.complex128,
                                **NO_CHECKS)
                                
        for i_s in tqdm(range(num_shots),desc="Taking Measurements"):
            #Single shot modifies psi so we need a new copy
            psi_shot = basis.project_from(psi).copy().todense()
            psi_result,angles = apply_product(psi_shot,product_gates,size_args,basis_full,bc="PBC")

            #Perform measurement
            b = projective_measurement(psi_result,z2_configs)

            shadow_table[i_s,:] = [angles,b]

        return shadow_table, product_gates
        
def run_experiment_dual_product(psi,num_shots,size_args,basis,basis_full):
    '''
    Run many rounds of measurements and collect the results

    args:
        psi (array) - state that is being measured (including ancilla)

        num_shots (int) - number of measurement rounds

        size_args (list) - list of system size parameters (excluding ancilla)

        basis (quspin user basis) - user defined basis for Z2 gauge
        theory with ancilla
    
    returns:
        shadow table (array, size=(num_shots,2)) - each row is a
        measurement. The first column is unitaries (in dict format),
        the second are the pairs, the third is a bitstring result.
        
    '''
    #We need to map amplitudes to bitstrings
    Nx,Ny = size_args
    N_ising_spins = Nx*Ny
    z2_configs = [to_binary(s,2*N_ising_spins+1) for s in basis_full.states]
    
    shadow_table = np.zeros((num_shots,2),dtype=object)
        
    #Precompute the ising y,z operators
    product_gates = np.zeros((N_ising_spins,2),dtype=object)
    
    for i in range(N_ising_spins):
        coords = dual2lgt(i,size_args)
        product_gates[i,:] = create_dual_yz(coords,size_args,basis)
                           
    for i_s in tqdm(range(num_shots),desc="Taking Measurements"):
        #Single shot modifies psi so we need a new copy
        psi_shot = psi.copy()
        psi_result,angles = apply_dual_product(psi_shot,product_gates,size_args)

        #Perform measurement
        psi_full = basis.project_from(psi_result).copy().toarray()
        psi_result_x = fast_walsh(psi_full)

        b = projective_measurement(psi_result_x,z2_configs)

        shadow_table[i_s,:] = [angles,b]

    return shadow_table
        
def run_experiment_obc(psi,num_shots,size_args,basis,basis_full):
    '''
    Run many rounds of measurements and collect the results

    args:
        psi (array) - state that is being measured

        num_shots (int) - number of measurement rounds

        size_args (list) - list of system size parameters

        basis (quspin user basis) - user defined basis for Z2 gauge
        theory
    
    returns:
        shadow table (array, size=(num_shots,3)) - each row is a
        measurement. The first column is unitaries (in dict format),
        the second are the pairs, the third is a bitstring result.
        
    Notes:
        - This currently only works for spin basis 1d since we cannot implement "I"
        in the user basis
    '''

    #We need to map amplitudes to bitstrings
    Nx,Ny = size_args
    Nspins = 2*Nx*Ny-Nx-Ny
    z2_configs = [to_binary(s,Nspins) for s in basis_full.states]
    
    shadow_table = np.zeros((num_shots,3),dtype=object)
    
    for i_s in tqdm(range(num_shots),desc="Taking Measurements"):
        #Single shot modifies psi so we need a new copy
        psi_shot = psi.copy()
        psi_result,u,pairs = single_shot(psi_shot,size_args,basis_full,bc="OBC")

        #Perform measurement
        psi_full = psi_result.copy()
        psi_result_x = fast_walsh(psi_full)
        b = projective_measurement(psi_result_x,z2_configs)

        shadow_table[i_s,:] = [u,pairs,b]
        
    return shadow_table

#------------------------------------------------------------------------------ #
# Measurement pre/post-processing                                               #
#------------------------------------------------------------------------------ #

@njit
def fast_walsh(psi):
    '''
    jit version of fast hadamard-walsh transform
    '''
    psi = psi.copy()
    N = psi.shape[0]
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            a = psi[i:i + h].copy()       # avoid aliasing
            b = psi[i + h:i + 2 * h].copy()
            psi[i:i + h] = a + b
            psi[i + h:i + 2 * h] = a - b
        h *= 2

    psi /= np.linalg.norm(psi)
    
    return psi

def projective_measurement(psi,spin_configs):
    '''
    Given a state that has been rotated by unitaries, perform a projective
    measurment and output the resulting bitstring

    args:
        psi (array) - state vector in the reduced basis from random measurement

        spin_configs (List[int]) - basis for the system we are using
    
    returns:
        measured_str (str) - measured bitstring on the gauge theory side
    '''
    
    probs = []
    states = []


    #Note that the ordering of z2_to_dual_configs is important
    for i_a,a in enumerate(psi):
        prob = abs(a.item())**2

        probs.append(prob)
        states.append(spin_configs[i_a])
    
    measured_str = np.random.choice(states,p=probs)
    
    return measured_str
    
def state_from_amp(z2_str,size_args,basis,basis_full,basis_z2_full,bc="PBC",ancilla=False):
    '''
    This function takes a Z2 basis string (from the shadow table), and produces the ising
    state vector in the user defined basis

    args:
        z2_str (string) - z2 str from the shadow table

        size_args (list) - Nx, Ny extent of the lattice

        basis (quspin user basis) - This is the reduced basis on the ising side
        that is projected into

        basis_full (quspin spin_basis_1d) - This is the full basis on the ising side.
        This must be used instead of the user basis since the rotation operators that
        account for the global Hadamard will vanish in the reduced basis
        
        basis_z2_full (quspin spin_basis_1d) - This is the full basis on the z2 side.
        This must be used instead of the user basis since the rotation operators that
        account for the global Hadamard will vanish in the reduced basis

        ancilla (bool) - Whether or not the input state includes ancilla qubit for dual product protocol

    returns:
        dual_state_reduced (array) - The state vector in the reduced basis on the ising
        side
    '''
    Nx, Ny = size_args
    Nplaq = Nx*Ny if bc == "PBC" else (Nx-1)*(Ny-1)
    state = np.zeros((2**(Nplaq)),dtype=np.complex128)
    
    dual_state1 = measure_plaquettes(z2_str,size_args,bc=bc,ancilla=ancilla)

    xp1_ind = basis_full.state_to_int(dual_state1)

    state[xp1_ind] = 1

    if bc == "PBC":
        if ancilla:
            return state.reshape(-1,)
        else:
            return basis.project_to(state).toarray().reshape(-1,)
    elif bc == "OBC":
        return state.reshape(-1,)

def measure_plaquettes(z2_bs,size,even_phase=1,bc="PBC",ancilla=False):
    '''
    Takes in Z2 bitstring, and returns corrsponding ising plaquette reading

    args:
        z2_bs (str) - Z2 bit string

        size (list) - Nx,Ny size of Z2 lattice

        even_phase (int) - phase to assign to even plaquette parity string

        bc (str) - either "PBC" or "OBC"

        ancilla (bool) - Whether or not the input state includes ancilla qubit for dual product protocol
    returns:
        ising_b (str) - Ising bitstring corresponding to plaquette parity
    notes:
        The choice of phase corresponding to an even plaquette parity string is arbitrary,
        we set it to 1
    '''

    Nx,Ny = size

    ising_bs = []
    
    x_lim, y_lim = Nx,Ny
    
    if bc == "OBC":
        x_lim -= 1
        y_lim -= 1

    for nx in range(x_lim):
        for ny in range(y_lim):
            plaq_par = 1
            
            plaq_inds = [indx(nx,ny,0,Nx,Ny,bc=bc,ancilla=ancilla),indx(nx+1,ny,1,Nx,Ny,bc=bc,ancilla=ancilla),indx(nx,ny+1,0,Nx,Ny,bc=bc,ancilla=ancilla),indx(nx,ny,1,Nx,Ny,bc=bc,ancilla=ancilla)]


            for ind in plaq_inds:
                plaq_par *= (-1)**(1-int(z2_bs[ind]))

            ising_bs.append(int((1-plaq_par)/2))

    ising_bs = "".join([str(i) for i in ising_bs])

    return ising_bs
