import numpy as np
import math
import random
import warnings
import scipy as sp
import itertools
from functools import reduce

from scipy.stats import unitary_group

# Single-qubit Pauli matrices
paulis = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}

#-------------------------------------------- #
# Generate angles from CUE (SU(2) unitaries)  #
#-------------------------------------------- #

def angles_from_unitary(U):
    '''
    Given an SU(2) unitary, decomposes it into three angles theta1,theta2,theta3
    '''
    U1=U[0,0]
    U2=U[0,1]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", np.exceptions.ComplexWarning)
    
        theta1 = np.real( math.atan2(-1j*(np.conj(U1)-U1),(np.conj(U1)+U1)) - math.atan2(-1j*(np.conj(U2)-U2),(np.conj(U2)+U2))  )/2
        theta2 = np.real( math.atan( np.sqrt( ((np.abs(U2))**2) / ((np.abs(U1))**2) )  ) )
        theta3 = np.real( math.atan2(-1j*(np.conj(U1)-U1),(np.conj(U1)+U1)) + math.atan2(-1j*(np.conj(U2)-U2),(np.conj(U2)+U2))  )/2
        
    return theta1,theta2,theta3
    
def haar_measure(n):
    '''
    Generates an n-dimensional haar random unitary using the QR algorithm
    
    args:
        n (int) - The dimensionsion of the unitary
        
    returns:
        n-dimensional haar random unitary
        
    notes:
        For the SU(4) rotations below the scipy package is used instead.
        This is a design choice made after this code was implemented.
    '''
    
    z = (np.random.normal(loc=0,scale=1,size=(n,n)) + 1j*np.random.normal(loc=0,scale=1,size=(n,n))) / np.sqrt(2)
    
    q,r = np.linalg.qr(z)
    
    d = np.diagonal(r)
    lamb = d / abs(d)
    lamb = np.diag(lamb)
    
    return q @ lamb

def my_haar():
    '''
    Generates an SU(2) unitary
    
    returns:
        U (array) - 2x2 unitary
        
    notes:
        The determinant is fixed to 1
    '''
    U = unitary_group.rvs(2)
    U_det = np.linalg.det(U)
    U *= np.sqrt(np.conj(U_det))
    
    return U

def generate_CUE_angles(N):
    '''
    Generates CUE angle for N SU(2) unitaries
    
    args:
        N (int) - the number of unitaries
        
    returns:
        angles (list) - 3N list of angles
    '''
    angles = np.zeros(3*N,float)

    for q in range(N):
        U = my_haar()
        angles[3*q+0],angles[3*q+1],angles[3*q+2]=angles_from_unitary(U)
        
    return angles

#-------------------------------------------- #
#          Generate SU(4) Unitaries           #
#-------------------------------------------- #

def pauli_string_to_matrix(s):
    """Constructs the matrix from a Pauli string (e.g., 'IXZ')."""
    return reduce(np.kron,[paulis[c] for c in s])

def all_pauli_strings(n):
    """Generate all n-qubit Pauli strings."""
    return [''.join(p) for p in itertools.product('Ixyz', repeat=n)]

def decompose_unitary(U):
    """Decomposes a 2^n x 2^n unitary into the Pauli basis."""
    n = int(np.log2(U.shape[0]))
    coeffs = {}
    for pstr in all_pauli_strings(n):
        P = pauli_string_to_matrix(pstr)
        coeff = np.trace(P.conj().T @ U) / (2**n)
        if np.abs(coeff) > 1e-10:  # filter small terms
            coeffs[pstr] = coeff
    return coeffs

def create_2q_U():
    su4 = unitary_group.rvs(4)

    return decompose_unitary(su4)
