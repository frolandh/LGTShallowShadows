import numpy as np

from quspin.operators import hamiltonian
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import pre_check_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from quspin.basis.user import pre_check_state_sig_64,op_sig_64,map_sig_64           # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32,uint64 # numba data types

from quspin.basis import spin_basis_1d

from src.utils import indx,indx_d, modx, mody, indx_obc

NO_CHECKS = dict(check_symm=False, check_pcon=False, check_herm=False)

def calculate_z2_basis(size_args, parity_args,bc="PBC",ancilla=False):
    '''
    Create the user defined basis for the gauge theory, where we only use states tha obey Gauss' laws

    args:
        size_args (list) - Nx andNy extent of the lattice
        
        parity_args (list) - Vx andVy sector that we want to restrict to

        bc (str) - Specifies boundary conditions of the hamiltonian

        ancilla (bool) - If True, include a single ancilla qubit for basis for Dual Product Protocol

    returns:
        basis (quspin user basis) - Only the states that obey Gauss' laws
    '''
    if(bc == "OBC"):
        Nx, Ny = size_args
        Nspins = Nx*(Ny-1)+Ny*(Nx-1)
        N_G_laws = Nx * Ny - 1
       
        # Initialize superindx and j_superindx arrays
        superindx = np.zeros((Nx, Ny, 2), int)
        j_superindx = np.zeros((Nx, Ny, 2), int)

        for nx in range(0, Nx):
            for ny in range(0, Ny):
                for i in range(0, 2):
                    if((nx == Nx-1 and i == 0) or (ny==Ny-1 and i == 1)): #no index assigned along edge
                        continue
                    superindx[nx, ny, i] = indx_obc(nx, ny, i, Nx, Ny)
                    j_superindx[nx, ny, i] = Nspins - superindx[nx, ny, i] - 1
        
        # Define operator function
        @cfunc(op_sig_64, locals=dict(s=int32,b=uint32))
        def op(op_struct_ptr, op_str, ind, N, args):
            op_struct = carray(op_struct_ptr, 1)[0]
            err = 0
            ind = N - ind - 1  # Convention for QuSpin mapping
            s = (((op_struct.state >> ind) & 1) << 1) - 1
            b = (1 << ind)
            
            if op_str == 73:     # "I"
                op_struct.state = b
            elif op_str == 120:  # "x"
                op_struct.state ^= b
            elif op_str == 121:  # "y"
                op_struct.state ^= b
                op_struct.matrix_ele *= 1.0j * s
            elif op_str == 122:  # "z"
                op_struct.matrix_ele *= s
            else:
                op_struct.matrix_ele = 0
                err = -1

            return err

        # Define pre-check state function
        @cfunc(pre_check_state_sig_64,locals=dict(check=int32,jp=uint32,occ_n=int32,occ_nminus1=int32,sum=int32), )
        def pre_check_state(s, N, args):
            mycheck = True
            
            # Check Gauss laws
            for nx in range(0, Nx):
                if not mycheck:
                    break
                for ny in range(0, Ny):
                    if not mycheck:
                        break
                    
                    nx_m = nx - 1
                    ny_m = ny - 1

                    phase = 1

                    if(nx != Nx - 1):#only check if not on boundary
                        j_x = j_superindx[nx, ny, 0]
                        occ_x = (s >> j_x) & 1
                        phase *= (2 * occ_x - 1)

                    if(nx != 0):#only check if not on boundary
                        j_x_m = j_superindx[nx_m, ny, 0]
                        occ_x_m = (s >> j_x_m) & 1
                        phase *= (2 * occ_x_m - 1)

                    if(ny != Ny - 1):#only check if not on boundary
                        j_y = j_superindx[nx, ny, 1]
                        occ_y = (s >> j_y) & 1
                        phase *= (2 * occ_y - 1)

                    if(ny != 0):#only check if not on boundary
                        j_y_m = j_superindx[nx, ny_m, 1]
                        occ_y_m = (s >> j_y_m) & 1
                        phase *= (2 * occ_y_m - 1)

                    if phase == -1:
                        mycheck = False
                        break            

            return mycheck
        
        # Define identity mapping function
        @cfunc(map_sig_64)
        def ident_mapping(s, a, b, c):
            return s

        # Estimate number of states
        estimated_states = 2**(Nspins - N_G_laws) 

        op_args=np.array([],dtype=np.uint64)
        op_dict = dict(op=op,op_args=op_args)
        pre_check_state_args=None
        pre_check_state = (pre_check_state, None)
        maps = dict(id_block=(ident_mapping, 1, 0, np.array([], dtype=np.uint64)))

        # Call user_basis (you'll need to define this function or import it from a suitable library)
        basis = user_basis(np.uint64, Nspins, op_dict, allowed_ops=set("Ixyz"), sps=2, pre_check_state=pre_check_state, Ns_block_est=estimated_states, **maps)

    else:
        Nx, Ny = size_args
        Vx, Vy = parity_args
        Nspins = 2 * Nx * Ny
        N_G_laws = Nx * Ny - 1
        
        # Initialize superindx and j_superindx arrays
        superindx = np.zeros((Nx, Ny, 2), int)
        j_superindx = np.zeros((Nx, Ny, 2), int)

        for nx in range(0, Nx):
            for ny in range(0, Ny):
                for i in range(0, 2):
                    superindx[nx, ny, i] = indx(nx, ny, i, Nx, Ny)
                    j_superindx[nx, ny, i] = Nspins - superindx[nx, ny, i] - 1
                    if ancilla:
                        j_superindx[nx, ny, i] += 1
        
        # Define operator function
        @cfunc(op_sig_64, locals=dict(s=int32,b=uint32))
        def op(op_struct_ptr, op_str, ind, N, args):
            op_struct = carray(op_struct_ptr, 1)[0]
            err = 0
            ind = N - ind - 1  # Convention for QuSpin mapping
            s = (((op_struct.state >> ind) & 1) << 1) - 1
            b = (1 << ind)
            
            if op_str == 120:  # "x"
                op_struct.state ^= b
            elif op_str == 121:  # "y"
                op_struct.state ^= b
                op_struct.matrix_ele *= 1.0j * s
            elif op_str == 122:  # "z"
                op_struct.matrix_ele *= s
            else:
                op_struct.matrix_ele = 0
                err = -1

            return err

        # Define pre-check state function
        @cfunc(pre_check_state_sig_64,locals=dict(check=int32,jp=uint32,occ_n=int32,occ_nminus1=int32,sum=int32), )
        def pre_check_state(s, N, args):
            mycheck = True
            
            if ancilla:
                occ_a = (s >> 0) & 1
                                
            # Check Gauss laws
            for nx in range(0, Nx):
                if not mycheck:
                    break
                for ny in range(0, Ny):
                    if not mycheck:
                        break
                    
                    if nx == 0:
                        nx_m = Nx - 1
                    else:
                        nx_m = nx - 1

                    if ny == 0:
                        ny_m = Ny - 1
                    else:
                        ny_m = ny - 1

                    phase = 1
                    j_x = j_superindx[nx, ny, 0]
                    occ_x = (s >> j_x) & 1
                    phase *= (2 * occ_x - 1)

                    j_x_m = j_superindx[nx_m, ny, 0]
                    occ_x_m = (s >> j_x_m) & 1
                    phase *= (2 * occ_x_m - 1)

                    j_y = j_superindx[nx, ny, 1]
                    occ_y = (s >> j_y) & 1
                    phase *= (2 * occ_y - 1)
                    if ancilla and nx == 0 and ny == Ny-1:
                        phase *= (2*occ_a - 1)

                    j_y_m = j_superindx[nx, ny_m, 1]
                    occ_y_m = (s >> j_y_m) & 1
                    phase *= (2 * occ_y_m - 1)
                    if ancilla and nx == 0 and ny_m == Ny-1:
                        phase *= (2*occ_a - 1)
                    
                    if phase == -1:
                        mycheck = False
                        break
            
            # Check Vy and Vx symmetry sectors
            if np.abs(Vy) == 1 and mycheck:
                phase = 1
                for ny in range(Ny):
                    j = j_superindx[0, ny, 0]
                    occ = (s >> j) & 1
                    phase *= (2 * occ - 1)
                if phase != Vy:
                    mycheck = False

            if np.abs(Vx) == 1 and mycheck:
                phase = 1
                for nx in range(Nx):
                    j = j_superindx[nx, 0, 1]
                    occ = (s >> j) & 1
                    phase *= (2 * occ - 1)
                    if ancilla and nx == 0:
                        phase *= (2*occ_a - 1)
                
                if phase != Vx:
                    mycheck = False

            return mycheck
        
        # Define identity mapping function
        @cfunc(map_sig_64)
        def ident_mapping(s, a, b, c):
            return s

        # Estimate number of states
        if Vy == 0 and Vx == 0:
            estimated_states = 2**(Nspins) + 1
        else:
            estimated_states = 2**(Nspins - 2) + 1

        op_args=np.array([],dtype=np.uint64)
        op_dict = dict(op=op,op_args=op_args)
        pre_check_state_args=None
        pre_check_state = (pre_check_state, None)
        maps = dict(id_block=(ident_mapping, 1, 0, np.array([], dtype=np.uint64)))

        # Call user_basis (you'll need to define this function or import it from a suitable library)
        if(ancilla):
            Nspins += 1
            estimated_states *= 2
        basis = user_basis(np.uint64, Nspins, op_dict, allowed_ops=set("xyz"), sps=2, pre_check_state=pre_check_state, Ns_block_est=estimated_states, **maps)

    return basis

def calculate_ising_basis(size_args,bc="PBC"):
    '''
    Create the user defined basis for the dual ising theory, where we only use
    states in a given parity sector if PBC

    args:
        size_args (list) - Nx and Ny extent of the lattice

    returns:
        basis (quspin user basis) - Only the states that have global parity conserved, if PBC
    '''
    if(bc == "OBC"):
        Nx, Ny = size_args
        Nspins = (Nx-1)*(Ny-1)
        basis = spin_basis_1d(Nspins)
        return basis
    else:
        Nx, Ny = size_args
        Nspins = Nx * Ny
    
    # Define operator function
    @cfunc(op_sig_64, locals=dict(s=int32,b=uint32))
    def op(op_struct_ptr,op_str,ind,N,args):
        # using struct pointer to pass op_struct_ptr back to C++ see numba Records
        op_struct = carray(op_struct_ptr,1)[0]
        err = 0
        ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
        s = (((op_struct.state>>ind)&1)<<1)-1
        b = (1<<ind)
        # We implement xyz operator, +/- are not needed for our spin Hamiltonian
        if op_str==120: # "x" is integer value 120 (check with ord("x"))
            op_struct.state ^= b
        elif op_str==121: # "y" is integer value 120 (check with ord("y"))
            op_struct.state ^= b
            op_struct.matrix_ele *= 1.0j*s
        elif op_str==122: # "z" is integer value 120 (check with ord("z"))
            op_struct.matrix_ele *= s
        else:
            op_struct.matrix_ele = 0
            err = -1
        
        return err

    # Define pre-check state function
    @cfunc(pre_check_state_sig_64,locals=dict(check=int32,jp=uint32,occ_n=int32,occ_nminus1=int32,sum=int32), )
    def pre_check_state(s,N,args):
        mycheck = True
        phase=1

        for nx in range(Nx):
            phase_x=1
        
            for ny in range(Ny):
                j = nx*Ny+ny
                j = N - 1 - j
                
                occ = (s>>j)&1
                phase *= (2*occ-1)
        if phase == -1:
            mycheck=False
        
        return mycheck
    
    # Define identity mapping function
    @cfunc(map_sig_64) # or map_sig_64 for 64-bit integer
    def ident_mapping(s,a,b,c):
        return s
    ident_args=np.array([],dtype=np.uint64)
    maps = dict(id_block=(ident_mapping,1,0,ident_args), )

    op_args=np.array([],dtype=np.uint64)
    op_dict = dict(op=op,op_args=op_args)
    pre_check_state_args=None
    pre_check_state=(pre_check_state,pre_check_state_args)      # None gives a null pointer to args

    # Estimate number of states
    estimated_states=2**(Nx*Ny)

    basis = user_basis(np.uint64,Nspins,op_dict,allowed_ops=set("xyz"),sps=2,pre_check_state=pre_check_state,Ns_block_est=estimated_states,**maps)

    return basis

def z2_to_ising(size_args,z2_basis):
    '''
    This provides a mapping between bitstrings on the gauge theory side
    and bitstrings on the Ising side. We find them by moving through each
    lattice site and multiplying adjacent links to get relative parity.
    This is relevant for the post processing.

    args:
        size_args (list) - List of the Nx and Ny extent of the lattice
        z2_basis (quspin user basis) - The  gauge fixed basis we use (must be on Z2 side)

    returns:
        z2_to_dual_configs (dict) - The keys are the Z2 bitstrings and the
        two Ising bitstrings associated with this string

    notes:
        Since we find the Ising states by relative parity, we need to set
        a value for the first qubit. This is why there are two Ising strings
        for each Z2 string, related to one another by inversion of all spins
        
        This function is now deprecated, the product of Zs around a plaquette
        is now used.
    '''
    Nx,Ny = size_args

    dual_configs = {n:[] for n in z2_basis.states}

    for n in z2_basis.states:
        s = [int(i) for i in z2_basis.int_to_state(n)[1:-1] if i != ' ']
        
        dual_configs[n].append(s)

        l1 = [0]

        for nx in range(Nx):
            ny_init = 1 if nx == 0 else 0
                
            if ny_init == 0:
                j_right = indx(nx,0,1,Nx,Ny)
                rel_phase = 2*s[j_right]-1
                p0 = 2*l1[-Ny]-1

                l1.append(int((1 + p0 * rel_phase)/2))
                
            for ny in range(1,Ny):
                j_up = indx(nx,ny,0,Nx,Ny)
                rel_phase = 2*s[j_up]-1
                p0 = 2*l1[-1]-1

                l1.append(int((1 + p0 * rel_phase)/2))
                
        dual_configs[n].append(l1)

        l2 = [1]

        for nx in range(Nx):
            ny_init = 1 if nx == 0 else 0
                
            if ny_init == 0:
                j_right = indx(nx,0,1,Nx,Ny)
                rel_phase = 2*s[j_right]-1
                p0 = 2*l2[-Ny]-1

                l2.append(int((1 + p0 * rel_phase)/2))
                
            for ny in range(1,Ny):
                j_up = indx(nx,ny,0,Nx,Ny)
                rel_phase = 2*s[j_up]-1
                p0 = 2*l2[-1]-1

                l2.append(int((1 + p0 * rel_phase)/2))
                
        dual_configs[n].append(l2)
        
    z2_to_dual_configs = {''.join([str(i) for i in value[0]]): 
                          [''.join([str(i) for i in v]) for v in value[1:]] 
                          for value in dual_configs.values()}
    
    return z2_to_dual_configs

#------------------------------------------------------------------------------ #
# Construct Hamiltonians                                                        #
#------------------------------------------------------------------------------ #

def getH_Z2(size_args,coupling_args,basis,bc="PBC"):
    '''
    This creates the Z2 hamiltonian

    args:
        size_args (list) - Nx and Ny extent of the lattice
        coupling_args (list) - The strength of the electric and magnetic field
        basis (quspin user basis) - Gauge fixed basis

    returns:
        H (quspin hamiltonian) - The Gauge theory hamiltonian. Convert to sparse
        matrix by using .tocsc() method

    notes:
        We have flipped x and z compared to the notes since the Gauss' law can only
        be checked in the z basis (this is how quspin does things). 
    '''

    Nx, Ny = size_args
    K,g = coupling_args

    static, dynamic = [], []
    
    # plaquette terms ---------------------------------------------------------- #
   
    if np.abs(K)>1e-15:            
        h_plaquette = []        
        for nx in range(Nx):
            for ny in range(Ny):
                #The ordering here is ctr-clockwise are plaquette
                if(bc=="OBC"):
                    if(nx == Nx-1 or ny == Ny-1):
                        #if on boundary continue
                        continue
                    h_plaquette.append([K,indx_obc(nx,ny,0,Nx,Ny),indx_obc(nx+1,ny,1,Nx,Ny)
                                   ,indx_obc(nx,ny+1,0,Nx,Ny),indx_obc(nx,ny,1,Nx,Ny)])
                else:
                    h_plaquette.append([K,indx(nx,ny,0,Nx,Ny),indx(nx+1,ny,1,Nx,Ny)
                                   ,indx(nx,ny+1,0,Nx,Ny),indx(nx,ny,1,Nx,Ny)])
                    
        static.append(["xxxx",h_plaquette])
    # END plaquette terms ---------------------------------------------------------- #
    
    # electric field terms -- x direction --------------------------------------- #
   
    if np.abs(g)>1e-15:
        h_ex = []
            
        for nx in range(Nx):
            for ny in range(Ny):
                if(bc=="OBC"):
                    if(nx ==Nx-1):
                        continue
                    h_ex.append([g,indx_obc(nx,ny,0,Nx,Ny)])
                else:
                    h_ex.append([g,indx(nx,ny,0,Nx,Ny)])
                    
        static.append(["z",h_ex])
    # END electric field terms -- x direction ------------------------------------- #
    
    # electric field terms -- y direction --------------------------------------- #
   
    if np.abs(g)>1e-15:    
        h_ey = []
        for nx in range(Nx):
            for ny in range(Ny):
                if(bc=="OBC"):
                    if(ny == Ny-1):
                        continue
                    h_ey.append([g,indx_obc(nx,ny,1,Nx,Ny)])
                else:
                    h_ey.append([g,indx(nx,ny,1,Nx,Ny)])
                
        static.append(["z",h_ey])
    
    # END electric field terms -- y direction --------------------------------------- #

    NO_CHECKS=dict(check_symm=False, check_pcon=False, check_herm=False)
    
    H =  hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
    return H

def getH_ising(size_args,coupling_args,parity_args,basis,bc="PBC"):
    '''
        This creates the Z2 hamiltonian

        args:
            size_args (list) - Nx and Ny extent of the lattice
            
            coupling_args (list) - The strength of the electric and magnetic field
            
            parity_args (list) - Vx, Vy values
            
            my_basis (quspin user basis) - Gauge fixed basis
            
            bc (str) - Specifies boundary conditions of the hamiltonian

        returns:
            H (quspin hamiltonian) - The ising theory hamiltonian. Convert to sparse
            matrix by using .tocsc() method

        notes:
            We do not implement different ribbon sectors (i.e. Vx,Vy) values. This Hamiltonian
            has only been tested for Vx=Vy=1
        '''

    if bc == "PBC":
        Vx, Vy = parity_args
        assert abs(Vx) == 1 and abs(Vy) == 1, "Vx,Vy sectors must be either +1,-1 for PBC"
        
        static, dynamic = getH_isingPBC(size_args,coupling_args,parity_args,basis)
        H =  hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
        
    else:
        Nx, Ny = size_args
        K,g = coupling_args
        static,dynamic=[],[]

        if abs(K) > 1e-15:
            h_p = []
            for nx in range(Nx-1):
                for ny in range(Ny-1):
                    h_p.append([K,indx_d(nx,ny,Ny-1)])

            static.append(["z",h_p])

        if abs(g) >= 1e-15:
            # Bulk electric fields in X-direction
            h_e = []
            for nx in range(Nx-1):
                for ny in range(1,Ny-1):
                    h_e.append([g,indx_d(nx,ny,Ny-1),indx_d(nx,(ny-1)%(Ny-1),(Ny-1))])

            static.append(["xx",h_e])

            # Bulk electric fields in Y-direction
            h_e = []
            for nx in range(1,Nx-1):
                for ny in range(Ny-1):
                    h_e.append([g,indx_d(nx,ny,(Ny-1)),indx_d((nx-1)%(Nx-1),ny,Ny-1)])

            static.append(["xx",h_e])

            # Boundary electric fields in X-direction
            h_e = []
            for nx in range(Nx-1):
                h_e.append([g,indx_d(nx,0,Ny-1)])
                h_e.append([g,indx_d(nx,Ny-2,Ny-1)])

            static.append(["x",h_e])

            # Boundary electric fields in Y-direction
            h_e = []
            for ny in range(Ny-1):
                h_e.append([g,indx_d(0,ny,(Ny-1))])
                h_e.append([g,indx_d(Nx-2,ny,(Ny-1))])

            static.append(["x",h_e])
            
        H =  hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

    return H
    
def getH_isingPBC(size_args,coupling_args,parity_args,basis):
    '''
        This creates the Ising hamiltonian with PBC

        args:
            size_args (list) - Nx and Ny extent of the lattice
            
            coupling_args (list) - The strength of the electric and magnetic field
            
            parity_args (list) - Vx, Vy values
            
            my_basis (quspin user basis) - Gauge fixed basis

        returns:
            H (quspin hamiltonian) - The ising theory hamiltonian. Convert to sparse
            matrix by using .tocsc() method

        notes:
            We do not implement different ribbon sectors (i.e. Vx,Vy) values. This Hamiltonian
            has only been tested for Vx=Vy=1
        '''

    Nx, Ny = size_args
    Vx,Vy = parity_args
    K,g = coupling_args

    static,dynamic=[],[]

    if abs(K) > 1e-15:
        h_p = []
        for nx in range(Nx):
            for ny in range(Ny):
                h_p.append([K,indx_d(nx,ny,Ny)])

        static.append(["z",h_p])

    if abs(g) >= 1e-15:
        # Electric fields in X-direction
        h_e = []
        for nx in range(Nx):
            for ny in range(Ny):
                coupling = g if ny>0 else g*Vy
                h_e.append([coupling,indx_d(nx,ny,Ny),indx_d(nx,(ny-1)%Ny,Ny)])

        static.append(["xx",h_e])

        # Electric fields in Y-direction
        h_e = []
        for nx in range(Nx):
            for ny in range(Ny):
                coupling = g if nx > 0 else g*Vx
                h_e.append([coupling,indx_d(nx,ny,Ny),indx_d((nx-1)%Nx,ny,Ny)])

        static.append(["xx",h_e])
        
    return static, dynamic
