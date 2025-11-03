import numpy as np

from quspin.operators import hamiltonian, quantum_operator, exp_op
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

from src.utils import indx, modx, mody, link_path, dual2lgt

NO_CHECKS = dict(check_symm=False, check_pcon=False, check_herm=False)

#------------------------------------------------------------------------------ #
# Build Single Plaquette Rotations (only for Z2)                                #
#------------------------------------------------------------------------------ #

def build_Xtilde(pos_args,size_args,basis):
    '''
    Build the X tilde operator (corresponds to path of zs)

    args:
        nx,ny - coordinates of X
        size_args (list) - Gives the size of the lattice in each direction
        basis (quspin user basis) - The user basis we use

    returns:
        LGT dual of Ising Xtilde operator

    notes:
        for now we assume the path travels horizontally
        along the bottom of the lattice then goes up.
        This is not parallelizable
    '''

    nx,ny = pos_args
    Nx,Ny = size_args

    static = []
    dynamic = []
    
    path_x = [indx(i,0,1,Ny) for i in range(nx+1)]
    path_y = [indx(nx,mody(j+1,Nx),0,Ny) for j in range(ny)]

    path = path_x+path_y
    couple = [[1.0]+path]

    x_str = "z" * (len(path_x)+len(path_y))
    static.append([x_str,couple])

    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

def build_Ytilde(pos_args,size_args,basis):
    '''
    Build the Y tilde operator (corresponds to path of zs and plaquette of xs)

    args:
        nx,ny - coordinates of Y

        size_args (list) - Gives the size of the lattice in each direction

        basis (quspin user basis) - The user basis we use

    returns:
        LGT dual of Ising Ytilde operator

    notes:
        This is just iXtildeZtilde
    '''

    nx,ny = pos_args
    Nx,Ny = size_args

    static = []
    dynamic = []

    path_x = [indx(i,0,1,Ny) for i in range(nx+1)]
    path_y = [indx(nx,mody(j+1,Ny),0,Ny) for j in range(ny)]
    path = path_x+path_y

    h_plaq = [indx(nx,ny,0,Ny),indx(modx(nx+1,Nx),ny,1,Ny),indx(nx,mody(ny+1,Ny),0,Ny),indx(nx,ny,1,Ny)]

    couple = [[1.0j] + path + h_plaq]

    xz_str = "z" * (len(path_x)+len(path_y)) + "xxxx"
    static.append([xz_str,couple])

    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)

def build_Ztilde(pos_args,size_args,basis,bc="PBC"):
    '''
    Build the Z tilde operator (corresponds to plaquette of xs)

    args:
        nx,ny - coordinates of Z
        size_args (list) - Gives the size of the lattice in each direction
        basis (quspin user basis) - The user basis we use

    returns:
        LGT dual of Ising Ztilde operator
    '''

    nx,ny = pos_args
    Nx,Ny = size_args
    
    static = []
    dynamic = []
    
    if bc == "PBC":
        h_plaq = [[1,indx(nx,ny,0,Nx,Ny),indx(nx+1,ny,1,Nx,Ny),indx(nx,ny+1,0,Nx,Ny),indx(nx,ny,1,Nx,Ny)]]
    elif bc == "OBC":
        h_plaq = [[1,indx(nx,ny,0,Nx-1,Ny-1,bc="OBC"),indx(nx+1,ny,1,Nx-1,Ny-1,bc="OBC"),indx(nx,ny+1,0,Nx-1,Ny-1,bc="OBC"),indx(nx,ny,1,Nx-1,Ny-1,bc="OBC")]]
                
    static.append(["xxxx",h_plaq])
    
    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
def build_ZI_rotation_Z2(pos_args,size_args,theta,parity,basis,bc):
    '''
    Build the ZI rotation generator with Z2 operators

    args:
        pos_args (list) - list of two coordinates that specify which qubits
        the unitary acts on
        
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

    i_pos, j_pos = pos_args

    zti_Itj = build_Ztilde(i_pos,size_args,basis,bc).tocsc()
    Iti_ztj = build_Ztilde(j_pos,size_args,basis,bc).tocsc()
    
    H_rot = zti_Itj + parity*Iti_ztj
    
    R_theta = expm_multiply_parallel(H_rot,a=1j*theta/2)
    
    return R_theta

def build_YX_rotation_Z2(pos_args,size_args,theta,parity,basis,bc="PBC"):
    '''
    Build the YX rotation generator with Z2 operators

    args:
        pos_args (list) - list of two coordinates that specify which qubits
        the unitary acts on
        
        size_args (list) - Gives the size of the lattice in each direction
        
        theta (float) - The angle we rotate by
        
        parity (+- 1) - This specifies whether the gate acts on the even
        or odd parity sector
        
        basis (quspin user basis) - The user basis we use

    returns:
        R_theta (quspin expmultiplyparallel) - Rotation operator.
        Use R_theta.dot(psi) to apply this to a states

    notes:
        We don't store this as a matrix for efficiency
    '''

    i_coords, j_coords = pos_args
    Nx,Ny = size_args

    static = []
    dynamic = []
    
    # We now directly build the YiXj \pm XiYj operators
    
    #Gives links for placement of x,y,z ops in wilson line
    #0 if the y is on i, 1 if index is j
    yixj_path_z,yixj_path_y,yixj_path_x = link_path(i_coords,j_coords,Nx,Ny,0,bc=bc)
    yixj_path_total = yixj_path_z + [yixj_path_y] + yixj_path_x
    xiyj_path_z,xiyj_path_y,xiyj_path_x = link_path(i_coords,j_coords,Nx,Ny,1,bc=bc)
    xiyj_path_total = xiyj_path_z + [xiyj_path_y] + xiyj_path_x
    
    static,dynamics = [],[]
    
    yixj_path_total.insert(0,-1)
    xiyj_path_total.insert(0,-1*parity)
    h_yx = [yixj_path_total]
    h_xy = [xiyj_path_total]
    
    static.append(["z"*len(yixj_path_z)+"y"+"xxx",h_yx])
    static.append(["z"*len(xiyj_path_z)+"y"+"xxx",h_xy])

    H_rot = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS).tocsc()
    
    R_theta = expm_multiply_parallel(H_rot,a=1j*theta/2)
    
    return R_theta
    
#------------------------------------------------------------------------------ #
# This is for pairs OBC and for Ising Product                                   #
#------------------------------------------------------------------------------ #

def create_SU4_unitary(unitary_dict,pos_args,size_args,basis):
    '''
    This creates an SU(4) unitary in terms of Z2 operators.
    
    args:
        unitary_dict (dict) - Dictionary specifiying the decomposition of the unitary into
        two qubit paulis
        
        pos_args (list) - list of two coordinates that specify which qubits
        the unitary acts on
        
        size_args (list) - Gives the size of the lattice in each direction
        
        basis (quspin user basis) - The user basis we use
        
    returns:
        H_U (quspin.hamiltonian) - The hamiltonian that corresponds to the unitary
    '''
    i_coords, j_coords = pos_args
    ix,iy = i_coords
    jx,jy = j_coords
    Nx,Ny = size_args
        
    #Array that stores whether there will be a x,y,z operator on a Z2 link
    #After the entire operator is built this will be used to simplify things
    i_xz_graph = np.zeros((Nx*(Ny-1)+Ny*(Nx-1),2),dtype=int)
    j_xz_graph = np.zeros((Nx*(Ny-1)+Ny*(Nx-1),2),dtype=int)
    
    i_ribbon = [1]
    j_ribbon = [1]
    if i_coords == [0,0]:
        i_plaquette = [0,1,2,2*Ny]
    else:
        tail_i, neck_i, head_i = link_path([0,0],i_coords,Nx,Ny,1,bc="OBC")
        i_ribbon += tail_i+[neck_i]
        i_plaquette = [neck_i]+head_i

    if j_coords == [0,0]:
        j_plaquette = [0,1,2,2*Ny]
    else:
        tail_j, neck_j, head_j = link_path([0,0],j_coords,Nx,Ny,1,bc="OBC")
        j_ribbon += tail_j+[neck_j]
        j_plaquette = [neck_j]+head_j
    
    if ix == 0:
        i_indx = indx(ix,iy,1,Nx,Ny,bc="OBC")
        i_xz_graph[i_indx,1] = 1
    elif ix == Nx-2:
        i_indx = indx(ix+1,iy,1,Nx,Ny,bc="OBC")
        i_xz_graph[i_indx,1] = 1
    elif iy == 0:
        i_indx = indx(ix,iy,0,Nx,Ny,bc="OBC")
        i_xz_graph[i_indx,1] = 1
    elif iy == Nx-2:
        i_indx = indx(ix,iy+1,0,Nx,Ny,bc="OBC")
        i_xz_graph[i_indx,1] = 1
    else:
        i_xz_graph[i_ribbon,1] = 1

    i_xz_graph[i_plaquette,0] = 1
    
    if jx == 0:
        j_indx = indx(jx,jy,1,Nx,Ny,bc="OBC")
        j_xz_graph[j_indx,1] = 1
    elif jx == Nx-2:
        j_indx = indx(jx+1,jy,1,Nx,Ny,bc="OBC")
        j_xz_graph[j_indx,1] = 1
    elif jy == 0:
        j_indx = indx(jx,jy,0,Nx,Ny,bc="OBC")
        j_xz_graph[j_indx,1] = 1
    elif jy == Nx-2:
        j_indx = indx(jx,jy+1,0,Nx,Ny,bc="OBC")
        j_xz_graph[j_indx,1] = 1
    else:
        j_xz_graph[j_ribbon,1] = 1

    j_xz_graph[j_plaquette,0] = 1

    static, dynamic = [], []

    # II - this is a constant operator - can be added in after
    id_couple = [[unitary_dict["II"]]+list(range(Nx*(Ny-1)+Ny*(Nx-1)))]
    static.append(["I"*(Nx*(Ny-1)+Ny*(Nx-1)),id_couple])

    #Build Xi, Xj
    Xi_inds,Xj_inds = i_xz_graph[:,1],j_xz_graph[:,1]
    zi_inds, zj_inds = [i for i,z in enumerate(Xi_inds) if z == 1], [i for i,z in enumerate(Xj_inds) if z == 1]

    xi_couple,xj_couple = [[unitary_dict["xI"]]+zi_inds],[[unitary_dict["Ix"]]+zj_inds]
    static.append(["z"*len(zi_inds),xi_couple])
    static.append(["z"*len(zj_inds),xj_couple])
    
    #Build Zi, Zj
    Zi_inds,Zj_inds = i_xz_graph[:,0],j_xz_graph[:,0]
    xi_inds, xj_inds = [i for i,x in enumerate(Zi_inds) if x == 1], [i for i,x in enumerate(Zj_inds) if x == 1]

    zi_couple,zj_couple = [[unitary_dict["zI"]]+xi_inds],[[unitary_dict["Iz"]]+xj_inds]
    static.append(["x"*len(xi_inds),zi_couple])
    static.append(["x"*len(xj_inds),zj_couple])
    
    #Build Yi, Yj - find overlap, should only be 1
    #coefficient is 1 since Y=iXZ and XZ ~ prod z prod x (i.e. i * i = -1; iy=zx)
    yi_inds,yj_inds = [i for i in range(len(Xi_inds)) if Xi_inds[i] & Zi_inds[i]],[i for i in range(len(Xj_inds)) if Xj_inds[i] & Zj_inds[i]]
    zi_inds1,zj_inds1 = [i for i in range(len(Xi_inds)) if Xi_inds[i] == 1 and i not in yi_inds],[i for i in range(len(Xj_inds)) if Xj_inds[i] == 1 and i not in yj_inds]
    xi_inds1,xj_inds1 = [i for i in range(len(Zi_inds)) if Zi_inds[i] == 1 and i not in yi_inds],[i for i in range(len(Zj_inds)) if Zj_inds[i] == 1 and i not in yj_inds]

    coeffY = -1.0
    yi_couple,yj_couple = [[coeffY*unitary_dict["yI"]]+xi_inds1+yi_inds+zi_inds1],[[coeffY*unitary_dict["Iy"]]+xj_inds1+yj_inds+zj_inds1]
    static.append(["x"*len(xi_inds1)+"y"*len(yi_inds)+"z"*len(zi_inds1),yi_couple])
    static.append(["x"*len(xj_inds1)+"y"*len(yj_inds)+"z"*len(zj_inds1),yj_couple])

    #Build XiZj,ZiXj
    #Coefficient will be flipped between two cases, i.e. prod z prod x / prod x prod z (i.e. i^overlap/-i^overlap)
    yij_inds,yji_inds = [i for i in range(len(Xi_inds)) if Xi_inds[i] & Zj_inds[i]],[i for i in range(len(Xi_inds)) if Xj_inds[i] & Zi_inds[i]]
    zij_inds,zji_inds = [i for i in range(len(Xi_inds)) if Xi_inds[i] == 1 and i not in yij_inds],[i for i in range(len(Xj_inds)) if Xj_inds[i] == 1 and i not in yji_inds]
    xij_inds,xji_inds = [i for i in range(len(Zj_inds)) if Zj_inds[i] == 1 and i not in yij_inds],[i for i in range(len(Zi_inds)) if Zi_inds[i] == 1 and i not in yji_inds]

    coeffXZ = (1.0j)**len(yij_inds)
    coeffZX = (-1.0j)**len(yji_inds)
    xizj_couple,zixj_couple = [[coeffXZ*unitary_dict["xz"]]+xij_inds+yij_inds+zij_inds],[[coeffZX*unitary_dict["zx"]]+xji_inds+yji_inds+zji_inds]
    static.append(["x"*len(xij_inds)+"y"*len(yij_inds)+"z"*len(zij_inds),xizj_couple])
    static.append(["x"*len(xji_inds)+"y"*len(yji_inds)+"z"*len(zji_inds),zixj_couple])
    
    #Build XiYj=iXiXjZj,YiXj=-iZiXiXj
    # First cancel zs, then convert into ys
    # coefficient will have i from Y=iXZ, then prod z prod x/prod x prod z (i^overlap/-i^overlap)
    zij_inds1 = [i for i in range(len(Xi_inds)) if (i in zj_inds and i not in zi_inds) or (i not in zj_inds and i in zi_inds)]
    xij_inds1 = xj_inds
    yij_inds1 = [i for i in range(len(Xi_inds)) if i in xij_inds1 and i in zij_inds1]
    zij_inds2 = [i for i in range(len(Xi_inds)) if i in zij_inds1 and i not in yij_inds1]
    xij_inds2 = [i for i in range(len(Xi_inds)) if i in xij_inds1 and i not in yij_inds1]
    
    zji_inds1 = [i for i in range(len(Xi_inds)) if (i in zi_inds and i not in zj_inds) or (i not in zi_inds and i in zj_inds)]
    xji_inds1 = xi_inds
    yji_inds1 = [i for i in range(len(Xi_inds)) if i in xji_inds1 and i in zji_inds1]
    zji_inds2 = [i for i in range(len(Xi_inds)) if i in zji_inds1 and i not in yji_inds1]
    xji_inds2 = [i for i in range(len(Xi_inds)) if i in xji_inds1 and i not in yji_inds1]

    coeffXY = 1.0j*(1.0j)**len(yij_inds1)
    coeffYX = -1.0j*(-1.0j)**len(yji_inds1)
    xiyj_couple,yixj_couple = [[coeffXY*unitary_dict["xy"]]+xij_inds2+yij_inds1+zij_inds2],[[coeffYX*unitary_dict["yx"]]+xji_inds2+yji_inds1+zji_inds2]
    static.append(["x"*len(xij_inds2)+"y"*len(yij_inds1)+"z"*len(zij_inds2),xiyj_couple])
    static.append(["x"*len(xji_inds2)+"y"*len(yji_inds1)+"z"*len(zji_inds2),yixj_couple])

    #Build ZiYj=-i*ZiZjXj, YiZj=iXiZiZj
    #First cancel xs, then convert into ys
    # coefficient will have -i/i from Y=iXZ, then prod x prod z/prod z prod x (-i^overlap/i^overlap)
    xij_inds3 = [i for i in range(len(Xi_inds)) if (i in xj_inds and i not in xi_inds) or (i not in xj_inds and i in xi_inds)]
    zij_inds3 = zj_inds
    yij_inds3 = [i for i in range(len(Xi_inds)) if i in xij_inds3 and i in zij_inds3]
    zij_inds4 = [i for i in range(len(Xi_inds)) if i in zij_inds3 and i not in yij_inds3]
    xij_inds4 = [i for i in range(len(Xi_inds)) if i in xij_inds3 and i not in yij_inds3]
    
    xji_inds3 = [i for i in range(len(Xi_inds)) if (i in xi_inds and i not in xj_inds) or (i not in xi_inds and i in xj_inds)]
    zji_inds3 = zi_inds
    yji_inds3 = [i for i in range(len(Xi_inds)) if i in xji_inds3 and i in zji_inds3]
    zji_inds4 = [i for i in range(len(Xi_inds)) if i in zji_inds3 and i not in yji_inds3]
    xji_inds4 = [i for i in range(len(Xi_inds)) if i in xji_inds3 and i not in yji_inds3]

    coeffZY = -1.0j*(-1.0j)**len(yij_inds3)
    coeffYZ = 1.0j*(1.0j)**len(yji_inds3)
    ziyj_couple,yizj_couple = [[coeffZY*unitary_dict["zy"]]+xij_inds4+yij_inds3+zij_inds4],[[coeffYZ*unitary_dict["yz"]]+xji_inds4+yji_inds3+zji_inds4]
    static.append(["x"*len(xij_inds4)+"y"*len(yij_inds3)+"z"*len(zij_inds4),ziyj_couple])
    static.append(["x"*len(xji_inds4)+"y"*len(yji_inds3)+"z"*len(zji_inds4),yizj_couple])

    #Build XiXj
    xixj_couple = [[unitary_dict["xx"]]+zij_inds1]
    static.append(["z"*len(zij_inds1),xixj_couple])

    #Build YiYj=i*i * XiZiXjZj
    #Get cancelled xs and zs and convert into ys
    # coefficient will have i^2 from YY=-XZXZ, then prod x prod z/prod z prod x (i^overlap/-i^overlap)
    yij_inds4 = [i for i in range(len(Xi_inds)) if i in xij_inds3 and i in zij_inds1]
    zij_inds5 = [i for i in range(len(Xi_inds)) if i in zij_inds1 and i not in yij_inds4]
    xij_inds5 = [i for i in range(len(Xi_inds)) if i in xij_inds3 and i not in yij_inds4]

    coeffYY = -1.0*(1.0j)**len(yij_inds4)#unitary
    yiyj_couple = [[coeffYY*unitary_dict["yy"]]+xij_inds5+yij_inds4+zij_inds5]
    static.append(["x"*len(xij_inds5)+"y"*len(yij_inds4)+"z"*len(zij_inds5),yiyj_couple])

    #Build ZiZj
    zizj_couple = [[unitary_dict["zz"]]+xij_inds3]
    static.append(["x"*len(xij_inds3),zizj_couple])

    H_U = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
    return H_U.tocsc()
    
#------------------------------------------------------------------------------ #
# This is for dual Product                                                      #
#------------------------------------------------------------------------------ #

def create_dual_yz(coords,size_args,basis):
    '''
    This creates the z and y ising operators, necessary for a single qubit 2-design
    
    args:
        coords (list[int]) - gives the ising coordinates
        
        size_args (list) - Gives the size of the lattice in each direction
        
        basis (quspin user basis) - The user basis we use
        
    returns:
        H_list (list[quspin.hamiltonian]) - List of all the single qubit ising operators
    '''
    
    ix,iy = coords
    Nx,Ny = size_args
        
    #Array that stores whether there will be a x,z operator on a Z2 link
    #After the entire operator is built this will be used to simplify things
    i_xz_graph = np.zeros((2*Nx*Ny+1,2),dtype=int)
    
    #Add base position
    i_ribbon = [1]
    if coords == [0,0]:
        i_plaquette = [0,1,2,2*Ny+1]
    else:
        tail_i, neck_i, head_i = link_path([0,0],coords,Nx,Ny,1,bc="PBC",ancilla=True)
        i_ribbon += tail_i+[neck_i]
        i_plaquette = [neck_i]+head_i
    
    i_xz_graph[i_ribbon,1] = 1
    i_xz_graph[i_plaquette,0] = 1

    staticy,staticz, dynamic = [], [], []
    
    #Build Zi
    Zi_inds = i_xz_graph[:,0]
    xi_inds = [i for i,x in enumerate(Zi_inds) if x == 1]

    zi_couple = [[1]+xi_inds]
    staticz.append(["x"*len(xi_inds),zi_couple])
    
    #Build Yi - find overlap, should only be 1
    #coefficient is -1 since Y=iXZ and XZ ~ prod z prod x (i.e. i * i = -1; iy=zx)
    Xi_inds = i_xz_graph[:,1]
    yi_inds = [i for i in range(len(Xi_inds)) if Xi_inds[i] & Zi_inds[i]]
    zi_inds1 = [i for i in range(len(Xi_inds)) if Xi_inds[i] == 1 and i not in yi_inds]
    xi_inds1 = [i for i in range(len(Zi_inds)) if Zi_inds[i] == 1 and i not in yi_inds]

    coeffY = -1.0
    yi_couple = [[coeffY]+xi_inds1+yi_inds+zi_inds1]
    staticy.append(["x"*len(xi_inds1)+"y"*len(yi_inds)+"z"*len(zi_inds1),yi_couple])

    H_y = hamiltonian(staticy, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    H_z = hamiltonian(staticz, dynamic, basis=basis, dtype=np.complex128,**NO_CHECKS)
    
    return H_y, H_z
