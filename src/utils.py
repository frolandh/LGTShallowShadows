# Define auxiliary functions
import numpy as np

#PBC x direction
def modx(n, Nx):
    if n < 0:
        return int(n + Nx)
    elif n > Nx - 1:
        return int(n - Nx)
    else:
        return n

#PBC y direction
def mody(n, Ny):
    if n < 0:
        return int(n + Ny)
    elif n > Ny - 1:
        return int(n - Ny)
    else:
        return n
        
#Only to be used for the Z2 theory
def indx(nx, ny, i, Nx, Ny,bc="PBC",ancilla=False):
    if bc == "PBC":
        if(ancilla and nx==Nx and ny==0 and i == 1):
            return 2*Nx*Ny
        return 2 * (np.mod(ny,Ny) + np.mod(nx,Nx) * Ny) + i
    elif bc == "OBC":
        return indx_obc(nx,ny,i,Nx,Ny)
    else:
        return "This is not a boundary condition"
        
def indx_obc(nx, ny, i, Nx, Ny):
    if nx == Nx-1:
        assert i == 1, "Must chooose different link orientation"
        return (Nx-1)*(2*Ny-1) + ny
    elif ny == Ny-1:
        assert i == 0, "Must chooose different link orientation"
        return nx*(2*Ny-1) + 2*(Ny-1)
    else:
        return 2*ny + nx*(2*Ny-1) + i
        
def dual2lgt(i,size_args,bc="PBC"):
    '''
    Converts an index i on the Ising side to the corresponding Z2 coordinates

    args:
        i (int) - Ising index (0<= i < Nx*Ny/Nx(N-1)+Nx(Nx-1))

        size_args (list) - Nx and Ny extent of system
    
    returns:
        [nx,ny] (list) - x and y coordinates for the gauge theory
    '''
    
    if bc == "PBC":
        _,Ny = size_args
    elif bc == "OBC":
        _,Ny = size_args
        Ny -= 1
    
    nx,ny = i//(Ny),i%(Ny)
        
    return [nx,ny]
        

def indx_d(nx,ny,Ny):
    '''
    For use in the ising theory
    '''
    
    return nx*Ny+ny

#Convert integer s to binary string
def to_binary(s, N):
    return format(s, '0' + str(N) + 'b')

##################################
# Functions for Operator Mapping #
##################################

def link_path(i_coords, j_coords,Nx,Ny,head,bc="PBC",ancilla=False):
    '''
    Builds the wilson line between two plaquettes between

    args:
        i_coords (list) - coordinates of plaq 1
        
        j_coords (list) - coordinates of plaq 2

        Nx,Ny (list) - Ssize of the system

        head (quspin user basis) - Says which index (i or j) the xxxy operators should appear.
        Associated with whichever the Y index is in the YX/XY operator
        
        ancilla (bool) - Accounts for addition of an ancilla; to be used with dual pairs

    returns:
        path_z (list) - link indices on Z2 associated with the z ribbon

        path_y (int) - index associated with the y operator on the neck
        
        path_x (list) - 3 indices associated with x operators on head of the line
    '''
    
    def remove_second_appearances(lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result
    
    def step_range(a, b,c):
        step = 1 if b > a else -1
        return range(a + step, b + c*step, step)

    jx, jy = j_coords
    ix, iy = i_coords
    
    if ix <= jx and iy <= jy:
        if ix < jx:
            path = [indx(jx,jy,1, Nx, Ny,bc,ancilla=ancilla)]
        else:
            path = []
            
        path.extend(indx(x, jy, 1, Nx, Ny,bc,ancilla=ancilla) for x in step_range(jx, ix,0))
        
        if iy != jy:
            path.append(indx(ix, jy, 0, Nx, Ny,bc,ancilla=ancilla))
            
        path.extend(indx(ix, y, 0, Nx, Ny,bc,ancilla=ancilla) for y in step_range(jy, iy,0))
        
    elif ix <= jx and iy >= jy:
        if ix < jx:
            path = [indx(jx,jy,1, Nx, Ny,bc,ancilla=ancilla)]
        else:
            path = []
            
        path.extend(indx(x, jy, 1, Nx, Ny,bc,ancilla=ancilla) for x in step_range(jx, ix,0))
        
        if iy != jy:
            path.append(indx(ix, jy+1, 0, Nx, Ny,bc,ancilla=ancilla))
            
        path.extend(indx(ix, y, 0, Nx, Ny,bc,ancilla=ancilla) for y in step_range(jy, iy,1))
        
    elif ix >= jx and iy <= jy:
        if ix > jx:
            path = [indx(jx+1,jy,1, Nx, Ny,bc,ancilla=ancilla)]
        else:
            path = []
            
        path.extend(indx(x, jy, 1, Nx, Ny,bc,ancilla=ancilla) for x in step_range(jx+1, ix,1))
        
        if iy != jy:
            path.append(indx(ix, jy, 0, Nx, Ny,bc,ancilla=ancilla))
            path.append(indx(ix, jy, 1, Nx, Ny,bc,ancilla=ancilla))
            
        path.extend(indx(ix, y, 0, Nx, Ny,bc,ancilla=ancilla) for y in step_range(jy, iy,0))
        
    elif ix >= jx and iy >= jy:
        if ix > jx:
            path = [indx(jx+1,jy,1 ,Nx,Ny,bc,ancilla=ancilla)]
        else:
            path = []
            
        path.extend(indx(x, jy, 1, Nx, Ny,bc,ancilla=ancilla) for x in step_range(jx, ix,0))
        
        if iy != jy:
            path.append(indx(ix, jy, 1, Nx, Ny,bc,ancilla=ancilla))
            
        path.extend(indx(ix, y, 0, Nx, Ny,bc,ancilla=ancilla) for y in step_range(jy, iy,1))

    path = remove_second_appearances(path)

    path_x = []
    if head == 0:
        path_z = path[:-1]
        path_y = path[-1]
        
        path_x.append(indx(ix, iy, 0, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(ix, iy, 1, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(ix+1, iy, 1, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(ix, iy+1, 0, Nx, Ny,bc,ancilla=ancilla))
    else:
        path_z = path[1:]
        path_y = path[0]
        
        path_x.append(indx(jx, jy, 0, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(jx, jy, 1, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(jx+1, jy, 1, Nx, Ny,bc,ancilla=ancilla))
        path_x.append(indx(jx, jy+1, 0, Nx, Ny,bc,ancilla=ancilla))

    path_x = [l for l in path_x if l != path_y]

    return path_z,path_y,path_x
