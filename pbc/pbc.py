import numpy as np
import scipy.linalg
import math.cexp
import math.pi as pi

class Cell(pyscf.Mole)
    def __init__(self, mol, h):
        pyscf.Mole.__init__(self, mol)
        # h in HM (Eq. (3.1))
        self.h=h
        
def get_gv(gs):
    '''
    integer cube of indices, -gs...gs along each direction
    gs: [gsx, gsy, gsz]
    
    Returns 
        [[-gsx, -gsx+1, ..., gsx],
         [-gsy  -gsy+1, ..., gsy],
         [-gsz, -gsz+1, ..., gsz]]
    '''
    ngs=2*gs+1
    gv=np.ndindex(2*gs+1))
    gv-=np.array([gsx,gsy,gsz])
    return gv.T

def get_G(h, gv):
    '''
    cube of G vectors (Eq. (3.8), HM),

    Returns 
        np.array([3, ngs], np.complex128)
    '''
    invh=scipy.linalg.inv(h)
    G=2*pi*np.dot(h,gs)
    return G

def get_SI(cell, G):
    '''
    structure factor (Eq. (3.34), HM)

    Returns 
        np.array([natm, ngs], np.complex128)
    '''
    ngs=G.shape[1]
    SI=np.empty([natm, ngs], np.complex128)

    for ia in range(cell.natm):
        SI[ia,:]=math.cexp(-1j*np.dot(G.T, cell.atom_coord(ia)))
    return SI

def ewald_real_space(cell, SI, ewrc, ewcut):
    '''
    Real-space Ewald sum 

    This implements Eq. (3.46), Eq. (3.47) in HM, but 
    has only a single Ewald length (Rc is same for all ions)
    and includes the additional second term in the last line of Eq. (F.5) in M,
    which corrects the self-interaction of an ion with the uniform background

    ewrc : Ewald length (Rc param in HM)
    ewcut : [ewcutx, ewcuty, ewcutz]

    Returns
        float
    '''
    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    ewovrl = 0.
    for ix, iy, iz in ndenumerate(ewcut):
        L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]

        # prime in summation to avoid self-interaction in unit cell
        for ia in range(cell.natm):
            qi = chargs[ia]
            ri = coords[ia]

            if ix == 0 and iy == 0 and iz == 0:
                for ja in range(ia):
                    qj = chargs[ja]
                    rj = coords[ja]
                    r = np.linalg.norm(ri-rj)
                    ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
            else:
                for ja in range(cell.natm):
                    qj=chargs[ja]
                    rj=coords[ja]

                    r=np.linalg.norm(ri-rj+L)
                    ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
        
    ewself = 1./(math.sqrt(2*pi) * ewrc) * np.dot(chargs,charge)
    
    # 2nd term in last line of Eq. (F.5), M
    eta = 1./(sqrt(2*ewrc)) # conversion to (F.5) notation
    ewself += 1./2 * np.sum(chargs)**2 * pi / (eta**2 * cell.volume)

    return ewovrl - ewself
    


    


