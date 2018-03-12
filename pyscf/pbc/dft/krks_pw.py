#!/usr/bin/env python
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''

import kpw_helper
import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.dft import rks
from pyscf.pbc.scf import khf_pw  

#Get Vxc from LDA approximation via formula
# Vxc = rho^(1/3)*constant term
def get_vxc(rho):
   xalpha = 2./3.
   return (-1.5)*xalpha*(3.*rho/np.pi)**(1./3.)

def get_exc(rho):
   return (-3./4.)*((3./np.pi)**(1./3.))*(rho**(4./3.))

#Get Coulomb Potential via formula Vh = 4*Pi*rho(G)^2/G^2
def get_vh(index,rhog,g2):
   return 4*np.pi*rhog[index]/g2[index]

def get_e_vh(index,rhog,g2,v):
   #e_vh = (2*np.pi/v)*rhog[index]*np.conjugate(rhog[index])/g2[index]
   e_vh = (2*np.pi)*rhog[index]*np.conjugate(rhog[index])/g2[index] 

   return np.sum(e_vh)

#Wrapper that returns the correct miller indices per G
def get_mill(x1,mill_Gd,grid_dim):
   index1=mill_Gd[x1][0]
   index2=mill_Gd[x1][1]
   index3=mill_Gd[x1][2]

   if index1<0:
      index1=index1+grid_dim[0]
   if index2<0:
      index2=index2+grid_dim[1]
   if index3<0:
      index3=index3+grid_dim[2]

   return index1, index2, index3


def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional
    '''

    import time

    #retrieve grids
    if cell is None: cell = self.cell
    if kpts is None: kpts = self.kpts
    
    indg=self.pw_grid_params[9]
    gs = self.pw_grid_params[0]
    gridsize=len(self.pw_grid_params[7])
    v = self.pw_grid_params[12]
    indgk = self.pw_grid_params[10]
    mill_Gd = self.pw_grid_params[11]
    grid_dim = self.pw_grid_params[2]
    Gd_ind = self.pw_grid_params[4]
    g2_cutoff=1e-8
    Gd2 = self.pw_grid_params[5]
    npw = self.pw_grid_params[15]

    if isinstance(kpts[0], (np.ndarray,np.generic)) is False:
        kpts = [kpts]

    #return zero matrix if on first iteration (i.e. no mocoeff yet)
    if hasattr(dm,'mo_coeff') is False:
        veff = []
        for x in range(len(kpts)):
            temp = np.zeros([npw[x],npw[x]])
            temp = lib.tag_array(temp, ecoul=0, exc=0, vj=None, vk=None)
            veff.append(temp)
        return veff

    #else generate density from mocoeff, [nkpts,[grid_dim]] for both this and last iteration
    else:
        eigvec = np.array(getattr(dm,'mo_coeff'))
        #nbands = np.count_nonzero(np.array(getattr(dm,'mo_occ')))
        #nbands = nbands/(len(kpts))**(1./3.)

        
        mocc = getattr(dm,'mo_occ')
        nbands=0
        for x in range(len(mocc)):
            for y in range(len(mocc[x])):
               if mocc[x][y]!=0:
                  nbands+=mocc[x][y]

        nbands = nbands/(2*len(kpts))
        
        #error_avoid = np.zeros(15)
        rhotot=np.empty(grid_dim,dtype='float64')
        rho_tot = np.zeros(grid_dim,dtype='float64')

        for nk in range(len(kpts)):
            for pp in range(4):
                aux=np.zeros(grid_dim,dtype='complex128')
                for tt in range(npw[nk]):
                    ik=indgk[nk][tt]
                    aux[get_mill(ik,mill_Gd,grid_dim)]=eigvec[nk][tt,pp]
                aux=(1./np.sqrt(v))*np.fft.fftn(aux)
                rho_tot+=(2./len(kpts))*np.absolute(aux)**2.


    #Get vg=vxc+j vectorized
    vg=np.zeros(len(Gd_ind),dtype='float64')
    vgstore = np.zeros([len(kpts),len(Gd2)],dtype='float64')
    v_exc = np.zeros(len(Gd_ind),dtype='float64')
    v_excstore = np.zeros([len(kpts),len(Gd2)],dtype='float64')
    rhog = np.zeros(len(Gd_ind),dtype='float64')
    e_coul_store = np.zeros([len(kpts)],dtype='float64')
    e_xc_store = np.zeros([len(kpts)],dtype='float64')

    for x in range(0, len(kpts)):
        vxc=get_vxc(rho_tot)
        exc = get_exc(rho_tot)
        recip_vxc=np.fft.ifftn(vxc)
        recip_exc=np.fft.ifftn(exc)
        temp_rhog=np.fft.ifftn(rho_tot)

        for ng in range(0,len(Gd_ind)):
            vg[ng]=np.real(recip_vxc[get_mill(ng, mill_Gd, grid_dim)])
            rhog[ng]=np.real(temp_rhog[get_mill(ng, mill_Gd, grid_dim)])
            v_exc[ng]=np.real(recip_exc[get_mill(ng,mill_Gd,grid_dim)])

        #Ensure that zero vector does not contribute
        largeind=Gd2>g2_cutoff
        vg[largeind]+=get_vh(largeind,rhog,Gd2)
        vgstore[x]=vg
        v_excstore[x]=v_exc
    

    #fill up [nao,nao] matrix with vg per kpt and return veff
    veff = []
    for y in range(0, len(kpts)):
        fill = np.zeros((npw[y],npw[y]),dtype='complex128')
        gkind=indgk[y,:npw[y]]
        for aa in range(npw[y]):
            ik = indgk[y][aa]
            gdiff = mill_Gd[ik]-mill_Gd[gkind[aa:]]+np.array(gs)
            inds = indg[gdiff.T.tolist()]
            fill[aa,aa:]=vgstore[y][inds]
        veff.append(fill)

    #calculate ecoul and exc
    for x in range(0,len(kpts)):
       e_coul_store[x] = get_e_vh(largeind,rhog,Gd2,v)
       for y in range(len(v_excstore[x])):
           e_xc_store[x] += v_excstore[x][y]   


    #Temporarily tag veff with ecoul and exc to avoid error in total_energy comput for HF
    #Needs to be fixed with proper total energy computation later
    veff = np.asarray(veff)
    veff = lib.tag_array(veff, ecoul=e_coul_store[0], exc=e_xc_store[0], vj=None, vk=None) 
    return veff


class KRKS(khf_pw.KSCF_PW):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, h, b, v, kpts=np.zeros((1,3))):
        khf_pw.KSCF_PW.__init__(self, cell, kpts)
        self.xc = 'LDA,VWN'
         
        self.pw_grid_params=kpw_helper.return_grids(cell,kpts,h,b,v)

        self.grids = gen_grid.UniformGrids(cell)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        # Note Do not refer to .with_df._numint because gs/coords may be different
        self._numint = numint._KNumInt(kpts)
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        khf.KRHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        if hasattr(dm_kpts,'mo_coeff') is False:
            return[0] 

        nbands = np.count_nonzero(np.array(getattr(dm_kpts,'mo_occ')))/2

        print 'nbands',nbands   

        eigvec = np.array(getattr(dm_kpts,'mo_coeff'))

        weight = 1./len(h1e_kpts)
        #e1 = weight * np.einsum('kij,kji', h1e_kpts, dm_kpts).real

        e1 = 0
      
        for x in range(len(h1e_kpts)):  
            for y in range(nbands):
                for m1 in range(len(eigvec[x][y])):
                    for m2 in range(len(eigvec[x][y])):
                        e1+=(weight**2)*np.conj(eigvec[x][y,m1])*h1e_kpts[x][m1,m2]*eigvec[x][y,m2] 
   
        tot_e = e1 + vhf.ecoul + vhf.exc

        print 'ecoul: ',vhf.ecoul
        print 'exc: ',vhf.exc
        print 'e1: ',e1

        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
        return tot_e, vhf.ecoul + vhf.exc

    define_xc_ = rks.define_xc_

    density_fit = rks._patch_df_beckegrids(khf.KRHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(khf.KRHF.mix_density_fit)


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    mf = KRKS(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
