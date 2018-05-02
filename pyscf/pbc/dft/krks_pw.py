#!/usr/bin/env python
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu <jyu5@caltech.edu>

'''
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling using a plane wave basis set

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

#Get Vxc from LDA approximation via formula: Vxc = rho^(1/3)*constant term
def get_vxc(rho):
   xalpha = 2./3.
   return (-1.5)*xalpha*(3.*rho/np.pi)**(1./3.)


#Get XC energy contribution from LDA func for use in get_energies
def get_exc(rho):
   return (-3./4.)*((3./np.pi)**(1./3.))*(rho**(1./3.))


#Get total xc and coulomb energies
def get_energies(self,cell,kpts,largeind,rhog,v_excstore):
   #calculate ecoul and exc
   Gd2 = self.pw_grid_params[5]
   e_coul_store = np.zeros([len(kpts)],dtype='float64')
   e_xc_store = np.zeros([len(kpts)],dtype='float64')

   for x in range(0,len(kpts)):
      e_coul_store[x] = get_e_vh(largeind,rhog,Gd2,cell.vol)
      for y in range(len(v_excstore[x])):
         e_xc_store[x] += v_excstore[x][y]*rhog[y]*cell.vol
   return e_coul_store, e_xc_store


#Get Coulomb Potential via formula Vh = 4*Pi*rho(G)^2/G^2
def get_vh(index,rhog,g2):
   return 4*np.pi*rhog[index]/g2[index]


#Get Coulomb energy contribution for use in get_energies
def get_e_vh(index,rhog,g2,v):
   e_vh =v*(2*np.pi)*rhog[index]*np.conjugate(rhog[index])/g2[index]

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


#Get the number of total occupied bands
def get_bands(mocc,kpts):
   nbands=0
   for x in range(len(mocc)):
      for y in range(len(mocc[x])):
         if mocc[x][y]!=0:
            nbands+=mocc[x][y]

   return int(nbands/(2*len(kpts)))


#Calculate the weighted density from eigvecs
def get_density(self,cell,kpts,eigvec,nbands):
   
   grid_dim = self.pw_grid_params[2]
   rho_tot = np.zeros(grid_dim,dtype='float64')
   mill_Gd = self.pw_grid_params[11]
   npw = self.pw_grid_params[15]
   indgk = self.pw_grid_params[10]

   for nk in range(len(kpts)):
      for pp in range(nbands):
            aux=np.zeros(grid_dim,dtype='complex128')
            for tt in range(npw[nk]):
               ik=indgk[nk][tt]
               aux[get_mill(ik,mill_Gd,grid_dim)]=eigvec[nk][tt,pp]
            aux=(1./np.sqrt(cell.vol))*np.fft.fftn(aux)
            rho_tot+=(2./len(kpts))*np.absolute(aux)**2.

   return rho_tot


#Get vectorized coulomb and xc potentials vg = vh+vxc
#returns vg, v_excstore (structure necessary to compute xc Energy), largeind 
#(indices of nonzero G vectors, and rhog (the density in reciprocal space)
def get_vg(self,cell,kpts,rho_tot):
   Gd_ind = self.pw_grid_params[4]
   Gd2 = self.pw_grid_params[5]
   indg = self.pw_grid_params[9]
   gs = self.pw_grid_params[0]
   indgk = self.pw_grid_params[10]
   grid_dim = self.pw_grid_params[2]
   g2_cutoff = 1e-8   #To catch divergent G=0 term we set a g2 cutoff
   mill_Gd = self.pw_grid_params[11]

   #Get vg=vxc+j vectorized
   vg = np.zeros(len(Gd_ind),dtype='float64')
   vgstore = np.zeros([len(kpts),len(Gd2)],dtype='float64')
   v_exc = np.zeros(len(Gd_ind),dtype='float64')
   v_excstore = np.zeros([len(kpts),len(Gd2)],dtype='float64')
   rhog = np.zeros(len(Gd_ind),dtype='float64')

   for x in range(0, len(kpts)):
      vxc = get_vxc(rho_tot)
      exc = get_exc(rho_tot)
      recip_vxc = np.fft.ifftn(vxc)
      recip_exc = np.fft.ifftn(exc)
      temp_rhog = np.fft.ifftn(rho_tot)

      for ng in range(0,len(Gd_ind)):
          vg[ng] = np.real(recip_vxc[get_mill(ng, mill_Gd, grid_dim)])
          rhog[ng] = np.real(temp_rhog[get_mill(ng, mill_Gd, grid_dim)])
          v_exc[ng] = np.real(recip_exc[get_mill(ng,mill_Gd,grid_dim)])

      #Ensure that zero vector does not contribute
      largeind = Gd2>g2_cutoff
      vg[largeind] += get_vh(largeind,rhog,Gd2)
      vgstore[x] = vg
      v_excstore[x] = v_exc

   return vgstore,v_excstore,largeind,rhog


#Build Veff matrix to be added to fock matrix
def fill_veff(self, cell, kpts, vgstore):
   mill_Gd = self.pw_grid_params[11]
   npw = self.pw_grid_params[15]
   indg = self.pw_grid_params[9]
   indgk = self.pw_grid_params[10]
   gs = self.pw_grid_params[0]

   veff = []
   for y in range(0, len(kpts)):
      fill = np.zeros((npw[y],npw[y]),dtype='complex128')
      gkind = indgk[y,:npw[y]]
      for aa in range(npw[y]):
         ik = indgk[y][aa]
         gdiff = mill_Gd[ik]-mill_Gd[gkind[aa:]]+np.array(gs)
         inds = indg[gdiff.T.tolist()]
         fill[aa,aa:]=vgstore[y][inds]
      veff.append(fill)

   return veff


#Return Veff matrix to SCF kernel
def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional
    '''

    import time

    #retrieve grids
    if cell is None: cell = self.cell
    if kpts is None: kpts = self.kpts
    
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
        mocc = getattr(dm,'mo_occ')
        nbands = get_bands(mocc,kpts)
        rho_tot = get_density(self,cell,kpts,eigvec,nbands)

    #Get vectorized vxc + vh 
    vgstore, v_excstore, largeind, rhog = get_vg(self,cell,kpts,rho_tot) 
    
    #Fill up [nao,nao] matrix with vg per kpt 
    veff = fill_veff(self,cell,kpts,vgstore)

    #Get hartree and xc energies
    e_coul_store, e_xc_store = get_energies(self,cell,kpts,largeind,rhog,v_excstore)

    #Tag energies to veff and return
    veff = np.asarray(veff)
    veff = lib.tag_array(veff, ecoul=e_coul_store[0], exc=e_xc_store[0], vj=None, vk=None) 
    return veff


class KRKS(khf_pw.KSCF_PW):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    #def __init__(self, cell, h, b, v, kpts=np.zeros((1,3))):
    def __init__(self, cell, kpts=np.zeros((1,3))):
        khf_pw.KSCF_PW.__init__(self, cell, cell.kpts)
        self.xc = 'LDA,VWN'
        
        self.pw_grid_params=kpw_helper.return_grids(cell,cell.kpts,cell.a,cell.reciprocal_vectors(),cell.vol,cell.ke_cutoff)

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

        kpts = self.kpts
        mocc = getattr(dm_kpts,'mo_occ')
        nbands = get_bands(mocc,kpts)
        eigvec = np.array(getattr(dm_kpts,'mo_coeff'))
        weight = 1./len(h1e_kpts)
        e1 = 0.0

        #Convert h1e from upper triangular, compute E1, and convert back
        for i in range(len(h1e_kpts)):
            temp = h1e_kpts[i].diagonal()
            h1e_kpts[i]=((h1e_kpts[i]+h1e_kpts[i].T))
            np.fill_diagonal(h1e_kpts[i],temp) 

        for x in range(len(h1e_kpts)):  
            for y in range(nbands):
               e1+=2*weight*np.dot(np.conj(eigvec[x][:,y]),np.dot(h1e_kpts[x],eigvec[x][:,y])) 
   
        for i in range(len(h1e_kpts)):
            h1e_kpts[i]=np.triu(h1e_kpts[i])

        #Compute total energy 
        tot_e = e1 + vhf.ecoul + vhf.exc

        print 'ecoul: ',vhf.ecoul
        print 'exc: ',vhf.exc
        print 'e1: ',np.real(e1)

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
