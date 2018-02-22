#!/usr/bin/env python
#
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point 

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''

import pw_helper
import time
import numpy 
import pyscf.dft
from pyscf.pbc.scf import hf_pw as pbchf_pw
from pyscf.pbc.scf import hf as pbchf
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


#Get Vxc from LDA approximation via formula
# Vxc = rho^(1/3)*constant term
def get_vxc(rho):
   xalpha = 2./3.
   return (-1.5)*xalpha*(3.*rho/numpy.pi)**(1./3.)


#Get Coulomb Potential via formula Vh = 4*Pi*rho(G)^2/G^2
def get_vh(index,rhog,g2):
   return 4*numpy.pi*rhog[index]/g2[index]


def get_veff(self, rhoin, i, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpt_band=None):

   '''
   Modified get_veff for plane waves. Computes Vxc and Vh and constructs matrix

   Input:
   ks
   rhoin
   Gd
   Gd_ind
   g2_cutoff

   Output:
   vg

   '''

   Gd_ind = self.pw_grid_params[4]
   g2_cutoff=1e-8
   Gd2 = self.pw_grid_params[5]
   mill_Gd = self.pw_grid_params[11]
   grid_dim = self.pw_grid_params[2]

   #Set up data structures for combined hartree and xc potential
   vg=numpy.zeros(len(Gd_ind),dtype='float64')
   rhog=numpy.zeros(len(Gd_ind),dtype='float64')
   rhoinstore=numpy.empty([0]+grid_dim,dtype='float64')
   rhooutstore=numpy.empty([0]+grid_dim,dtype='float64')

   #Calculate Vxc and Vh potentials. 
   #Vxc is computed in real space while Vh is computed in reciprocal space
   #Vxc takes real space density input, then iffts before building it on the density grid
   #Vh takes the reciprocal density that is first put on the density grid
   vxc=get_vxc(rhoin)
   recip_vxc=numpy.fft.ifftn(vxc)
   temp_rhog=numpy.fft.ifftn(rhoin)
   for ng in range(0,len(Gd_ind)):
      vg[ng]=numpy.real(recip_vxc[pw_helper.get_mill(ng, mill_Gd, grid_dim)])
      rhog[ng]=numpy.real(temp_rhog[pw_helper.get_mill(ng, mill_Gd, grid_dim)])

   #Ensure that zero vector does not contribute
   largeind=Gd2>g2_cutoff
   vg[largeind]+=get_vh(largeind,rhog,Gd2)


   if i ==0:
      return numpy.zeros(len(Gd_ind),dtype='float64')
   else:
      return vg


class RKS_PW(pbchf_pw.RHF_PW):
    '''RKS class adapted for PBCs. 
    
    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.

    '''
    def __init__(self, cell, kpt=numpy.zeros(3)):
        #pbchf.RHF.__init__(self, cell, kpt)
        pbchf_pw.RHF_PW.__init__(self,cell,kpt)
        self.xc = 'LDA,VWN'
        
        #self.pw_grid_params=[pw_helper.return_grids()]
        self.pw_grid_params=pw_helper.return_grids()
        
        self.grids = gen_grid.UniformGrids(cell)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        pbchf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = pyscf.dft.rks.energy_elec

