#!/usr/bin/python

def get_pyscf_cell(atomtype,unittype,lc,kpts):
   import ase
   import pyscf.pbc.tools.pyscf_ase as pyscf_ase
   import numpy as np
   from numpy import linalg as LA
   import pyscf.pbc.gto as pgto   
   
   #use ase to build unit cell
   ase_atom=ase.build.bulk(atomtype,unittype,a=lc)
   #initialize pyscf cell structure
   cell = pgto.Cell()

   newatom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
   cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)

   cell.a=ase_atom.cell
   cell.pseudo = {atomtype:'gth-pade'}
   cell.ke_cutoff=150/27.21138602
   cell.precision=1.e-8
   cell.dimension=3
   cell.unit = 'B'
   cell.build()
   cell.kpts=cell.make_kpts(kpts,wrap_around=True)
   k = cell.kpts
   a=cell.a
   h=cell.reciprocal_vectors()
   omega=np.linalg.det(a)
   
   return k,a,h,omega,cell


#if __name__=="__main__":
#   k,h,b,v,cell=get_pyscf_cell('Si','diamond',10.26,[1,1,1])
   # mf = pbcdft.RKS(cell)
   #print mf.pw_grid_params
   #print rks_pw.get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpt=None, kpt_band=None)
