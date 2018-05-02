#!/usr/bin/python
'''
Helper/wrapper for pw-dft implementation

Authors: 
Jason Yu <jyu5@caltech.edu>
Narbe Maldrossian

'''

from pyscf import lib
import call
import numpy as np


#Get three dimensional g-vectors. indicies go according to [0...Ni, -Ni...-1] to match FFT
#Thus, the number of grid points go to 2*Ni+1, where Ni is the number of grid points
#in the i = x,y,z direction
def get_Gv(gs, b):
   gxrange = np.append(range(gs[0]+1), range(-gs[0],0))
   gyrange = np.append(range(gs[1]+1), range(-gs[1],0))
   gzrange = np.append(range(gs[2]+1), range(-gs[2],0))
   gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
   Gv = np.dot(gxyz, b)
   return Gv,gxyz

#Get three dimensional g-vectors corresponding to G^2 < 4(ECUT)
def get_Gd(gv2, gv, ecut, r):
   temp=[]
   temp_ind=[]
   temp2=[]
   #loop over number of column vectors
   for x in range(0, len(gv2)):
      if gv2[x]/2. < 4.*ecut:
         temp.append(gv[x])
         temp_ind.append(x)
         temp2.append(r[x])
   rd = np.array(temp2)
   Gd = np.array(temp)
   Gd_ind = np.array(temp_ind)
   return Gd, Gd_ind, rd

#create list containing miller indices for each G vector on density grid
def create_mill_Gd(Gd_ind,r):
   tempmill=[]
   for x in range(len(Gd_ind)):
      temp=[r[Gd_ind[x]][0],r[Gd_ind[x]][1],r[Gd_ind[x]][2]]
      tempmill.append(temp)
   return np.array(tempmill,dtype='int')

#Get three dimensional g-vectors corresponding to G^2 < ECUT
def get_GH(gv2, gv, ecut, k):
   
   npw=np.zeros(len(k),dtype='int')     

   for x in range(len(k)):
      for y in range(len(gv)):
         kgtmp = gv[y]+k[x]
         kgtmp2 = np.dot(kgtmp,kgtmp)
         if kgtmp2/2. <= ecut:
            npw[x]+=1

   #loop over number of column vectors containing grid points in Gd
   
   GH = []
   GH_ind = []

   for y in range(len(k)):
      temp = []
      temp_ind = []
      for x in range(len(gv)):
         kgtmp = gv[x]+k[y]
         kgtmp2 = np.dot(kgtmp,kgtmp)
         if kgtmp2/2. <= ecut:
            temp.append(gv[x])
            temp_ind.append(x)
      GH.append(np.array(temp))
      GH_ind.append(np.array(temp_ind))
   return GH, GH_ind, npw

#Get G squared
def get_G2(G):
   temp = []
   for x in range(0, len(G)):
      temp.append(np.dot(G[x],G[x]))
   return temp

#Get miller index for element on fine grid given an input index
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

   return index1,index2,index3

def return_grids(cell,k,h,b,v,ke_cutoff):
   import call
   import math
   import numpy as np
   from pyscf import lib
   import pyscf.gto.moleintor
   import scipy

   gs = cell.gs 
   lc = np.cbrt(cell.vol)
   
   #Get a specified fraction of real space "gridpoints" and corresponding real space latice vectors
   Gv, r = get_Gv(gs, b)
   G2 = np.array(get_G2(Gv))
   grid_dim=[2*gs[0]+1,2*gs[1]+1,2*gs[2]+1]

   #Get fine grid in real space and corresponding miller indices 
   Gd, Gd_ind, rd = get_Gd(G2, Gv, ke_cutoff, r)
   mill_Gd=create_mill_Gd(Gd_ind,r)
   Gd2 = np.array(get_G2(Gd))

   #COARSE GRID
   GH, GH_ind, npw = get_GH(G2, Gv, ke_cutoff, k)
   ngs=len(GH_ind)

   #STRUCTURE FACTOR
   sf=np.zeros(len(Gd_ind),dtype='complex128')
   for i in range(len(Gd_ind)):
      sf[i]=2.*np.cos(np.dot(Gd[i],np.array([lc,lc,lc])/8.))

   #miller indices by number of Gd vectors
   indg=np.ones(grid_dim,dtype='int')*1000000
   for i in range(len(Gd)):
      indg[mill_Gd[i,0]+gs[0],mill_Gd[i,1]+gs[1],mill_Gd[i,2]+gs[2]]=i

   #indgk re-indexes Gd vectors fitting the ke_cutoff into a [1,169] list
   #the dimensions change depending on the number of kpts, so it is kept general for now
   indgk=np.ones((len(k),np.amax(npw)),dtype='int')*1000000
   for x0 in range(len(k)):
      ind=0
      for x1 in range(len(Gd)):
         temp1=k[x0]+Gd[x1]
         temp2 = np.dot(temp1,temp1)
         if(temp2/2. <= ke_cutoff):
            indgk[x0][ind]=x1
            ind+=1

   return gs, Gv, grid_dim, Gd, Gd_ind, Gd2, GH, GH_ind, sf, indg, indgk, mill_Gd, v, rd, r, npw


if __name__=="__main__":
   np.set_printoptions(threshold='nan')
   from pyscf.pbc.gto.pseudo import pp_pw
   from pyscf.pbc.gto.pseudo import pp
   import pyscf.pbc.dft as pbcdft  
   from pyscf.scf import hf as scfhf 

   cell_atom = 'Si'
   lattice = 'diamond'
   l_constant = 10.26
   kpts = [1,1,1]
   ke_cutoff = 136.05698066
   gsize = [7,7,7]

   #Build cell with helper function that uses ase to find lattice cell
   cell=call.get_pyscf_cell(cell_atom, lattice, l_constant, gsize, ke_cutoff, kpts)
   
   mf = pbcdft.KRKS_PW(cell)
   mf.diis=False

   #########RUN KERNEL######################
   e = scfhf.kernel(mf)
   print e[1]

                                                                                                           
