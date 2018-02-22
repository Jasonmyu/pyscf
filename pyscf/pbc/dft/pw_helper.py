#!/usr/bin/python
from pyscf import lib
import call
import numpy as np


#put number of grid points into array   
def get_gs(Nx, Ny, Nz):
   gs = np.asarray([Nx, Ny, Nz])
   return gs

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
def get_Gd(gv2,gv, ecut ,r):
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
def get_GH(gv2, gv, ecut):
   temp=[]
   temp_ind=[]
   #loop over number of column vectors containing grid points in Gd
   for x in range(0, len(gv)):
      if gv2[x]/2. < ecut:
         temp.append(gv[x])
         temp_ind.append(x)
   GH = np.array(temp)
   GH_ind = np.array(temp_ind)
   return GH, GH_ind

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


def return_grids():
   import call
   import math
   import numpy as np
   from pyscf import lib
   import pyscf.gto.moleintor
   import scipy

   gs = get_gs(7, 7, 7)
   kpts=[1,1,1]
   lc = 10.26
   k,h,b,v,cell=call.get_pyscf_cell('Si','diamond',10.26,kpts)
   ke_cutoff = 150/27.21138602
   
   #Get a specified fraction of real space "gridpoints" and corresponding real space latice vectors
   Gv, r = get_Gv(gs, b)
   G2 = np.array(get_G2(Gv))
   grid_dim=[2*gs[0]+1,2*gs[1]+1,2*gs[2]+1]

   #Get fine grid in real space and corresponding miller indices 
   Gd, Gd_ind, rd = get_Gd(G2, Gv, ke_cutoff, r)
   mill_Gd=create_mill_Gd(Gd_ind,r)
   Gd2 = np.array(get_G2(Gd))

   #COARSE GRID
   GH, GH_ind = get_GH(G2, Gv, ke_cutoff)
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
   indgk=np.ones((1,len(GH)),dtype='int')*1000000
   ind=0
   for x0 in range(len(k)):
      for x1 in range(len(Gd)):
         temp1=k[x0]+Gd[x1]
         temp2 = np.dot(temp1,temp1)
         if(temp2/2. <= ke_cutoff):
            indgk[x0][ind]=x1
            ind+=1
   
   return gs, Gv, grid_dim, Gd, Gd_ind, Gd2, GH, GH_ind, sf, indg, indgk, mill_Gd, v, rd, r


if __name__=="__main__":
   np.set_printoptions(threshold='nan')
   #gs, Gv, grid_dim, Gd, Gd_ind, Gd2, GH, GH_ind, sf, indg, indgk = return_grids()
   #print grid_dim
   from pyscf.pbc.gto.pseudo import pp_pw
   from pyscf.pbc.gto.pseudo import test_pp
   from pyscf.pbc.gto.pseudo import pp
   import pyscf.pbc.dft as pbcdft  
   from pyscf.scf import hf as scfhf 

   k,h,b,v,cell=call.get_pyscf_cell('Si','diamond',10.26,[1,1,1])
   mf = pbcdft.RKS_PW(cell)

   k_test= [0,0,0]
   #k_test = k_test.T 
  
   #print cell.get_SI().flatten()
   #print mf.pw_grid_params[0][8]
   #print pp_pw.get_pploc_gth_pw(cell, mf.pw_grid_params[0][3])
   #print len(mf.pw_grid_params)

   #print pp_pw.get_pp(cell,mf.pw_grid_params[3],mf.pw_grid_params[13],mf.pw_grid_params[12], mf.pw_grid_params[6],mf.pw_grid_params[0], k_test).real 
   
   print pp_pw.get_pp(cell, mf.pw_grid_params[1],mf.pw_grid_params[14],mf.pw_grid_params[12], mf.pw_grid_params[6], mf.pw_grid_params[0],k_test).real
   #print test_pp.get_pp(cell,k_test).shape
   #print pp.get_pp(cell,k_test)

   #print mf.kpt
   print scfhf.kernel_pw(mf)

   #print rks_pw.get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpt=None, kpt_band=None)
                                                                                                           

