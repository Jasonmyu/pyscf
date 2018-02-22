#!/usr/bin/python

######################################
#TO DO
#generalize program to run on other lattice configs/atoms
#test other lattices with qe as reference
#implement into pyscf - ask qiming for general guidelines
######################################

'''
pwdft.py

This code performs a gamma-point DFT calculation on 
a Si diamond lattice in the plane wave basis, estimating
the band gap. The eventual goal is to generalize this code
to accessible lattice/atom types and incorporation into PySCF

The Cell() class from PySCF is explicitly used, in addition to 
the ASE package.

Authors:
Jason Yu - jyu5[at]caltech.edu
Narbe Maldrossian 

Works Referenced:
Hutter book
GTH Pseudopotential paper 1998
Pulay DIIS paper 1986
Martin book

FOR Si Example:
FINE GRID: [1363,3] contains miller indicies for density grid
COARSE GRID: [169,3] contains miller indicies for orbital grid

'''

import ase
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
from numpy import linalg as LA
import sys
from pyscf import lib
from pyscf.pbc import gto as pgto
import pyscf.gto
import pyscf.gto.moleintor
import pyscf.dft
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
import scipy

def get_pyscf_cell(atomtype,unittype,lc,kpts):
   #use ase to build unit cell
   ase_atom=ase.build.bulk(atomtype,unittype,a=lc)
   #initialize pyscf cell structure
   cell = pgto.Cell()

   newatom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
   cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)

   cell.a=ase_atom.cell
   cell.pseudo = {'Si':'gth-pade'}
   cell.ke_cutoff=150/27.21138602
   cell.precision=1.e-8
   cell.dimension=3
   cell.unit = 'B'
   cell.build()
   k=cell.make_kpts(kpts,wrap_around=True) 
   a=cell.a
   h=cell.reciprocal_vectors()
   omega=np.linalg.det(a)

   return k,a,h,omega,cell

def get_k_vec(structure,lc):
   if structure=='diamond':
      return np.array([np.pi*2/lc,np.pi*2/lc,np.pi*2/lc])

def get_density_guess(Gd,omega,ne):
   l = len(Gd)
   a = np.zeros(grid_dim,dtype='float64')
   for x in range(0, 15):
      for y in range(0,15):
         for z in range(0,15):
            a[x][y][z]= ne/(omega)
   return a 
   
def get_lattice_vectors(brav_type, lc):
   if brav_type=='diamond':
      a1=[0, lc/2, lc/2]
      a2=[lc/2, 0, lc/2]
      a3=[lc/2, lc/2, 0]

   h = [a1,a2,a3]
   return h

def get_volume(h):
   omega = np.linalg.det(h)
   return omega

def get_reciprocal_lattice(h, dim, norm_to=2*np.pi):   
   a = h 
   b = np.linalg.inv(a)
   return norm_to*b    
   
def gen_real_grid(gs, h):
   #real space reprenstation R = hNq
   #gs are the real space g vectors
   ngs = 2*np.asarray(gs)+1
   qv = lib.cartesian_prod([np.arange(x) for x in ngs])
   a_frac = np.einsum('i, ij->ij', 1./ngs, h)
   r = np.dot(qv, a_frac)
   return r

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
   
def get_kpts(nks,b):
   ks_each_axis=[]
   for n in nks:
      ks = (np.arange(n)+0.5)/n-0.5
      ks_each_axis.append(ks)
   scaled_kpts = lib.cartesian_prod(ks_each_axis)
   #get scaled k-pts of 1/bohr given scaled k-points in fractions of lattice vectors
   kpts = 1./(2*np.pi)*np.dot(scaled_kpts, b)
   return kpts

def get_special_kpts(IBZ_sample):
   k=np.zeros(shape=[1,3],dtype='float64')
   k[0]=IBZ_sample[0]
   ind=0
   for x in range(len(IBZ_sample)-1):
      vec=IBZ_sample[x+1]-IBZ_sample[x]
      inc = vec/float(spt_div)
      for y in range(spt_div):
         ind+=1
         k=np.concatenate((k,np.expand_dims(IBZ_sample[x]+(y+1)*inc,axis=0)))
   return k

#Get kinetic energy contribution
def get_T(j,k,gh,hcore):
   indices=indgk[j,:len(gh)]
   gh=Gd[indices]+k
   gh2=np.einsum('ij,ij->i',gh,gh)/2.+hcore.diagonal()
   return gh2

def get_T_bands(npw,hcore):
   indicies=indgk[0,:npw]
   gh=Gd[indicies]
   gh2=np.einsum('ij,ij->i',gh,gh)/2.+hcore.diagonal()
   return gh2

#Get G squared
def get_G2(G):
   temp = []
   for x in range(0, len(G)):
      temp.append(np.dot(G[x],G[x]))
   return temp

#Get Vxc from LDA approximation via formula
# Vxc = rho^(1/3)*constant term
def get_vxc(rho):
   xalpha = 2./3.
   return (-1.5)*xalpha*(3.*rho/np.pi)**(1./3.)

#Get Coulomb Potential via formula Vh = 4*Pi*rho(G)^2/G^2
def get_vh(index,rhog,g2):
   return 4*np.pi*rhog[index]/g2[largeind]

#Get local pseudopotential component
def get_locpp_single(cell, g2, znuc, zion, rloc, c1, c2, c3, c4, numatom):

   gr2 = rloc*rloc*g2 
   gr4 = gr2**2
   gr6 = gr2**3

   if g2 > g2_cutoff:
      loc = np.exp(-g2*rloc*rloc/2.)*(-4.*np.pi*zion/g2+np.sqrt(8.*np.pi**3.)*rloc*rloc*rloc*(c1+c2*(3.-gr2)+c3*(15.-10.*gr2+gr4)+c4*(105.-105.*gr2+21.*gr4-gr6)))
   else:
      loc=2.*np.pi*gr2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*np.pi)*rloc*zion)

   return loc/v

#Compute the vectorized local pseudopotential according to GTH 1998
def get_loc_pp_vec(g2,v,c1,c2,c3,c4,rloc,Zion):
    g2_cutoff=1e-8

    loc=np.zeros(len(g2),dtype='float64')
    largeind=g2>g2_cutoff
    smallind=g2<=g2_cutoff
    g2=g2[largeind]
    rloc2=rloc*rloc
    rloc3=rloc*rloc2
    gr2=g2*rloc2
    gr4=gr2*gr2
    gr6=gr2*gr4

    loc_large=np.exp(-gr2/2.)*(-4.*np.pi*Zion/g2+np.sqrt(8.*np.pi**3.)*rloc3*(c1+c2*(3.-g2*rloc2)+c3*(15.-10.*gr2+gr4)+c4*(105.-105.*gr2+21.*gr4-gr6)))
    loc_small=2.*np.pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*np.pi)*rloc+Zion)
    loc[largeind]=loc_large
    loc[smallind]=loc_small

    return loc/v

#Read in stored values for the projectors and spherical harmonics to compute nonloc pp
def get_nonloc_pp_vec(sphg,pg,gind,pp):
   #First set h coefficients required for triple sum
   #note that this is a hardcoded example for Si and more coefficents will be required for different atoms
   
   hgth=np.zeros((2,3,3),dtype='float64')
   #hgth[0,0,0]=5.906928
   hgth[0,0,0]=pp[5][2][0][0]
   #hgth[0,1,1]=3.258196
   hgth[0,1,1]=pp[5][2][1][1]
   hgth[0,0,1]=hgth[0,1,0]=-0.5*np.sqrt(3./5.)*hgth[0,1,1]
   #hgth[1,0,0]=2.727013
   hgth[1,0,0]=pp[6][2][0][0]

   #Begin triple sum over i,j, and m.
   #note that i,j range from [0,l] (starting from 0), m ranges from [-l,l]
   #The (-1)^l term is omitted here for this example
   vsg=0.
   for l in [0,1]:
      vsgij=vsgsp=0.
      for i in [0,1]:
         for j in [0,1]:
            vsgij+=pg[l,i,gind]*hgth[l,i,j]*pg[l,j,:]
      for m in range(-l, l+1):
         vsgsp+=sphg[l,m+l,gind]*sphg[l,m+l,:].conj()
      vsg+=vsgij*vsgsp
   
   return vsg/v

def store_nonloc_pp(gv,pp):
   #get polar coordinates for reciprocal space vectors 
   #enter GTH parameters for silicon (currently hard-coded)
   
   rgv,thetagv,phigv=pgto.pseudo.pp.cart2polar(gv)
   #r0=0.422738
   #r1=0.484278
   r0=pp[5][0]
   r1=pp[6][0]
   rl=[r0,r1]

   #max values for angular momentum quantum numbers (currently hard-coded)
   lmax=2
   mmax=2*(lmax-1)+1
   imax=2
   gmax=len(gv)

   #Data structures to store projectors and spherical harmonics
   SHstore=np.zeros((lmax,mmax,gmax),dtype='complex128')
   Pstore=np.zeros((lmax,imax,gmax),dtype='complex128')

   #compute and store values for spherical harmonics from scipy.special.sph_harm
   #compute values for projectors through using pgto.pseudo.pp.projG_li from pyscf
   for l in range(lmax):
      for m in range(-l,l+1):
         SHstore[l,m+l,:]=scipy.special.sph_harm(m,l,phigv,thetagv)
      for i in range(imax):
         Pstore[l,i,:]=pgto.pseudo.pp.projG_li(rgv,l,i,rl[l])

   return SHstore, Pstore


#Get difference vector for non-vectorized loop 
def get_g_gprime(gh_ind, r):
   temp = []
   temp2 = []
   i = 0
   j = 0
   for x in gh_ind:
      for y in gh_ind:
         temp.append(r[x]-r[y])
         temp2.append([i,j])
         i+=1
      j+=1
      i=0
   return temp, temp2

#Get miller index for element on fine grid given an input index
def get_mill(x1):
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


def return_pp():
   lc = 10.26
   h = get_lattice_vectors('diamond',lc)
   spt_div= 10
   kpts = [1,1,1]

   np.set_printoptions(threshold='nan')
   
   #kinetic energy cutoff in hartree
   ke_cutoff=150/27.21138602
   maxcyc = 50

   #Get volume, recip lattice vectors, and grid size
   v = get_volume(h)
   b = get_reciprocal_lattice(h, 1)
   gs = get_gs(7, 7, 7)

   #############GET CELL PARAMETERS FROM ASE/PYSCF################
   k,h,b,v,cell=get_pyscf_cell('Si','diamond',10.26,kpts)

   for ia in range(cell.natm):
      symb=cell.atom_symbol(ia)
      if symb not in cell._pseudo:
         continue
      Zion = cell.atom_charge(ia)
      pseudop = cell._pseudo[symb]
      rloc, nexp, cexp = pseudop[1:3+1]

   c=np.zeros(4,dtype='float64')      
   if len(cexp)==4:
      c[0]=cexp[0]
      c[1]=cexp[1]
      c[2]=cexp[2]
      c[3]=cexp[3]
   elif len(cexp)==3:
      c[0]=cexp[0]
      c[1]=cexp[1]
      c[2]=cexp[2]
   elif len(cexp)==2:
      c[0]=cexp[0]
      c[1]=cexp[1]
   else:
      c[0]=cexp[0]
   #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

   #Get a specified fraction of real space "gridpoints" and corresponding real space latice vectors
   Gv, r = get_Gv(gs, b)
   G2 = np.array(get_G2(Gv))
   kvec = get_k_vec('diamond',lc)
   kpts = get_kpts([5,5,5], b)
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
   
   #Misc. Properties
   g2_cutoff=1e-8
   nbands=4
   
   ####################TEST VALUES################################
   #Gd=np.load('retryg.npy')
   #indg=np.load('retryindg.npy')
   #indgk=np.load('retryindgk.npy')
   #mill_Gd=np.load('retrymill.npy')
   #Gd2=np.load('retryg2.npy')
   ########################################################
   #STRUCTURE FACTOR
   #sf=np.zeros(len(Gd_ind),dtype='complex128')
   #for i in range(len(Gd_ind)):
   #   sf[i]=2.*np.cos(np.dot(Gd[i],np.array([lc,lc,lc])/8.))
   

   print 'Setting up initial hcore shell...'

   #create empty hcore and fill up with kinetic energy diagonal   
   #recip_dens=np.fft.ifftn(ini_den)

   #####################################################################################################
   #BEGIN SELF CONSISTENT LOOP 

   #Initial Density Guess which goes as 8/v
   rhoin=get_density_guess(Gd,v,8)
   drho2=1000.

   #Set up data structures for combined hartree and xc potential
   #in addition to DIIS density mixing scheme
   vg=np.zeros(len(Gd_ind),dtype='float64')
   rhog=np.zeros(len(Gd_ind),dtype='float64')
   rhoinstore=np.empty([0]+grid_dim,dtype='float64')
   rhooutstore=np.empty([0]+grid_dim,dtype='float64')
   mss=5
   alphamix=1.0

   #Begin main loop

   for i in range(maxcyc):
      rhoout=np.zeros(grid_dim,dtype='float64')
      iternum=i+1
      print "beginning cycle ",i
      for j in range(0,len(k)):
         hcore=np.zeros((len(GH_ind),len(GH_ind)),dtype='complex128')
         gkind=indgk[j,:len(GH)]
         gk=Gd[gkind]
         sphg,pg=store_nonloc_pp(k[j]+gk,pseudop)

         #Calculate Vxc and Vh potentials. 
         #Vxc is computed in real space while Vh is computed in reciprocal space
         #Vxc takes real space density input, then iffts before building it on the density grid
         #Vh takes the reciprocal density that is first put on the density grid
         vxc=get_vxc(rhoin)
         recip_vxc=np.fft.ifftn(vxc)
         temp_rhog=np.fft.ifftn(rhoin)
         for ng in range(0,len(Gd_ind)):
            vg[ng]=np.real(recip_vxc[get_mill(ng)])
            rhog[ng]=np.real(temp_rhog[get_mill(ng)])

         #Ensure that zero vector does not contribute
         largeind=Gd2>g2_cutoff
         vg[largeind]+=get_vh(largeind,rhog,Gd2)

         #If on first iteration, zero out Vxc and Vh contribution
         if i ==0:
            vg=np.zeros(len(Gd_ind),dtype='float64')      

         #loc = get_loc_pp_vec(Gd2,v,c[0],c[1],c[2],c[3],rloc,Zion)*sf[0:len(Gd2)]

         #Building Fock matrix
         for aa in range(0,len(GH_ind)):
            ik = indgk[j][aa]
            gdiff = mill_Gd[ik]-mill_Gd[gkind[aa:]]+np.array(gs)
            inds = indg[gdiff.T.tolist()]
            loc = get_loc_pp_vec(Gd2[inds],v,c[0],c[1],c[2],c[3],rloc,Zion)+get_nonloc_pp_vec(sphg,pg,aa,pseudop)[aa:]
            #loc = get_loc_pp_vec(Gd2[inds],v,-7.33610297,0,0,0,0.44,4)
            hcore[aa,aa:]=loc*sf[inds]
            #hcore[aa,aa:]=loc*sf[inds]


         return hcore
        
         '''
         
         tdiag = get_T(j,k[j],GH,hcore)
         np.fill_diagonal(hcore,tdiag)

         #Diagonalize Fock matrix with scipy
         eigval,eigvec=scipy.linalg.eigh(hcore,lower=False,eigvals=(0,nbands-1))

         #Generate new density by summing over the number of bands and orbitals 
         #build auxillary density through inserting molecular coefficients
         #at their respective miller index, then square it.
         for pp in range(nbands):
            aux=np.zeros(grid_dim,dtype='complex128')
            for tt in range(0, len(GH_ind)):
               ik=indgk[j][tt]
               aux[get_mill(ik)]=eigvec[tt,pp]
            aux=(1./np.sqrt(v))*np.fft.fftn(aux)
            rhoout+=(2./len(k))*np.absolute(aux)**2

         print 'density sample',rhoout[0][0]
         print 'eigval sample',eigval*27.21138602

      res=rhoout-rhoin
      drho2=np.sqrt(v)*np.sqrt(np.mean(res**2))
 
      #RM-DIIS DENSITY MIXING SCHEME
      if iternum<=mss:
         rhoinstore=np.concatenate((rhoinstore,np.expand_dims(rhoin,axis=0)))
         rhooutstore=np.concatenate((rhooutstore,np.expand_dims(rhoout,axis=0)))
      else:
         rhoinstore[:mss-1,:,:,:]=rhoinstore[1:,:,:,:]
         rhoinstore[mss-1,:,:,:]=rhoin
         rhooutstore[:mss-1,:,:,:]=rhooutstore[1:,:,:,:]
         rhooutstore[mss-1,:,:,:]=rhoout

      DIISmatdim=np.amin([iternum,mss])

      #set up A matrix
      Amat=np.zeros((DIISmatdim,DIISmatdim),dtype='float64')
      for cyc1 in range(DIISmatdim):
         for cyc2 in range(DIISmatdim):
            Amat[cyc1,cyc2]=(v/(grid_dim[0]*grid_dim[1]*grid_dim[2]))*np.sum(np.abs((rhooutstore[cyc1,:,:,:]-rhoinstore[cyc1,:,:,:])*(rhooutstore[cyc2,:,:,:]-rhoinstore[cyc2,:,:,:])))

      alphamat=np.zeros(DIISmatdim,dtype='float64')

      #solve for alpha coefficients
      for cyc in range(DIISmatdim):
         alphamat[cyc]=np.sum(1./Amat[:,cyc])/np.sum(1./Amat)

      rhoinnew=0.
      rhooutnew=0.

      #get new density matrix from coefficients 
      for cyc in range(DIISmatdim):
         rhoinnew+=alphamat[cyc]*rhoinstore[cyc,:,:,:]
         rhooutnew+=alphamat[cyc]*rhooutstore[cyc,:,:,:]
      rhoin=alphamix*rhooutnew+(1-alphamix)*rhoinnew

      #LINEAR MIXING      
      ########################
      #rhoin=0.2*rhoout+0.8*rhoin
      ########################

      if drho2<1e-6:
         print "Converged"
         break
      else:
         print "this iteration did not converge"
         print "residual: ",drho2

   #Begin next iteration


   ###################################### Begin band calculation ############################################
   sample_IBZ=(2.*np.pi/lc)*np.array([[0.5,0.5,0.5],[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.5,0.0],[0.75,0.75,0.0],[0.0,0.0,0.0]])
   k = get_special_kpts(sample_IBZ)
   
   #re-calculate indgk and npw for new kpts of BZ sample
   npw=np.zeros(len(k),dtype='int')
   for x0 in range(len(k)):
      for x1 in range(len(Gd)):
         temp1 = k[x0]+Gd[x1]
         temp2 = np.dot(temp1,temp1)
         if(temp2/2. <= ke_cutoff):
            npw[x0]+=1
   
   indgk=np.ones((len(k),np.amax(npw)),dtype='int')*10000000
   for x0 in range(0,len(k)):
      inc=0
      for x1 in range(0,len(Gd_ind)):
         temp1= k[x0]+Gd[x1]
         temp2= np.dot(temp1,temp1)
         if(temp2/2. <= ke_cutoff):
            indgk[x0,inc]=x1
            inc+=1
 
   saveeval=np.zeros((len(k),np.amin(npw)),dtype='float64')

   
   #calculate band energies with converged orbital coefficients
   for i in range(0,len(k)):
      hcore=np.zeros((npw[i],npw[i]),dtype='complex128')
      gkind=indgk[i,:npw[i]]
      gk=Gd[gkind]
      sphg,pg=store_nonloc_pp(k[i]+gk,pseudop)

      for aa in range(0,npw[i]):
         ik = indgk[i][aa]
         gdiff = mill_Gd[ik]-mill_Gd[gkind]+np.array(gs)
         inds = indg[gdiff.T.tolist()]
         loc = get_loc_pp_vec(Gd2[inds],v,-7.336103,0,0,0,0.44,4)+get_nonloc_pp_vec(sphg,pg,aa,pseudop)
         hcore[aa,:]=loc*sf[inds]+vg[inds]

      tempT=k[i]+Gd[gkind]
      tdiag=np.einsum('ij,ij->i',tempT,tempT)/2.+hcore.diagonal()
      np.fill_diagonal(hcore,tdiag)

      #diagonalize hcore and save the lowest 4 eigenvalues
      saveeval[i,:]=scipy.linalg.eigvalsh(hcore,eigvals=(0,np.amin(npw)-1))[:np.amin(npw)]
      print 'band energy ',saveeval[i,:nbands]*27.21138602

   #subtracts every value in saveeval by the max value in the range [:,:nbands] (first four bands for each kpt)
   saveeval=(saveeval-np.amax(saveeval[:,:nbands]))*27.21138602
   IBZlist=["L","$\Gamma$","X","W","K","$\Gamma$"]
   #calculates IBG by subtracting the min value in [:,nbands:] from max value in [:,:nbands]
   #i.e. subtracts lowest eigenvalue above band gap and highest eigenvalue below band gap
   IBG=np.amin(saveeval[:,nbands:])-np.amax(saveeval[:,:nbands])
   #Does the same for DBG, only uses gamma point index
   DBG=np.amin(saveeval[spt_div*IBZlist.index('$\\Gamma$'),nbands:])-np.amax(saveeval[spt_div*IBZlist.index('$\\Gamma$'),:nbands])
    
   print 'IBG',IBG
   print 'DBG',DBG

   #################################################################################################
   #end program
   print 'terminating program..'

   #code to plot figure
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(Gd[:,0],Gd[:,1],Gd[:,2], c='r', marker='o')
   plt.show()
   '''
   '''
   #get G-G'vectors
   g_gprime, g_gprime_i = get_g_gprime(GH, r, GH_ind)
   g_gprime = np.array(g_gprime)
   g_gprime_i=np.array(g_gprime_i)
   g_gprime2 = np.array(get_G2(g_gprime))
   print g_gprime2.shape

   SI = get_structure_fac(g_gprime,cl.atom_coords())

   #Example of how to get local part of pseudopotential
   locpp = get_locpp(kpts[0], r, cl, v, g_gprime, SI, 14, 4, 0.44, -6.9136286, 0, 0, 0, 8)
   locpp=locpp.reshape((169,169))
   print 'local pp shape',locpp.shape
   '''

   '''
   N = [[1/Nx,0,0],[0,1/Ny,0],[0,0,1/Nz]]
   q = np.zeroes((3, Nx*4+1))
   q[:,0]=[0,0,0]

   #generate q
   val_iterator = 1 
   for y in range(0, Nx*4+1,4):
      q[:,y] = [val_iterator,0,0]
      q[:,y+1] = [0,val_iterator,0]
      q[:,y+2] = [0,0,val_iterator]
      q[:,y+3] = [val_iterator,val_iterator,val_iterator]
      
   R = np.dot(h, np.dot(N,q))
   '''


