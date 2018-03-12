#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock
'''

import sys
import tempfile
import time
from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import diis
from pyscf.scf import _vhf
from pyscf.scf import chkfile

def kernel_pw(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):

   import return_pp
   mol = mf.mol

   #Initial Density Guess which goes as 8/v
   rhoin=mf.get_density_guess_pw(mol)

   #store for DIIS and convergence threshold
   rhoinstore=numpy.empty([0]+mf.pw_grid_params[2],dtype='float64')
   rhooutstore=numpy.empty([0]+mf.pw_grid_params[2],dtype='float64')
   drho2=1000.

   #hard code nbands for now, will change after preliminary implementation
   nbands=4

   #Set up data structures for combined hartree and xc potential
   #in addition to DIIS density mixing scheme
   mss=5
   alphamix=1.0
   maxcyc=50

   #Catch case where len(kpts)=1, i.e. for gamma point calculation
   kpts = mf.kpt
   if isinstance(kpts[0], list) is False:
      kpts = [kpts]

   #Begin main SCF loop
   for i in range(maxcyc):
      rhoout=mf.get_density_shell(mol)
      iternum=i+1
      print "beginning cycle ",i
      for j in range(0,len(kpts)):

         #If on first iteration, zero out Vxc and Vh contribution
         vg = mf.get_veff(rhoin,i)

         #Get core hamiltonian including t + pp (need to add pp still)
         h1e = mf.get_hcore(j)
         
         #Build Fock Matrix
         fock = mf.get_fock_no_pypseudo(j,vg,h1e)

         #slice in correct pp to test
         fock += return_pp.return_pp_func()      

         #Diagonalize Fock matrix with scipy
         eigval,eigvec=scipy.linalg.eigh(fock,lower=False,eigvals=(0,nbands-1))

         #Generate new density
         rhoout = mf.gen_dens(j, eigvec, kpts[j])

         print 'density sample',rhoout[0][0]
         print 'eigval sample',eigval*27.21138602

         #To do: remove diis from scf kernel
         v = mf.pw_grid_params[12]
         grid_dim = mf.pw_grid_params[2]

         #Calculate residual
         res=rhoout-rhoin
         drho2=numpy.sqrt(v)*numpy.sqrt(numpy.mean(res**2))

         #RM-DIIS density mixing scheme
         if iternum<=mss:
            rhoinstore=numpy.concatenate((rhoinstore,numpy.expand_dims(rhoin,axis=0)))
            rhooutstore=numpy.concatenate((rhooutstore,numpy.expand_dims(rhoout,axis=0)))
         else:
            rhoinstore[:mss-1,:,:,:]=rhoinstore[1:,:,:,:]
            rhoinstore[mss-1,:,:,:]=rhoin
            rhooutstore[:mss-1,:,:,:]=rhooutstore[1:,:,:,:]
            rhooutstore[mss-1,:,:,:]=rhoout

         DIISmatdim=numpy.amin([iternum,mss])

         #set up A matrix
         Amat=numpy.zeros((DIISmatdim,DIISmatdim),dtype='float64')
         for cyc1 in range(DIISmatdim):
            for cyc2 in range(DIISmatdim):
               Amat[cyc1,cyc2]=(v/(grid_dim[0]*grid_dim[1]*grid_dim[2]))*numpy.sum(numpy.abs((rhooutstore[cyc1,:,:,:] - \
               rhoinstore[cyc1,:,:,:])*(rhooutstore[cyc2,:,:,:]-rhoinstore[cyc2,:,:,:])))

         alphamat=numpy.zeros(DIISmatdim,dtype='float64')

         #solve for alpha coefficients
         for cyc in range(DIISmatdim):
            alphamat[cyc]=numpy.sum(1./Amat[:,cyc])/numpy.sum(1./Amat)

         rhoinnew=0.
         rhooutnew=0.

         #get new density matrix from coefficients 
         for cyc in range(DIISmatdim):
            rhoinnew+=alphamat[cyc]*rhoinstore[cyc,:,:,:]
            rhooutnew+=alphamat[cyc]*rhooutstore[cyc,:,:,:]
         rhoin=alphamix*rhooutnew+(1-alphamix)*rhoinnew

         if drho2<1e-6:
            print 'converged'
            #return eigval,eigvec
            return eigval*27.21138602,eigvec
         else:
            print "this iteration did not converge"
            print "residual: ",drho2


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    '''kernel: the SCF driver.

    Args:
        mf : an instance of SCF class
            To hold the flags to control SCF.  Besides the control parameters,
            one can modify its function members to change the behavior of SCF.
            The member functions which are called in kernel are

            | mf.get_init_guess
            | mf.get_hcore
            | mf.get_ovlp
            | mf.get_fock
            | mf.get_grad
            | mf.eig
            | mf.get_occ
            | mf.make_rdm1
            | mf.energy_tot
            | mf.dump_chk

    Kwargs:
        conv_tol : float
            converge threshold.
        conv_tol_grad : float
            gradients converge threshold.
        dump_chk : bool
            Whether to save SCF intermediate results in the checkpoint file
        dm0 : ndarray
            Initial guess density matrix.  If not given (the default), the kernel
            takes the density matrix generated by ``mf.get_init_guess``.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.

    Returns:
        A list :   scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

        scf_conv : bool
            True means SCF converged
        e_tot : float
            Hartree-Fock energy of last iteration
        mo_energy : 1D float array
            Orbital energies.  Depending the eig function provided by mf
            object, the orbital energies may NOT be sorted.
        mo_coeff : 2D array
            Orbital coefficients.
        mo_occ : 1D array
            Orbital occupancies.  The occupancies may NOT be sorted from large
            to small.

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=numpy.eye(mol.nao_nr()))
    >>> print('conv = %s, E(HF) = %.12f' % (conv, e))
    conv = True, E(HF) = -1.081170784378
    '''
    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (time.clock(), time.time())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)

    s1e = mf.get_ovlp(mol)

    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        mf_diis = diis.SCF_DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    vhf = mf.get_veff(mol, dm)

    e_tot = mf.energy_tot(dm, h1e, vhf)
 
    logger.info(mf, 'init E= %.15g', e_tot)

    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    scf_conv = False
    cycle = 0
    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    while not scf_conv and cycle < max(1, mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis=False)

        #print 'pptest', numpy.linalg.norm(pptest)
        #print 'h1e',numpy.linalg.norm(h1e)
        #print 's1e',numpy.linalg.norm(s1e)
        #print 'vhf',numpy.linalg.norm(vhf)
        #print 'fock: ', numpy.linalg.norm(fock[0])
        
        mo_energy, mo_coeff = mf.eig(fock, s1e)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)

        print 'iteration',cycle
        for x in range(len(fock)):
            print 'lowest eig, k=',x,' ',mo_energy[x][0:4]*27.21138602

        '''
        dm = lib.tag_array(dm, mo_coeff=eigvec_filled, mo_occ=mo_occ)
        '''

        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        e_tot = mf.energy_tot(dm, h1e, vhf)
        print 'e_tot',e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, diis=False)  # = h1e + vhf, no DIIS

        #ignore for now, will hack it to zero later
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if (abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    if conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        scf_conv = (abs(e_tot-last_hf_e) < conv_tol*10 or
                    norm_gorb < conv_tol_grad*3)
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())
    logger.timer(mf, 'scf_cycle', *cput0)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def energy_elec(mf, dm=None, h1e=None, vhf=None):
    r'''Electronic part of Hartree-Fock energy, for given core hamiltonian and
    HF potential

    ... math::

        E = \sum_{ij}h_{ij} \gamma_{ji}
          + \frac{1}{2}\sum_{ijkl} \gamma_{ji}\gamma_{lk} \langle ik||jl\rangle

    Args:
        mf : an instance of SCF class

    Kwargs:
        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian
        vhf : 2D ndarray
            HF potential

    Returns:
        Hartree-Fock electronic energy and the Coulomb energy

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> dm = mf.make_rdm1()
    >>> scf.hf.energy_elec(mf, dm)
    (-1.5176090667746334, 0.60917167853723675)
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
    e_coul = numpy.einsum('ji,ji', vhf.conj(), dm).real * .5
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulstion
    See :func:`scf.hf.energy_elec` for the electron part
    '''
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    print 'ewald',mf.energy_nuc()
    return e_tot.real


def get_hcore(mol):
    '''Core Hamiltonian

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> scf.hf.get_hcore(mol)
    array([[-0.93767904, -0.59316327],
           [-0.59316327, -0.93767904]])
    '''
    h = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    if mol.has_ecp():
        h += mol.intor_symmetric('ECPscalar')
    return h


def get_ovlp(mol):
    '''Overlap matrix
    '''
    return mol.intor_symmetric('int1e_ovlp')


def init_guess_by_minao(mol):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Returns:
        Density matrix, 2D ndarray

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> scf.hf.init_guess_by_minao(mol)
    array([[ 0.94758917,  0.09227308],
           [ 0.09227308,  0.94758917]])
    '''
    from pyscf.scf import atom_hf
    from pyscf.scf import addons

    def minao_basis(symb, nelec_ecp):
        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
        occ = []
        basis_ano = []
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp)
        #coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            if coreshl[l] > 0:
                occ.extend([0]*coreshl[l]*(2*l+1))
            if ndocc > coreshl[l]:
                occ.extend([2]*(ndocc-coreshl[l])*(2*l+1))
            if frac > 1e-15:
                occ.extend([frac]*(2*l+1))
                ndocc += 1
            if ndocc > 0:
                basis_ano.append([l] + [b[:ndocc+1] for b in basis_add[l][1:]])

        if nelec_ecp > 0:
            occ4ecp = []
            basis4ecp = []
            nelec_valence_left = gto.mole._charge(stdsymb) - nelec_ecp
            for l in range(4):
                if nelec_valence_left <= 0:
                    break
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                assert(ndocc >= coreshl[l])

                n_valenc_shell = ndocc - coreshl[l]
                l_occ = [2] * (n_valenc_shell*(2*l+1))
                if frac > 1e-15:
                    l_occ.extend([frac] * (2*l+1))
                    n_valenc_shell += 1

                shell_found = 0
                for bas in mol._basis[symb]:
                    if shell_found >= n_valenc_shell:
                        break
                    if bas[0] == l:
                        off = n_valenc_shell - shell_found
                        # b[:off+1] because the first column of bas[1] is exp
                        basis4ecp.append([l] + [b[:off+1] for b in bas[1:]])
                        shell_found += len(bas[1]) - 1

                nelec_valence_left -= int(sum(l_occ[:shell_found*(2*l+1)]))
                occ4ecp.extend(l_occ)

            if nelec_valence_left > 0:
                logger.debug(mol, 'Characters of %d valence electrons are '
                             'not identified in the minao initial guess.\n'
                             'Electron density of valence ANO for %s will '
                             'be used.', nelec_valence_left, symb)
                return occ, basis_ano

# Compared to ANO valence basis, to check whether the ECP basis set has
# reasonable AO-character contraction.  The ANO valence AO should have
# significant overlap to ECP basis if the ECP basis has AO-character.
            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)[:,numpy.array(occ)>0]
            if abs(numpy.linalg.det(s12)) > .1:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                logger.debug(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    atmlst = set([mol.atom_symbol(ia) for ia in range(mol.natm)])

    nelec_ecp_dic = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in nelec_ecp_dic:
            nelec_ecp_dic[symb] = mol.atom_nelec_core(ia)

    basis = {}
    occdic = {}
    for symb in atmlst:
        if 'GHOST' not in symb:
            nelec_ecp = nelec_ecp_dic[symb]
            occ_add, basis_add = minao_basis(symb, nelec_ecp)
            occdic[symb] = occ_add
            basis[symb] = basis_add
    occ = []
    new_atom = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if 'GHOST' not in symb:
            occ.append(occdic[symb])
            new_atom.append(mol._atom[ia])
    occ = numpy.hstack(occ)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(new_atom, basis, [])
    c = addons.project_mo_nr2nr(pmol, 1, mol)

    dm = numpy.dot(c*occ, c.T)
# normalize eletron number
#    s = mol.intor_symmetric('int1e_ovlp')
#    dm *= mol.nelectron / (dm*s).sum()
    return dm


def init_guess_by_1e(mol):
    '''Generate initial guess density matrix from core hamiltonian

    Returns:
        Density matrix, 2D ndarray
    '''
    mf = RHF(mol)
    return mf.init_guess_by_1e(mol)


def init_guess_by_atom(mol):
    '''Generate initial guess density matrix from superposition of atomic HF
    density matrix.  The atomic HF is occupancy averaged RHF

    Returns:
        Density matrix, 2D ndarray
    '''
    import copy
    from pyscf.scf import atom_hf
    from pyscf.scf import addons
    atm_scf = atom_hf.get_atm_nrhf(mol)
    mo = []
    mo_occ = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in atm_scf:
            e_hf, e, c, occ = atm_scf[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            e_hf, e, c, occ = atm_scf[symb]
        mo.append(c)
        mo_occ.append(occ)
    mo = scipy.linalg.block_diag(*mo)
    mo_occ = numpy.hstack(mo_occ)

    pmol = copy.copy(mol)
    pmol.cart = False
    c = addons.project_mo_nr2nr(pmol, mo, mol)
    dm = numpy.dot(c*mo_occ, c.T)

    for k, v in atm_scf.items():
        logger.debug1(mol, 'Atom %s, E = %.12g', k, v[0])
    return dm


def init_guess_by_chkfile(mol, chkfile_name, project=True):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``mol``

    Returns:
        Density matrix, 2D ndarray
    '''
    from pyscf.scf import uhf
    dm = uhf.init_guess_by_chkfile(mol, chkfile_name, project)
    return dm[0] + dm[1]


def get_init_guess(mol, key='minao'):
    '''Pick a init_guess method

    Kwargs:
        key : str
            One of 'minao', 'atom', '1e', 'chkfile'.
    '''
    return RHF(mol).get_init_guess(mol, key)


# eigenvalue of d is 1
def level_shift(s, d, f, factor):
    r'''Apply level shift :math:`\Delta` to virtual orbitals

    .. math::
       :nowrap:

       \begin{align}
         FC &= SCE \\
         F &= F + SC \Lambda C^\dagger S \\
         \Lambda_{ij} &=
         \begin{cases}
            \delta_{ij}\Delta & i \in \text{virtual} \\
            0 & \text{otherwise}
         \end{cases}
       \end{align}

    Returns:
        New Fock matrix, 2D ndarray
    '''
    dm_vir = s - reduce(numpy.dot, (s, d, s))
    return f + dm_vir * factor


def damping(s, d, f, factor):
    #dm_vir = s - reduce(numpy.dot, (s,d,s))
    #sinv = numpy.linalg.inv(s)
    #f0 = reduce(numpy.dot, (dm_vir, sinv, f, d, s))
    dm_vir = numpy.eye(s.shape[0]) - numpy.dot(s, d)
    f0 = reduce(numpy.dot, (dm_vir, f, d, s))
    f0 = (f0+f0.T.conj()) * (factor/(factor+1.))
    return f - f0


# full density matrix for RHF
def make_rdm1(mo_coeff, mo_occ):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : 2D ndarray
            Orbital coefficients. Each column is one orbital.
        mo_occ : 1D ndarray
            Occupancy
    '''
    mocc = mo_coeff[:,mo_occ>0]
    return numpy.dot(mocc*mo_occ[mo_occ>0], mocc.T.conj())


################################################
# for general DM
# hermi = 0 : arbitary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def dot_eri_dm(eri, dm, hermi=0):
    '''Compute J, K matrices in terms of the given 2-electron integrals and
    density matrix:

    J ~ numpy.einsum('pqrs,qp->rs', eri, dm)
    K ~ numpy.einsum('pqrs,qr->ps', eri, dm)

    Args:
        eri : ndarray
            8-fold or 4-fold ERIs
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        Depending on the given dm, the function returns one J and one K matrix,
        or a list of J matrices and a list of K matrices, corresponding to the
        input density matrices.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
    >>> dms = numpy.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> j, k = scf.hf.dot_eri_dm(eri, dms, hermi=0)
    >>> print(j.shape)
    (3, 2, 2)
    '''
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        vj, vk = _vhf.incore(eri, dm, hermi=hermi)
    else:
        dm = numpy.asarray(dm, order='C')
        nao = dm.shape[-1]
        dms = dm.reshape(-1,nao,nao)
        vjk = [_vhf.incore(eri, dmi, hermi=hermi) for dmi in dms]
        vj = numpy.array([v[0] for v in vjk]).reshape(dm.shape)
        vk = numpy.array([v[1] for v in vjk]).reshape(dm.shape)
    return vj, vk


def get_jk(mol, dm, hermi=1, vhfopt=None):
    '''Compute J, K matrices for the given density matrix

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices

    Returns:
        Depending on the given dm, the function returns one J and one K matrix,
        or a list of J matrices and a list of K matrices, corresponding to the
        input density matrices.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dms = numpy.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> j, k = scf.hf.get_jk(mol, dms, hermi=0)
    >>> print(j.shape)
    (3, 2, 2)
    '''
    dm = numpy.asarray(dm, order='C')
    nao = dm.shape[-1]
    vj, vk = _vhf.direct(dm.reshape(-1,nao,nao), mol._atm, mol._bas, mol._env,
                         vhfopt=vhfopt, hermi=hermi, cart=mol.cart)
    return vj.reshape(dm.shape), vk.reshape(dm.shape)


def get_veff(mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
    '''Hartree-Fock potential matrix for the given density matrix

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices

    Returns:
        matrix Vhf = 2*J - K.  Vhf can be a list matrices, corresponding to the
        input density matrices.

    Examples:

    >>> import numpy
    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dm0 = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf0 = scf.hf.get_veff(mol, dm0, hermi=0)
    >>> dm1 = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf1 = scf.hf.get_veff(mol, dm1, hermi=0)
    >>> vhf2 = scf.hf.get_veff(mol, dm1, dm_last=dm0, vhf_last=vhf0, hermi=0)
    >>> numpy.allclose(vhf1, vhf2)
    True
    '''
    if dm_last is None:
        vj, vk = get_jk(mol, numpy.asarray(dm), hermi, vhfopt)
        return vj - vk * .5
    else:
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj, vk = get_jk(mol, ddm, hermi, vhfopt)
        return vj - vk * .5 + numpy.asarray(vhf_last)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    '''F = h^{core} + V^{HF}

    Special treatment (damping, DIIS, or level shift) will be applied to the
    Fock matrix if diis and cycle is specified (The two parameters are passed
    to get_fock function during the SCF iteration)

    Args:
        h1e : 2D ndarray
            Core hamiltonian
        s1e : 2D ndarray
            Overlap matrix, for DIIS
        vhf : 2D ndarray
            HF potential matrix
        dm : 2D ndarray
            Density matrix, for DIIS

    Kwargs:
        cycle : int
            Then present SCF iteration step, for DIIS
        diis : an object of :attr:`SCF.DIIS` class
            DIIS object to hold intermediate Fock and error vectors
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        level_shift_factor : float or int
            Level shift (in AU) for virtual space.  Default is 0.
    '''
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(dm=dm)
    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm*.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def get_occ(mf, mo_energy=None, mo_coeff=None):
    '''Label the occupancies for each orbital

    Kwargs:
        mo_energy : 1D ndarray
            Obital energies

        mo_coeff : 2D ndarray
            Obital coefficients

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1')
    >>> mf = scf.hf.SCF(mol)
    >>> energy = numpy.array([-10., -1., 1, -2., 0, -3])
    >>> mf.get_occ(energy)
    array([2, 2, 0, 2, 2, 2])
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g',
                        e_sort[nocc-1], e_sort[nocc])
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g',
                        e_sort[nocc-1], e_sort[nocc])

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''RHF Gradients

    Args:
        mo_coeff : 2D ndarray
            Obital coefficients
        mo_occ : 1D ndarray
            Orbital occupancy
        fock_ao : 2D ndarray
            Fock matrix in AO representation

    Returns:
        Gradients in MO representation.  It's a num_occ*num_vir vector.
    '''
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(numpy.dot, (mo_coeff[:,viridx].T.conj(), fock_ao,
                           mo_coeff[:,occidx])) * 2
    
    return g.ravel()


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=True, **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Diople moment.
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        log.note('**** MO energy ****')
        for i,c in enumerate(mo_occ):
            log.note('MO #%-3d energy= %-18.15g occ= %g', i+1, mo_energy[i], c)

    ovlp_ao = mf.get_ovlp()
    if verbose >= logger.DEBUG:
        label = mf.mol.ao_labels()
        if with_meta_lowdin:
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
            orth_coeff = orth.orth_ao(mf.mol, 'meta_lowdin', s=ovlp_ao)
            c = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff))
        else:
            log.debug(' ** MO coefficients (expansion on AOs) **')
            c = mo_coeff
        dump_mat.dump_rec(mf.stdout, c, label, start=1, **kwargs)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        return (mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log),
                mf.dip_moment(mf.mol, dm, verbose=log))
    else:
        return (mf.mulliken_pop(mf.mol, dm, s=ovlp_ao, verbose=log),
                mf.dip_moment(mf.mol, dm, verbose=log))

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    r'''Mulliken population analysis

    .. math:: M_{ij} = D_{ij} S_{ji}

    Mulliken charges

    .. math:: \delta_i = \sum_j M_{ij}

    '''
    if s is None:
        s = get_ovlp(mol)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        pop = numpy.einsum('ij,ji->i', dm, s).real
    else: # ROHF
        pop = numpy.einsum('ij,ji->i', dm[0]+dm[1], s).real
    label = mol.ao_labels(fmt=None)

    log.note(' ** Mulliken pop  **')
    for i, s in enumerate(label):
        log.note('pop of  %s %10.5f', '%d%s %s%-4s'%s, pop[i])

    log.note(' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg[ia] = mol.atom_charge(ia) - chg[ia]
        log.note('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg


def mulliken_meta(mol, dm, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    In the meta-lowdin, the AOs are grouped in three sets: core, valence and
    Rydberg, the orthogonalization are carreid out within each subsets.

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray or 2-item list of ndarray
            Density matrix.  ROHF dm is a 2-item list of 2D array

    Kwargs:
        verbose : int or instance of :class:`lib.logger.Logger`

        pre_orth_method : str
            Pre-orthogonalization, which localized GTOs for each atom.
            To obtain the occupied and unoccupied atomic shells, there are
            three methods

            | 'ano'   : Project GTOs to ANO basis
            | 'minao' : Project GTOs to MINAO basis
            | 'scf'   : Fraction-averaged RHF

    '''
    from pyscf.lo import orth
    if s is None:
        s = get_ovlp(mol)
    log = logger.new_logger(mol, verbose)
    c = orth.restore_ao_character(mol, pre_orth_method)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c, s=s)
    c_inv = numpy.dot(orth_coeff.T, s)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = reduce(numpy.dot, (c_inv, dm, c_inv.T.conj()))
    else:  # ROHF
        dm = reduce(numpy.dot, (c_inv, dm[0]+dm[1], c_inv.T.conj()))

    log.info(' ** Mulliken pop on meta-lowdin orthogonal AOs  **')
    return mulliken_pop(mol, dm, numpy.eye(orth_coeff.shape[0]), log)
mulliken_pop_meta_lowdin_ao = mulliken_meta


def eig(h, s):
    '''Solver for generalized eigenvalue problem

    .. math:: HC = SCE
    '''
    e, c = scipy.linalg.eigh(h, s)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)
    for idx in (coreidx, openidx, viridx):
        if numpy.count_nonzero(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = reduce(numpy.dot, (orb.T.conj(), fock, orb))
            e, c = scipy.linalg.eigh(f1)
            mo[:,idx] = numpy.dot(orb, c)
            mo_e[idx] = e
    return mo_e, mo

def dip_moment(mol, dm, unit_symbol='Debye', verbose=logger.NOTE):
    r''' Dipole moment calculation

    .. math::

        \mu_x = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|x|\mu) + \sum_A Q_A X_A\\
        \mu_y = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|y|\mu) + \sum_A Q_A Y_A\\
        \mu_z = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|z|\mu) + \sum_A Q_A Z_A

    where :math:`\mu_x, \mu_y, \mu_z` are the x, y and z components of dipole
    moment

    Args:
         mol: an instance of :class:`Mole`
         dm : a 2D ndarrays density matrices

    Return:
        A list: the dipole moment on x, y and z component
    '''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)

    if unit_symbol == 'Debye':
        unit = 2.541746    # a.u. to Debye
    else:
        unit = 1.0

    mol.set_common_orig((0,0,0))
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = numpy.einsum('i,ix->x', charges, coords)

    mol_dip = (nucl_dip - el_dip) * unit

    if unit_symbol == 'Debye' :
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    return mol_dip

############
# For orbital rotation
def uniq_var_indices(mo_occ):
    occidxa = mo_occ>0
    occidxb = mo_occ==2
    viridxa = ~occidxa
    viridxb = ~occidxb
    mask = (viridxa[:,None] & occidxa) | (viridxb[:,None] & occidxb)
    return mask

def pack_uniq_var(x1, mo_occ):
    idx = uniq_var_indices(mo_occ)
    return x1[idx]

def unpack_uniq_var(dx, mo_occ):
    nmo = len(mo_occ)
    idx = uniq_var_indices(mo_occ)

    x1 = numpy.zeros((nmo,nmo), dtype=dx.dtype)
    x1[idx] = dx
    return x1 - x1.T.conj()


def as_scanner(mf):
    '''Generating a scanner/solver for HF PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total HF energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    SCF object (DIIS, conv_tol, max_memory etc) are automatically applied in
    the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf
        >>> hf_scanner = scf.RHF(gto.Mole().set(verbose=0)).as_scanner()
        >>> hf_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        -98.552190448277955
        >>> hf_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
        -98.414750424294368
    '''
    import copy
    logger.info(mf, 'Create scanner for %s', mf.__class__)

    class SCF_Scanner(mf.__class__, lib.SinglePointScanner):
        def __init__(self, mf_obj):
            self.__dict__.update(mf_obj.__dict__)
            mf_obj = self
            # partial deepcopy to avoid overwriting existing object
            while mf_obj is not None:
                if hasattr(mf_obj, 'with_df'):
                    mf_obj.with_df = copy.copy(mf_obj.with_df)
                if hasattr(mf_obj, 'with_x2c'):
                    mf_obj.with_x2c = copy.copy(mf_obj.with_x2c)
                if hasattr(mf_obj, 'grids'):  # DFT
                    mf_obj.grids = copy.copy(mf_obj.grids)
                    mf_obj._numint = copy.copy(mf_obj._numint)
                if hasattr(mf_obj, '_scf'):
                    mf_obj._scf = copy.copy(mf_obj._scf)
                    mf_obj = mf_obj._scf
                else:
                    break

        def __call__(self, mol):
            mf_obj = self
            while mf_obj is not None:
                mf_obj.mol = mol
                mf_obj.opt = None
                mf_obj._eri = None
                if hasattr(mf_obj, 'with_df') and mf_obj.with_df:
                    mf_obj.with_df.mol = mol
                    mf_obj.with_df.auxmol = None
                    mf_obj.with_df._cderi = None
                if hasattr(mf_obj, 'with_x2c') and mf_obj.with_x2c:
                    mf_obj.with_x2c.mol = mol
                if hasattr(mf_obj, 'grids'):  # DFT
                    mf_obj.grids.mol = mol
                    mf_obj.grids.coords = None
                    mf_obj.grids.weights = None
                    mf_obj._dm_last = None
                mf_obj = getattr(mf_obj, '_scf', None)

            if self.mo_coeff is None:
                dm0 = None
#            elif mol.natm > 0:
# Project wfn from another geometry seems providing a bad initial guess
#                dm0 = self.from_chk(self.chkfile)
            else:
                dm0 = self.make_rdm1()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

    return SCF_Scanner(mf)

############



class SCF(lib.StreamObject):
    '''SCF base class.   non-relativistic RHF.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default equals to :class:`Mole.max_memory`
        chkfile : str
            checkpoint file to save MOs, orbital energies etc.
        conv_tol : float
            converge threshold.  Default is 1e-10
        conv_tol_grad : float
            gradients converge threshold.  Default is sqrt(conv_tol)
        max_cycle : int
            max number of iterations.  Default is 50
        init_guess : str
            initial guess method.  It can be one of 'minao', 'atom', '1e', 'chkfile'.
            Default is 'minao'
        diis : boolean or object of DIIS class listed in :mod:`scf.diis`
            Default is :class:`diis.SCF_DIIS`. Set it to None to turn off DIIS.
        diis_space : int
            DIIS space size.  By default, 8 Fock matrices and errors vector are stored.
        diis_start_cycle : int
            The step to start DIIS.  Default is 1.
        diis_file: 'str'
            File to store DIIS vectors and error vectors.
        level_shift : float or int
            Level shift (in AU) for virtual space.  Default is 0.
        direct_scf : bool
            Direct SCF is used by default.
        direct_scf_tol : float
            Direct SCF cutoff threshold.  Default is 1e-13.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        conv_check : bool
            An extra cycle to check convergence after SCF iterations.

    Saved results

        converged : bool
            SCF converged or not
        e_tot : float
            Total HF energy (electronic energy plus nuclear repulsion)
        mo_energy :
            Orbital energies
        mo_occ
            Orbital occupancy
        mo_coeff
            Orbital coefficients

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.hf.SCF(mol)
    >>> mf.verbose = 0
    >>> mf.level_shift = .4
    >>> mf.scf()
    -1.0811707843775884
    '''
    def __init__(self, mol):
        if not mol._built:
            sys.stderr.write('Warning: mol.build() is not called in input\n')
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

# the chkfile will be removed automatically, to save the chkfile, assign a
# filename to self.chkfile
        self._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.chkfile = self._chkfile.name
        self.conv_tol = 1e-9
        self.conv_tol_grad = None
        self.max_cycle = 100
        self.init_guess = 'minao'
        # To avoid diis pollution form previous run, self.diis should not be
        # initialized as DIIS instance here
        self.diis = True
        self.diis_space = 8
        self.diis_start_cycle = 1 # need > 0 if initial DM is numpy.zeros array
        self.diis_file = None
        # Give diis_space_rollback=True a trial if other efforts not converge
        self.diis_space_rollback = False
        self.damp = 0
        self.level_shift = 0
        self.direct_scf = True
        self.direct_scf_tol = 1e-13
        self.conv_check = True
##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.e_tot = 0
        self.converged = False
        self.callback = None

        self.opt = None
        self._eri = None
        self._keys = set(self.__dict__.keys())

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if (self.direct_scf and not mol.incore_anyway and
            not self._is_mem_enough()):
# Should I lazy initialize direct SCF?
            self.opt = self.init_direct_scf(mol)

    def dump_flags(self):
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'method = %s', self.__class__.__name__)
        logger.info(self, 'initial guess = %s', self.init_guess)
        logger.info(self, 'damping factor = %g', self.damp)
        logger.info(self, 'level shift factor = %s', self.level_shift)
        if isinstance(self.diis, lib.diis.DIIS):
            logger.info(self, 'DIIS = %s', self.diis)
            logger.info(self, 'DIIS start cycle = %d', self.diis_start_cycle)
            logger.info(self, 'DIIS space = %d', self.diis.space)
        elif self.diis:
            logger.info(self, 'DIIS = %s', diis.SCF_DIIS)
            logger.info(self, 'DIIS start cycle = %d', self.diis_start_cycle)
            logger.info(self, 'DIIS space = %d', self.diis_space)
        logger.info(self, 'SCF tol = %g', self.conv_tol)
        logger.info(self, 'SCF gradient tol = %s', self.conv_tol_grad)
        logger.info(self, 'max. SCF cycles = %d', self.max_cycle)
        logger.info(self, 'direct_scf = %s', self.direct_scf)
        if self.direct_scf:
            logger.info(self, 'direct_scf_tol = %g', self.direct_scf_tol)
        if self.chkfile:
            logger.info(self, 'chkfile to save SCF result = %s', self.chkfile)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self


    def _eigh(self, h, s):
        return eig(h, s)

    @lib.with_doc(eig.__doc__)
    def eig(self, h, s):
# An intermediate call to self._eigh so that the modification to eig function
# can be applied on different level.  Different SCF modules like RHF/UHF
# redifine only the eig solver and leave the other modifications (like removing
# linear dependence, sorting eigenvlaue) to low level ._eigh
        return self._eigh(h, s)

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)

    get_fock = get_fock
    get_occ = get_occ

    @lib.with_doc(get_grad.__doc__)
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['mo_energy'],
                             envs['mo_coeff'], envs['mo_occ'],
                             overwrite_mol=False)
        return self

    @lib.with_doc(init_guess_by_minao.__doc__)
    def init_guess_by_minao(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    @lib.with_doc(init_guess_by_atom.__doc__)
    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from superpostion of atomic densties.')
        return init_guess_by_atom(mol)

    @lib.with_doc(init_guess_by_1e.__doc__)
    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    @lib.with_doc(init_guess_by_chkfile.__doc__)
    def init_guess_by_chkfile(self, chkfile=None, project=True):
        if isinstance(chkfile, gto.Mole):
            raise TypeError('''
    You see this error message because of the API updates.
    The first argument is chkfile name.''')
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)
    def from_chk(self, chkfile=None, project=True):
        return self.init_guess_by_chkfile(chkfile, project)
    from_chk.__doc__ = init_guess_by_chkfile.__doc__

    def get_init_guess(self, mol=None, key='minao'):
        if mol is None:
            mol = self.mol
        if key.lower() == '1e':
            dm = self.init_guess_by_1e(mol)
        elif getattr(mol, 'natm', 0) == 0:
            logger.info(self, 'No atom found in mol. Use 1e initial guess')
            dm = self.init_guess_by_1e(mol)
        elif key.lower() == 'atom':
            dm = self.init_guess_by_atom(mol)
        elif key.lower().startswith('chk'):
            try:
                dm = self.init_guess_by_chkfile()
            except (IOError, KeyError):
                logger.warn(self, 'Fail in reading %s. Use MINAO initial guess',
                            self.chkfile)
                dm = self.init_guess_by_minao(mol)
        else:
            dm = self.init_guess_by_minao(mol)
        if self.verbose >= logger.DEBUG1:
            s = self.get_ovlp()
            if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
                nelec = numpy.einsum('ij,ji', dm, s).real
            else:  # UHF
                nelec =(numpy.einsum('ij,ji', dm[0], s).real,
                        numpy.einsum('ij,ji', dm[1], s).real)
            logger.debug1(self, 'Nelec from initial guess = %s', nelec)
        return dm

    # full density matrix for RHF
    @lib.with_doc(make_rdm1.__doc__)
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return make_rdm1(mo_coeff, mo_occ)

    energy_elec = energy_elec
    energy_tot = energy_tot

    def energy_nuc(self):
        return self.mol.energy_nuc()

    def scf(self, dm0=None):
        '''main routine for SCF

        Kwargs:
            dm0 : ndarray
                If given, it will be used as the initial guess density matrix

        Examples:

        >>> import numpy
        >>> from pyscf import gto, scf
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1')
        >>> mf = scf.hf.SCF(mol)
        >>> dm_guess = numpy.eye(mol.nao_nr())
        >>> mf.kernel(dm_guess)
        converged SCF energy = -98.5521904482821
        -98.552190448282104
        '''
        cput0 = (time.clock(), time.time())

        self.dump_flags()
        self.build(self.mol)
        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, self.conv_tol, self.conv_tol_grad,
                       dm0=dm0, callback=self.callback,
                       conv_check=self.conv_check)

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    def kernel(self, dm0=None):
        return self.scf(dm0)
    kernel.__doc__ = scf.__doc__

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g after %d cycles',
                        self.e_tot, self.max_cycle)
        return self

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        if mol.cart:
            intor = 'int2e_cart'
        else:
            intor = 'int2e_sph'
        opt = _vhf.VHFOpt(mol, intor, 'CVHFnrs8_prescreen',
                          'CVHFsetnr_direct_scf',
                          'CVHFsetnr_direct_scf_dm')
        opt.direct_scf_tol = self.direct_scf_tol
        return opt

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        dm = numpy.asarray(dm)
        nao = dm.shape[-1]
        vj, vk = get_jk(mol, dm.reshape(-1,nao,nao), hermi, self.opt)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj.reshape(dm.shape), vk.reshape(dm.shape)

    def get_j(self, mol=None, dm=None, hermi=1):
        '''Compute J matrix for the given density matrix.
        '''
        return self.get_jk(mol, dm, hermi)[0]

    def get_k(self, mol=None, dm=None, hermi=1):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(mol, dm, hermi)[1]

    @lib.with_doc(get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
# Be carefule with the effects of :attr:`SCF.direct_scf` on this function
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.asarray(vhf_last) + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    @lib.with_doc(analyze.__doc__)
    def analyze(self, verbose=None, with_meta_lowdin=True, **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    @lib.with_doc(mulliken_pop.__doc__)
    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, s=s, verbose=verbose)

    @lib.with_doc(mulliken_meta.__doc__)
    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method='ANO', s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_meta(mol, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)
    def mulliken_pop_meta_lowdin_ao(self, *args, **kwargs):
        return self.mulliken_meta(*args, **kwargs)
    def pop(self, *args, **kwargs):
        return self.mulliken_meta(*args, **kwargs)
    pop.__doc__ = mulliken_meta.__doc__

    canonicalize = canonicalize

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, mol=None, dm=None, unit_symbol=None, verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm =self.make_rdm1()
        if unit_symbol is None: unit_symbol='Debye'
        return dip_moment(mol, dm, unit_symbol, verbose=verbose)

    def _is_mem_enough(self):
        nbf = self.mol.nao_nr()
        return nbf**4/1e6+lib.current_memory()[0] < self.max_memory*.95

    def density_fit(self, auxbasis=None, with_df=None):
        import pyscf.df.df_jk
        return pyscf.df.df_jk.density_fit(self, auxbasis, with_df)

    def x2c1e(self):
        import pyscf.scf.x2c
        return pyscf.scf.x2c.sfx2c1e(self)
    def x2c(self):
        return self.x2c1e()

    def newton(self):
        import pyscf.scf.newton_ah
        return pyscf.scf.newton_ah.newton(self)

    def update(self, chkfile=None):
        '''Read attributes from the chkfile then replace the attributes of
        current object.  See also mf.update_from_chk
        '''
        return self.update_from_chk(chkfile)
    def update_from_chk(self, chkfile=None):
        from pyscf.scf import chkfile as chkmod
        if chkfile is None: chkfile = self.chkfile
        self.__dict__.update(chkmod.load(chkfile, 'scf'))
        return self

    as_scanner = as_scanner

    @property
    def hf_energy(self):
        sys.stderr.write('WARN: Attribute .hf_energy will be removed in PySCF v1.1. '
                         'It is replaced by attribute .e_tot\n')
        return self.e_tot
    @hf_energy.setter
    def hf_energy(self, x):
        sys.stderr.write('WARN: Attribute .hf_energy will be removed in PySCF v1.1. '
                         'It is replaced by attribute .e_tot\n')
        self.hf_energy = x

    @property
    def level_shift_factor(self):
        sys.stderr.write('WARN: Attribute .level_shift_factor will be removed in PySCF v1.1. '
                         'It is replaced by attribute .level_shift\n')
        return self.level_shift
    @level_shift_factor.setter
    def level_shift_factor(self, x):
        sys.stderr.write('WARN: Attribute .level_shift_factor will be removed in PySCF v1.1. '
                         'It is replaced by attribute .level_shift\n')
        self.level_shift = x

    @property
    def damp_factor(self):
        sys.stderr.write('WARN: Attribute .damp_factor will be removed in PySCF v1.1. '
                         'It is replaced by attribute .damp\n')
        return self.damp
    @damp_factor.setter
    def damp_factor(self, x):
        sys.stderr.write('WARN: Attribute .damp_factor will be removed in PySCF v1.1. '
                         'It is replaced by attribute .damp\n')
        self.damp = x


############


class RHF(SCF):
    __doc__ = SCF.__doc__

    def __init__(self, mol):
        if mol.nelectron != 1 and (mol.nelectron % 2) != 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
# Note: self._eri requires large amount of memory
        SCF.__init__(self, mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1):
# Note the incore version, which initializes an _eri array in memory.
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = dot_eri_dm(self._eri, dm, hermi)
        else:
            vj, vk = SCF.get_jk(self, mol, dm, hermi)
        return vj, vk

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        from pyscf.scf import addons
        addons.convert_to_rhf(mf, self)
        return self

    def stability(self, internal=True, external=False, verbose=None):
        '''
        RHF/RKS stability analysis.

        See also pyscf.scf.stability.rhf_stability function.

        Kwargs:
            internal : bool
                Internal stability, within the RHF optimization space.
            external : bool
                External stability. Including the RHF -> UHF and real -> complex
                stability analysis.

        Returns:
            New orbitals that are more close to the stable condition.  The return
            value includes two set of orbitals.  The first corresponds to the
            internal stablity and the second corresponds to the external stability.
        '''
        from pyscf.scf.stability import rhf_stability
        return rhf_stability(self, internal, external, verbose)

    def nuc_grad_method(self):
        from pyscf.grad import rhf
        return rhf.Gradients(self)


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [['He', (0, 0, 0)], ]
    mol.basis = 'ccpvdz'
    mol.build()

##############
# SCF result
    method = RHF(mol)
    method.init_guess = '1e'
    energy = method.scf()
    print(energy)
