#!/usr/bin/env python
#
#

'''
Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import pw_helper
import sys
import time
import numpy as np
import h5py
from pyscf.scf import hf
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf.hf import make_rdm1
from pyscf.pbc import tools
from pyscf.pbc.gto import ewald
from pyscf.pbc.gto import ecp
from pyscf.pbc.gto.pseudo import get_pp
from pyscf.pbc.scf import chkfile
from pyscf.pbc.scf import addons
from pyscf.pbc.gto.pseudo import pp_pw

def get_ovlp(cell, gridsize=0):
    #gridlength=len(self.pw_grid_params[7])

    return np.identity(gridsize)

def get_hcore(cell, kpt=np.zeros(3)):
    hcore = get_t(cell, kpt)
    if cell.pseudo:
        hcore += get_pp(cell, kpt)
    else:
        hcore += get_nuc(cell, kpt)
    if len(cell._ecpbas) > 0:
        hcore += ecp.ecp_int(cell, kpt)

    return hcore

def get_t(cell, indgk, j, Gd, gridsize, gh, fock, k):
    '''Get the kinetic energy AO matrix.
    '''
    #t_matrix = np.zeros([gridsize,gridsize])
    indices=indgk[j,:len(gh)]
    gh=Gd[indices]+k
    gh2=np.einsum('ij,ij->i',gh,gh)/2.+fock.diagonal()
    return gh2
    #np.fill_diagonal(t_matrix,gh2)
    #return t_matrix

def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    from pyscf.pbc import df
    return df.FFTDF(cell).get_nuc(kpt)


def get_j(cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpt_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
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
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which J is evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    from pyscf.pbc import df
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpt_band, with_k=False)[0]


def get_jk(mf, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpt_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
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
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    from pyscf.pbc import df
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpt_band, exxdiv=mf.exxdiv)


def get_bands(mf, kpt_band, cell=None, dm=None, kpt=None):
    '''Get energy bands at a given (arbitrary) 'band' k-point.

    Returns:
        mo_energy : (nao,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nao) ndarray
            Band orbitals psi_n(k)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    fock = (mf.get_hcore(kpt=kpt_band) +
            mf.get_veff(cell, dm, kpt=kpt, kpt_band=kpt_band))
    s1e = mf.get_ovlp(kpt=kpt_band)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    return mo_energy, mo_coeff


def init_guess_by_chkfile(cell, chkfile_name, project=True, kpt=None):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, (nao,nao) ndarray
    '''
    chk_cell, scf_rec = chkfile.load_scf(chkfile_name)
    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if kpt is None:
        kpt = np.zeros(3)
    if 'kpt' in scf_rec:
        chk_kpt = scf_rec['kpt']
    elif 'kpts' in scf_rec:
        kpts = scf_rec['kpts'] # the closest kpt from KRHF results
        where = np.argmin(lib.norm(kpts-kpt, axis=1))
        chk_kpt = kpts[where]
        if mo.ndim == 3:  # KRHF:
            mo = mo[where]
            mo_occ = mo_occ[where]
        else:
            mo = mo[:,where]
            mo_occ = mo_occ[:,where]
    else:  # from molecular code
        chk_kpt = np.zeros(3)

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell, chk_kpt-kpt)
        else:
            return mo
    if mo.ndim == 2:
        dm = make_rdm1(fproj(mo), mo_occ)
    else:  # UHF
        dm =(make_rdm1(fproj(mo[0]), mo_occ[0]) +
             make_rdm1(fproj(mo[1]), mo_occ[1]))

    # Real DM for gamma point
    if kpt is None or np.allclose(kpt, 0):
        dm = dm.real
    return dm


def dot_eri_dm(eri, dm, hermi=0):
    '''Compute J, K matrices in terms of the given 2-electron integrals and
    density matrix. eri or dm can be complex.

    Args:
        eri : ndarray
            complex integral array with N^4 elements (N is the number of orbitals)
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
    '''
    dm = np.asarray(dm)
    if eri.dtype == np.complex128:
        nao = dm.shape[-1]
        eri = eri.reshape((nao,)*4)
        def contract(dm):
            vj = np.einsum('ijkl,ji->kl', eri, dm)
            vk = np.einsum('ijkl,jk->il', eri, dm)
            return vj, vk
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            vj, vk = contract(dm)
        else:
            vjk = [contract(dmi) for dmi in dm]
            vj = lib.asarray([v[0] for v in vjk]).reshape(dm.shape)
            vk = lib.asarray([v[1] for v in vjk]).reshape(dm.shape)
    elif dm.dtype == np.complex128:
        vjR, vkR = hf.dot_eri_dm(eri, dm.real, hermi)
        vjI, vkI = hf.dot_eri_dm(eri, dm.imag, hermi)
        vj = vjR + vjI*1j
        vk = vkR + vkI*1j
    else:
        vj, vk = hf.dot_eri_dm(eri, dm, hermi)
    return vj, vk


class RHF_PW(hf.RHF):
    '''RHF class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.

        exxdiv : str
            Exchange divergence treatment, can be one of

            | None : ignore G=0 contribution in exchange integral
            | 'ewald' : Ewald summation for G=0 in exchange integral

        with_df : density fitting object
            Default is the FFT based DF model. For all-electron calculation,
            MDF model is favored for better accuracy.  See also :mod:`pyscf.pbc.df`.
    '''
    def __init__(self, cell, kpt=np.zeros(3), exxdiv='ewald'):
        from pyscf.pbc import df
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        hf.RHF.__init__(self, cell)
        self.pw_grid_params=[pw_helper.return_grids()]
        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        self.kpt = kpt
        self.direct_scf = False
        self._keys = self._keys.union(['cell', 'exxdiv', 'with_df'])

    @property
    def kpt(self):
        return self.with_df.kpts.reshape(3)
    @kpt.setter
    def kpt(self, x):
        self.with_df.kpts = np.reshape(x, (-1,3))

    def dump_flags(self):
        hf.RHF.dump_flags(self)
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'kpt = %s', self.kpt)
        logger.info(self, 'DF object = %s', self.with_df)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)

    ##########TO DO: PASS ITERATION NUMBER j FROM kernel_pw()#####################
    def get_hcore(self, j, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
   
        #test without pp for now, figure out how to get from pyscf later
        #t is compiled in get_fock
        
        indgk=self.pw_grid_params[10]
        GH = self.pw_grid_params[6]
        Gd=self.pw_grid_params[3]
        gridsize=len(self.pw_grid_params[7])
        #return nuc+get_t(cell,indgk,j,Gd,gridsize,kpt)
        return None 
        #return get_t(cell,indgk,j,Gd,gridsize,GH,kpt)

    def get_fock_no_pypseudo(self, j, vg, h1e, cell=None, kpt=None):
        if cell is None: cell=self.cell       
        if kpt is None: kpt = self.kpt
    
        GH = self.pw_grid_params[6]
        Gd = self.pw_grid_params[3]
        gridsize = len(self.pw_grid_params[7])
        GH_ind = self.pw_grid_params[7]   
        mill_Gd = self.pw_grid_params[11]
        indgk = self.pw_grid_params[10]
        gs = self.pw_grid_params[0]
        indg = self.pw_grid_params[9]   
        Gd2 = self.pw_grid_params[5]
        gkind=indgk[j,:len(GH_ind)]

        fock = np.zeros((len(GH_ind),len(GH_ind)),dtype='complex128')

        for aa in range(0,len(GH_ind)):
            ik = indgk[j][aa]
            gdiff = mill_Gd[ik]-mill_Gd[gkind[aa:]]+np.array(gs)
            inds = indg[gdiff.T.tolist()]
            #loc = pp_pw.get_pploc_gth_pw(cell, Gd2[inds])
            fock[aa,aa:]=vg[inds]

        tdiag = get_t(cell,indgk,j,Gd,gridsize,GH,fock,kpt)
        np.fill_diagonal(fock,tdiag)

        return fock

    def get_volume_gridsize(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        v = self.pw_grid_params[12]
        grid_dim = self.pw_grid_params[2]

        return v, grid_dim

    def get_density_guess_pw(self, cell=None, kpt=None):
        v = self.pw_grid_params[12]
        Gd = self.pw_grid_params[3]
        grid_dim=self.pw_grid_params[2]
        nelec = self.mol.nelectron

        l = len(Gd)
        a = np.zeros(grid_dim,dtype='float64')
        for x in range(0, 15):
            for y in range(0,15):
                for z in range(0,15):
                    a[x][y][z]= nelec/(v)
        return a

    def get_density_shell(self, cell=None, kpt=None):
        grid_dim = self.pw_grid_params[2]
        return np.zeros(grid_dim, dtype='float64')


    def gen_dens(self, j, eigvec, cell=None, kpt=None):
        from pyscf.pbc.dft import pw_helper

        v = self.pw_grid_params[12]
        indgk = self.pw_grid_params[10]
        GH_ind = self.pw_grid_params[7]
        mill_Gd = self.pw_grid_params[11]
        grid_dim = self.pw_grid_params[2]
        
        ############TO DO: GET nbands FROM PYSCF########################
        nbands = 4

        rho_add=np.zeros(grid_dim,dtype='float64')
        kpts = self.kpt
   
        if isinstance(kpts[0], list) is False:
            kpts = [kpts]

        for pp in range(nbands):
            aux=np.zeros(grid_dim,dtype='complex128')
            for tt in range(0, len(GH_ind)):
               ik=indgk[j][tt]
               aux[pw_helper.get_mill(ik,mill_Gd,grid_dim)]=eigvec[tt,pp]
            aux=(1./np.sqrt(v))*np.fft.fftn(aux)
            rho_add+=(2./len(kpts))*np.absolute(aux)**2

        return rho_add

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if gridsize is None: gridsize=len(self.pw_grid_params[7])
        return get_ovlp(cell, gridsize)

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        Note the incore version, which initializes an _eri array in memory.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (time.clock(), time.time())

        if (kpt_band is None and  # 4 indices of ._eri should have same kpt
            (self.exxdiv == 'ewald' or not self.exxdiv) and
            (self._eri is not None or cell.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                logger.debug(self, 'Building PBC AO integrals incore')
                self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            vj, vk = dot_eri_dm(self._eri, dm, hermi)

            if self.exxdiv == 'ewald':
                from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
                # G=0 is not inculded in the ._eri integrals
                _ewald_exxdiv_for_G0(self.cell, kpt, [dm], [vk])
        else:
            vj, vk = self.with_df.get_jk(dm, hermi, kpt, kpt_band,
                                         exxdiv=self.exxdiv)

        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Compute J matrix for the given density matrix.
        '''
        #return self.get_jk(cell, dm, hermi, kpt, kpt_band)[0]
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (time.clock(), time.time())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (kpt_band is None and
            (self._eri is not None or cell.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                logger.debug(self, 'Building PBC AO integrals incore')
                self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            vj, vk = dot_eri_dm(self._eri, dm.reshape(-1,nao,nao), hermi)
        else:
            vj = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi,
                                     kpt, kpt_band, with_k=False)[0]
        logger.timer(self, 'vj', *cpu0)
        return vj.reshape(dm.shape)

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpt_band)[1]

    def get_veff(self, rho, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpt_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        Gd_ind = self.pw_grid_params[4]
        Gd2 = self.pw_grid_params[5]
        g2_cutoff=1e-8

        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        vj, vk = self.get_jk(Gd_ind, rho, g2_cutoff, Gd2, cell, dm, hermi, kpt, kpt_band)
        return vj - vk * .5

    def get_jk_incore(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG, kpt=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.
        '''
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if self._eri is None:
            self._eri = self.with_df.get_ao_eri(kpt, compact=True)
        return self.get_jk(cell, dm, hermi, verbose, kpt)

    def energy_nuc(self):
        return self.cell.energy_nuc()

    get_bands = get_bands

    def init_guess_by_chkfile(self, chk=None, project=True, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return init_guess_by_chkfile(self.cell, chk, project, kpt)
    def from_chk(self, chk=None, project=True, kpt=None):
        return self.init_guess_by_chkfile(chk, project, kpt)

    def dump_chk(self, envs):
        hf.RHF.dump_chk(self, envs)
        if self.chkfile:
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpt'] = self.kpt
        return self

    def _is_mem_enough(self):
        nao = self.cell.nao_nr()
        if abs(self.kpt).sum() < 1e-9:
            mem_need = nao**4*8/4/1e6
        else:
            mem_need = nao**4*16/1e6
        return mem_need + lib.current_memory()[0] < self.max_memory*.95

