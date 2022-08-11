import numpy as np
import baccoemu_beta as baccoemu
import pyccl as ccl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BACCOCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations using the BACCO emulator.

    Args:
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
    """
    def __init__(self, log10k_min=-2, log10k_max=0, nk_per_decade=20, h=None, k_filter=None):

        self.h = h
        if np.log10(10**log10k_min/self.h) < -2:
            logger.info('Setting k_min to BACCO default.')
            log10k_min = np.log10((1e-2)*self.h)
        if np.log10(10**log10k_max/self.h) > np.log10(0.75):
            logger.info('Setting k_max to BACCO default.')
            log10k_max = np.log10(0.75*self.h)
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)

        self.bacco_emu = baccoemu.Lbias_expansion()
        self.bacco_table = None

        # redshifts for creating the Pk-2d object
        z = np.array([1.5, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
        self.a_s = 1. / (1. + z)
        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.ks/k_filter)**2)
        else:
            self.wk_low = np.ones(nk_total)


    def _cosmo_to_bacco(self, cosmo):
        pars = {
            'omega_matter': cosmo['Omega_b']+cosmo['Omega_c'], # No massive neutrinos
            'omega_baryon': cosmo['Omega_b'],
            'hubble': cosmo['H0']/100.,
            'ns': cosmo['n_s'],
            #TODO: Remove massive neutrinos from sigma_8
            'sigma8': cosmo['sigma8'],
            'neutrino_mass': np.sum(cosmo['m_nu']),
            'w0': cosmo['w0'],
            'wa': cosmo['wa'],
            'expfactor': 1.  # random scale factor just to initialize
        }
        return pars

    def update_pk(self, cosmo):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """

        # translate the pyccl cosmology parameters into bacco notation
        pars = self._cosmo_to_bacco(cosmo)
        # convert k_s [Mpc^-1] into h Mpc^-1 units just for the calculation
        k = self.ks / pars['hubble']

        # 10 redshifts, 15 combinations between bias params, and the ks (added b0)
        num_comb = 15
        pk2d_bacco = np.zeros((len(self.a_s), num_comb, len(self.ks)))

        # compute the power for each redshift
        # t1 = time.time()
        for i in range(len(self.a_s)):
            pars['expfactor'] = self.a_s[i]
            # call the emulator of the nonlinear 15 lagrangian bias expansion terms, shape is (15, len(k))
            # Use BACCO for z<=1.5
            if self.a_s[i] >= 0.4:
                _, pnn = self.bacco_emu.get_nonlinear_pnn(pars, k=k)
                pk2d_bacco[i, :, :] = pnn
            # Use LPT emulator for z>1.5
            else:
                _, plpt = self.bacco_emu.get_lpt_pk(pars, k=k) # B.H. I think tiny bug
                pk2d_bacco[i, :, :] = plpt

        # convert the spit out result from (Mpc/h)^3 to Mpc^3 (bacco uses h units, but pyccl doesn't)
        pk2d_bacco /= pars['hubble'] ** 3
        self.bacco_table = pk2d_bacco

    def get_pgg(self, Pnl, b11, b21, bs1, b12, b22, bs2,
                bk21=None, bk22=None, Pgrad=None):
        if self.bacco_table is None:
            raise ValueError("Please initialise BACCO calculator")
        # Clarification:
        # BACCO uses the following expansion for the galaxy overdensity:
        #   d_g = b1 d + b2 d2^2 + bs s^2 + bnabla2d nabla^2 d
        # (see Eq. 1 of https://arxiv.org/abs/2101.12187).
        #
        # The BACCO table below contains the following power spectra
        # in order:
        # <1,1>
        # <1,d>
        # <1,d^2>
        # <1,s^2>
        # <1,nabla^2 d>
        # <d,d>
        # <d,d^2>
        # <d,s^2>
        # <d,nabla^2 d>
        # <d^2,d^2> (!)
        # <d^2,s^2> (!)
        # <d^2,nabla^2 d>
        # <s^2,s^2> (!)
        # <s^2,nabla^2 d>
        # <nabla^2 d,nabla^2 d>
        #
        # So:
        #   a) The spectra involving b2 are for d^2 - convert to d^2/2
        #   b) The spectra involving bs are for s^2 - convert to s^2/2
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving d^2 and s2 to
        # make the definitions of b2, bs equivalent to what we have adopted for
        # the EPT and LPT expansions.
        bL11 = b11-1
        bL12 = b12-1
        if Pnl is None:
            Pdmdm = self.bacco_table[:, 0, :]
            Pdmd1 = self.bacco_table[:, 1, :]
            Pd1d1 = self.bacco_table[:, 5, :]
            pgg = (Pdmdm + (bL11+bL12)[:, None] * Pdmd1 +
                   (bL11*bL12)[:, None] * Pd1d1)
        else:
            pgg = (b11*b12)[:, None]*Pnl
        Pdmd2 = 0.5*self.bacco_table[:, 2, :]
        Pd1d2 = 0.5*self.bacco_table[:, 6, :]
        Pd2d2 = 0.25*self.bacco_table[:, 9, :]*self.wk_low[None, :]
        Pdms2 = 0.5*self.bacco_table[:, 3, :]
        Pd1s2 = 0.5*self.bacco_table[:, 7, :]
        Pd2s2 = 0.25*self.bacco_table[:, 10, :]*self.wk_low[None, :]
        Ps2s2 = 0.25*self.bacco_table[:, 12, :]*self.wk_low[None, :]
        #TODO: OK to use low-pass filter?

        #TODO: what to do with nonlocal bias?
        if Pgrad is None:
            Pgrad = Pnl
        Pd1k2 = 0.5*Pgrad * (self.ks**2)[None, :]

        if bk21 is None:
            bk21 = np.zeros_like(self.a_s)
        if bk22 is None:
            bk22 = np.zeros_like(self.a_s)

        pgg += ((b21 + b22)[:, None] * Pdmd2 +
                (bs1 + bs2)[:, None] * Pdms2 +
                (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
                (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
                (b21*b22)[:, None] * Pd2d2 +
                (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
                (bs1*bs2)[:, None] * Ps2s2 +
                (b12*bk21+b11*bk22)[:, None] * Pd1k2)

        return pgg

    def get_pgm(self, Pnl, b1, b2, bs, bk2=None, Pgrad=None):

        if self.bacco_table is None:
            raise ValueError("Please initialise BACCO calculator")

        bL1 = b1-1
        if Pnl is None:
            Pdmdm = self.bacco_table[:, 0, :]
            Pdmd1 = self.bacco_table[:, 1, :]
            pgm = Pdmdm + bL1[:, None] * Pdmd1
        else:
            pgm = b1[:, None]*Pnl
        Pdmd2 = 0.5*self.bacco_table[:, 2, :]
        Pdms2 = 0.5*self.bacco_table[:, 3, :]
        if Pgrad is None:
            Pgrad = Pnl
        Pd1k2 = 0.5*Pgrad * (self.ks**2)[None, :]

        # B.H. fixing bug
        if bk2 is None:
            bk2 = np.zeros_like(self.a_s)
        
        pgm += (b2[:, None] * Pdmd2 +
                bs[:, None] * Pdms2 +
                bk2[:, None] * Pd1k2)

        return pgm

    def get_pmm(self):

        if self.bacco_table is None:
            raise ValueError("Please initialise BACCO calculator")

        pmm = self.bacco_table[:, 0, :]

        return pmm


def get_bacco_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                 nonlin_pk_type='nonlinear',
                 nonloc_pk_type='nonlinear',
                 extrap_order_lok=1, extrap_order_hik=2):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

    .. note:: The full non-linear model for the cross-correlation
              between number counts and intrinsic alignments is
              still work in progress in FastPT. As a workaround
              CCL assumes a non-linear treatment of IAs, but only
              linearly biased number counts.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        tracer1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the first
            tracer being correlated.
        ptc (:class:`PTCalculator`): a perturbation theory
            calculator.
        tracer2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the second
            tracer being correlated. If `None`, the auto-correlation
            of the first tracer will be returned.
        nonlin_pk_type (str): type of 1-loop matter power spectrum
            to use. 'linear' for linear P(k), 'nonlinear' for the internal
            non-linear power spectrum, 'spt' for standard perturbation
            theory power spectrum. Default: 'nonlinear'.
        nonloc_pk_type (str): type of "non-local" matter power spectrum
            to use (i.e. the cross-spectrum between the overdensity and
            its Laplacian divided by :math:`k^2`). Same options as
            `nonlin_pk_type`. Default: 'nonlinear'.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        :class:`~pyccl.nl_pt.power.PTCalculator`: PT Calc [optional]
    """

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, ccl.nl_pt.PTTracer):
        raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    if not isinstance(tracer2, ccl.nl_pt.PTTracer):
        raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    if not isinstance(ptc, BACCOCalculator):
        raise TypeError("ptc should be of type `BACCOCalculator`")
    # z
    z_arr = 1. / ptc.a_s - 1

    if nonlin_pk_type == 'nonlinear':
        Pnl = np.array([ccl.nonlin_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'linear':
        Pnl = np.array([ccl.linear_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'spt':
        Pnl = None
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))

    Pgrad = None
    if (((tracer1.type == 'NC') or (tracer2.type == 'NC')) and
            (nonloc_pk_type != nonlin_pk_type)):
        if nonloc_pk_type == 'nonlinear':
            Pgrad = np.array([ccl.nonlin_matter_power(cosmo, ptc.ks, a)
                              for a in ptc.a_s])
        elif nonloc_pk_type == 'linear':
            Pgrad = np.array([ccl.linear_matter_power(cosmo, ptc.ks, a)
                              for a in ptc.a_s])
        elif nonloc_pk_type == 'spt':
            Pgrad = None
        elif nonloc_pk_type == 'lpt':
            Pgrad = ptc.get_pmm()
        else:
            raise NotImplementedError("Non-local option %s "
                                      "not implemented yet" %
                                      (nonloc_pk_type))

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if hasattr(tracer1, 'bk2'):
            bk21 = tracer1.bk2(z_arr)
        else:
            bk21 = None
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            if hasattr(tracer2, 'bk2'):
                bk22 = tracer2.bk2(z_arr)
            else:
                bk22 = None

            p_pt = ptc.get_pgg(Pnl,
                               b11, b21, bs1,
                               b12, b22, bs2,
                               bk21, bk22,
                               Pgrad)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(Pnl, b11, b21, bs1, bk21, Pgrad)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            if hasattr(tracer2, 'bk2'):
                bk22 = tracer2.bk2(z_arr)
            else:
                bk22 = None
            p_pt = ptc.get_pgm(Pnl, b12, b22, bs2, bk22, Pgrad)
        elif (tracer2.type == 'M'):
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = ccl.Pk2D(a_arr=ptc.a_s,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt,
                     is_logp=False)
    return pt_pk
