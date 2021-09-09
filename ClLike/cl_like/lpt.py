import numpy as np
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
import pyccl as ccl


class LPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on velocileptors
    (https://github.com/sfschen/velocileptors).

    Args:
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
    """
    def __init__(self, log10k_min=-4, log10k_max=2,
                 nk_per_decade=20, a_arr=None, h=None, k_filter=None):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        if a_arr is None:
            a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        self.a_s = a_arr
        self.h = h
        self.lpt_table = None
        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.ks/k_filter)**2)
        else:
            self.wk_low = np.ones(nk_total)

    def update_pk(self, pk, Dz):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if Dz.shape != self.a_s.shape:
            raise ValueError("Input growth has wrong shape")
        cleft = RKECLEFT(self.ks/self.h, pk*self.h**3)
        self.lpt_table = []
        for D in Dz:
            cleft.make_ptable(D=D, kmin=self.ks[0]/self.h,
                              kmax=self.ks[-1]/self.h, nk=self.ks.size)
            self.lpt_table.append(cleft.pktable)
        self.lpt_table = np.array(self.lpt_table)
        self.lpt_table /= self.h**3

    def get_pgg(self, Pnl, b11, b21, bs1, b12, b22, bs2):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        # Clarification:
        # CLEFT uses the followint expansion for the galaxy overdensity:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2
        # (see Eq. 4.4 of https://arxiv.org/pdf/2005.00523.pdf).
        # To add to the confusion, this is different from the prescription
        # used by EPT, where s^2 is divided by 2 :-|
        #
        # The LPT table below contains the following power spectra
        # in order:
        #  <1,1>
        #  2*<1,d>
        #  <d,d>
        #  2*<1,d^2/2>
        #  2*<d,d^2/2>
        #  <d^2/2,d^2/2> (!)
        #  2*<1,s^2>
        #  2*<d,s^2>
        #  2*<d^2/2,s^2> (!)
        #  <s^2,s^2> (!)
        #
        # So:
        #   a) The cross-correlations need to be divided by 2.
        #   b) The spectra involving b2 are for d^2/2, NOT d^2!!
        #   c) The spectra invoving bs are for s^2, NOT s^2/2!!
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving s2 to
        # make the definition of bs equivalent in the EPT and LPT
        # expansions.
        bL11 = b11-1
        bL12 = b12-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            Pd1d1 = self.lpt_table[:, :, 3]
            pgg = (Pdmdm + (bL11+bL12)[:, None] * Pdmd1 +
                   (bL11*bL12)[:, None] * Pd1d1)
        else:
            pgg = (b11*b12)[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pd1d2 = 0.5*self.lpt_table[:, :, 5]
        Pd2d2 = self.lpt_table[:, :, 6]*self.wk_low[None, :]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pd1s2 = 0.25*self.lpt_table[:, :, 8]
        Pd2s2 = 0.25*self.lpt_table[:, :, 9]*self.wk_low[None, :]
        Ps2s2 = 0.25*self.lpt_table[:, :, 10]*self.wk_low[None, :]

        pgg += ((b21 + b22)[:, None] * Pdmd2 +
                (bs1 + bs2)[:, None] * Pdms2 +
                (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
                (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
                (b21*b22)[:, None] * Pd2d2 +
                (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
                (bs1*bs2)[:, None] * Ps2s2)
        return pgg

    def get_pgm(self, Pnl, b1, b2, bs):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        bL1 = b1-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            pgm = Pdmdm + bL1[:, None] * Pdmd1
        else:
            pgm = b1[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]

        pgm += (b2[:, None] * Pdmd2 +
                bs[:, None] * Pdms2)
        return pgm


def get_lpt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                 nonlin_pk_type='nonlinear',
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

    if not isinstance(ptc, LPTCalculator):
        raise TypeError("ptc should be of type `LPTCalculator`")
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

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            p_pt = ptc.get_pgg(Pnl,
                               b11, b21, bs1,
                               b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(Pnl, b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgm(Pnl, b12, b22, bs2)
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
