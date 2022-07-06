import pyccl as ccl
import numpy as np

# For HaloProfileCIBM21
from pyccl.halos.profiles import HaloProfile, HaloProfileNFW
from pyccl.halos.profiles_2pt import Profile2pt
from pyccl.halos.concentration import Concentration
from scipy.integrate import simps

# For IvTracer
from astropy.io import fits
from pyccl import ccllib as lib
from pyccl.core import check
from pyccl.background import comoving_radial_distance
from pyccl.pyutils import _check_array_params, NoneArr, _vectorize_fn6, _get_spline1d_arrays
from scipy.interpolate import interp1d
from cibprof import Profile2ptCIB

class HalomodCorrection(object):
    """Provides methods to estimate the correction to the halo
    model in the 1h - 2h transition regime.

    Args:
        cosmo (:obj:`ccl.Cosmology`): cosmology.
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.
    """
    def __init__(self,
                 k_range=[1E-1, 5], nlk=20,
                 z_range=[0., 1.], nz=16):
        from scipy.interpolate import interp2d

        cosmo = ccl.CosmologyVanillaLCDM()
        lkarr = np.linspace(np.log10(k_range[0]),
                            np.log10(k_range[1]),
                            nlk)
        karr = 10.**lkarr
        zarr = np.linspace(z_range[0], z_range[1], nz)

        pk_hm = np.array([ccl.halomodel_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        ratio = pk_hf / pk_hm

        self.rk_func = interp2d(lkarr, 1/(1+zarr), ratio,
                                bounds_error=False, fill_value=1)

    def rk_interp(self, k, a):
        """
        Returns the halo model correction for an array of k
        values at a given redshift.

        Args:
            k (float or array): wavenumbers in units of Mpc^-1.
            a (float): value of the scale factor.
        """
        return self.rk_func(np.log10(k), a)-1


class ConcentrationDuffy08M500c(ccl.halos.Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486) extended to Delta = 500-critical.
    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08M500c'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08M500c, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = ccl.halos.MassDef(500, 'critical')

    def _check_mdef(self, mdef):
        if (mdef.Delta != 500) or (mdef.rho_type != 'critical'):
            return True
        return False

    def _setup(self):
        self.A = 3.67
        self.B = -0.0903
        self.C = -0.51

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)


# Implementation of HaloProfileCIBM21
class HaloProfileCIBM21(HaloProfile):
    """ CIB profile implementing the model by Maniyar et al.
    (A&A 645, A40 (2021)).

    The parametrization for the mean profile is:

    .. math::
        \\rho_{\\rm SFR}(z) = \\rho_{\\rm SFR}^{\\rm cen}(z)+
        \\rho_{\\rm SFR}^{\\rm sat}(z),

    where the star formation rate (SFR) density from centrals and satellites is
    modelled as:

    .. math::
        \\rho_{\\rm SFR}^{\\rm cen}(z) = \\int_{}^{}\\frac{dn}{dm},
        SFR(M_{\\rm h},z)dm,

    .. math::
        \\rho_{\\rm SFR}^{\\rm sat}(z) = \\int_{}^{}\\frac{dN}{dm},
        (\\int_{M_{\\rm min}}^{M}\\frac{dN_{\\rm sub}}{dm}SFR(M_{\\rm sub},z)dm),
        dm.

    Here, :math:`dN_{\\rm sub}/dm` is the subhalo mass function,
    and the SFR is parametrized as

    .. math::
        SFR(M,z) = \\eta(M,z)\\,
        BAR(M,z),

    where the mass dependence of the efficiency :math:'\\eta' is lognormal

    .. math::
        \\eta(M,z) = \\eta_{\\rm max},
        \\exp\\left[-\\frac{\\log_{10}^2(M/M_{\\rm eff})},
        {2\\sigma_{LM}^2(z)}\\right],

    with :math:'\\sigma_{LM}' defined as the redshift dependant logarithmic scatter in mass

    .. math::
        \\sigma_{LM}(z) = \\left\\{
        \\begin{array}{cc}
           \\sigma_{LM0} & M < M_{\\rm eff} \\\\
           \\sigma_{LM0} - \\tau max(0,z_{\\rm c}-z) & M \\geq M_{\\rm eff}
        \\end{array}
        \\right.,

    and :math:'BAR' is the Baryon Accretion Rate

    .. math::
        BAR(M,z) = \\frac{\\Omega_{\\rm b}}{\\Omega_{\\rm m}},
        MGR(M,z),

    where :math:'MGR' is the Mass Growth Rate
    
    .. math::
        MGR(M,z) = 46.1\\left(\\frac{M}{10^{12}M_{\\odot}}\\right)^{1.1},
        \\left(1+1.11z\\right)\sqrt{\\Omega_{\\rm m}(1+z)^{3}+\\Omega_{\\rm \\Lambda}},

    Args:
        cosmo (:obj:`Cosmology`): cosmology object containing
            the cosmological parameters.
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        log10meff (float): log10 of the most efficient mass.
        etamax (float) : star formation efficiency of the most efficient mass
        sigLM0 (float): logarithmic scatter in mass.
        tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
        zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
        Mmin (float): minimum subhalo mass.
    """
    name = 'CIBM21'

    def __init__(self, c_M_relation, log10meff=12.7, etamax=0.42,
                 sigLM0=1.75, tau=1.17, zc=1.5, Mmin=1E5):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.l10meff = log10meff
        self.etamax = etamax
        self.sigLM0 = sigLM0
        self.tau = tau
        self.zc = zc
        self.Mmin = Mmin
        self.pNFW = HaloProfileNFW(c_M_relation)
        super(HaloProfileCIBM21, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        """Subhalo mass function of Tinker & Wetzel (2010ApJ...719...88T)

        Args:
            Msub (float or array_like): sub-halo mass (in solar masses).
            Mparent (float): parent halo mass (in solar masses).

        Returns:
            float or array_like: average number of subhalos.
        """
        return 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)

    def update_parameters(self, log10meff=None, etamax=None,
                          sigLM0=None, tau=None, zc=None, Mmin=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            log10meff (float): log10 of the most efficient mass.
            etamax (float) : star formation efficiency of the most efficient mass
            sigLM0 (float): logarithmic scatter in mass.
            tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
            zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
            Mmin (float): minimum subhalo mass (in solar masses).
        """
        if log10meff is not None:
            self.l10meff = log10meff
        if etamax is not None:
            self.etamax = etamax
        if sigLM0 is not None:
            self.sigLM0 = sigLM0
        if tau is not None:
            self.tau = tau
        if zc is not None:
            self.zc = zc
        if Mmin is not None:
            self.Mmin = Mmin

    def sigLM(self, M, a):
        z = 1/a - 1
        if hasattr(M, "__len__"):
            sig = np.zeros_like(M)
            smallM = np.log10(M) < self.l10meff
            sig[smallM] = self.sigLM0
            sig[~smallM] = self.sigLM0 - self.tau * max(0, self.zc-z)
            return sig
        else:
            if np.log10(M) < self.l10meff:
                return self.sigLM0
            else:
                return self.sigLM0 - self.tau * max(0, self.zc-z)

    def _SFR(self, cosmo, M, a):
        Omega_b = cosmo['Omega_b']
        Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']
        Omega_L = 1 - cosmo['Omega_m']
        z = 1/a - 1
        # Efficiency - eta
        eta = self.etamax * np.exp(-0.5*((np.log(M) - np.log(10)*self.l10meff)/self.sigLM(M, a))**2)
        # Baryonic Accretion Rate - BAR
        MGR = 46.1 * (M/1e12)**1.1 * (1+1.11*z) * np.sqrt(Omega_m*(1+z)**3 + Omega_L)
        BAR = Omega_b/Omega_m * MGR
        return eta * BAR

    def _SFRcen(self, cosmo, M, a):
        fsub = 0.134
        M = M*(1-fsub)
        SFRcen = self._SFR(cosmo, M, a)
        return SFRcen

    def _SFRsat(self, cosmo, M, a):
        fsub = 0.134
        SFRsat = np.zeros_like(M)
        goodM = M >= self.Mmin
        M_use = (1-fsub)*M[goodM, None]
        nm = max(2, 3*int(np.log10(np.max(M_use)/self.Mmin)))
        Msub = np.geomspace(self.Mmin, np.max(M_use), nm+1)[None, :]
        # All these arrays are of shape [nM_parent, nM_sub]

        dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(Msub, M_use)
        SFRI = self._SFR(cosmo, Msub.flatten(), a)[None, :]
        SFRII = self._SFR(cosmo, M_use, a)*Msub/M_use
        Ismall = SFRI < SFRII
        SFR = SFRI*Ismall + SFRII*(~Ismall)

        integ = dnsubdlnm*SFR*(M_use >= Msub)
        SFRsat[goodM] = simps(integ, x=np.log(Msub))
        return SFRsat

    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)

        SFRs = self._SFRsat(cosmo, M_use, a)
        ur = 1

        prof = SFRs[:, None]*ur

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)

        SFRc = self._SFRcen(cosmo, M_use, a)
        SFRs = self._SFRsat(cosmo, M_use, a)
        uk = 1

        prof = SFRc[:, None]+SFRs[:, None]*uk

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)

        SFRc = self._SFRcen(cosmo, M_use, a)
        SFRs = self._SFRsat(cosmo, M_use, a)
        uk = 1

        prof = SFRs[:, None]*uk
        prof = 2*SFRc[:, None]*prof + prof**2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class Profile2ptCIBM21(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of McCarthy & Madhavacheril (2021PhRvD.103j3515M)).
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment for the CIB
        profile.

        Args:
            prof (:class:`HaloProfileCIBM21`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof2 (:class:`HaloProfileCIBM21`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfileCIBM21):
            raise TypeError("prof must be of type `HaloProfileCIB`")
        if prof2 is not None:
            if not isinstance(prof2, HaloProfileCIBM21):
                raise TypeError("prof must be of type `HaloProfileCIB`")
        return prof._fourier_variance(cosmo, k, M, a, mass_def)


# Implementation of IvTracer
def _Sig_MG(cosmo, a, k=None):
    """Redshift-dependent modification to Poisson equation for massless
    particles under modified gravity.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        k (float or array_like): Wavenumber for scale

    Returns:
        float or array_like: Modification to Poisson equation under \
            modified gravity at scale factor a. \
            Sig_MG is assumed to be proportional to Omega_Lambda(z), \
            see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.
    """
    return _vectorize_fn6(lib.Sig_MG, lib.Sig_MG_vec, cosmo, a, k)


def _check_background_spline_compatibility(cosmo, z):
    """Check that a redshift array lies within the support of the
    CCL background splines.
    """
    a_bg, _ = _get_spline1d_arrays(cosmo.cosmo.data.chi)
    a = 1/(1+z)

    if a.min() < a_bg.min() or a.max() > a_bg.max():
        raise ValueError(f"Tracer defined over wider redshift range than "
                         f"internal CCL splines. Tracer: "
                         f"z=[{1/a.max()-1}, {1/a.min()-1}]. Background "
                         f"splines: z=[{1/a_bg.max()-1}, {1/a_bg.min()-1}].")


def get_density_kernel(cosmo, dndz):
    """This convenience function returns the radial kernel for
    galaxy-clustering-like tracers. Given an unnormalized
    redshift distribution, it returns two arrays: chi, w(chi),
    where chi is an array of radial distances in units of
    Mpc and w(chi) = p(z) * H(z), where H(z) is the expansion
    rate in units of Mpc^-1 and p(z) is the normalized
    redshift distribution.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
            transform redshifts into distances.
        dndz (tulple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects.
            The units are arbitrary; N(z) will be normalized
            to unity.
    """
    z_n, n = _check_array_params(dndz, 'dndz')
    _check_background_spline_compatibility(cosmo, dndz[0])
    # this call inits the distance splines neded by the kernel functions
    chi = comoving_radial_distance(cosmo, 1./(1.+z_n))
    status = 0
    wchi, status = lib.get_number_counts_kernel_wrapper(cosmo.cosmo,
                                                        z_n, n,
                                                        len(z_n),
                                                        status)
    check(status, cosmo=cosmo)
    return chi, wchi


def get_lensing_kernel(cosmo, dndz, mag_bias=None):
    """This convenience function returns the radial kernel for
    weak-lensing-like. Given an unnormalized redshift distribution
    and an optional magnification bias function, it returns
    two arrays: chi, w(chi), where chi is an array of radial
    distances in units of Mpc and w(chi) is the lensing shear
    kernel (or the magnification one if `mag_bias` is not `None`).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
            transform redshifts into distances.
        dndz (tulple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects.
            The units are arbitrary; N(z) will be normalized
            to unity.
        mag_bias (tuple of arrays, optional): A tuple of arrays (z, s(z))
            giving the magnification bias as a function of redshift. If
            `None`, s=0 will be assumed
    """
    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias, 'mag_bias')
    _check_background_spline_compatibility(cosmo, dndz[0])

    # Calculate number of samples in chi
    nchi = lib.get_nchi_lensing_kernel_wrapper(z_n)
    # Compute array of chis
    status = 0
    chi, status = lib.get_chis_lensing_kernel_wrapper(cosmo.cosmo, z_n[-1],
                                                      nchi, status)
    # Compute kernel
    wchi, status = lib.get_lensing_kernel_wrapper(cosmo.cosmo,
                                                  z_n, n, z_n[-1],
                                                  int(has_magbias), z_s, s,
                                                  chi, nchi, status)
    check(status, cosmo=cosmo)
    return chi, wchi


def get_kappa_kernel(cosmo, z_source, nsamples):
    """This convenience function returns the radial kernel for
    CMB-lensing-like tracers.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        z_source (float): Redshift of source plane for CMB lensing.
        nsamples (int): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    _check_background_spline_compatibility(cosmo, np.array([z_source]))
    # this call inits the distance splines neded by the kernel functions
    chi_source = comoving_radial_distance(cosmo, 1./(1.+z_source))
    chi = np.linspace(0, chi_source, nsamples)

    status = 0
    wchi, status = lib.get_kappa_kernel_wrapper(cosmo.cosmo, chi_source,
                                                chi, nsamples, status)
    check(status, cosmo=cosmo)
    return chi, wchi


class Tracer(object):
    """Tracers contain the information necessary to describe the
    contribution of a given sky observable to its cross-power spectrum
    with any other tracer. Tracers are composed of 4 main ingredients:

    * A radial kernel: this expresses the support in redshift/distance
      over which this tracer extends.

    * A transfer function: this is a function of wavenumber and
      scale factor that describes the connection between the tracer
      and the power spectrum on different scales and at different
      cosmic times.

    * An ell-dependent prefactor: normally associated with angular
      derivatives of a given fundamental quantity.

    * The order of the derivative of the Bessel functions with which
      they enter the computation of the angular power spectrum.

    A `Tracer` object will in reality be a list of different such
    tracers that get combined linearly when computing power spectra.
    Further details can be found in Section 4.9 of the CCL note.
    """
    def __init__(self):
        """By default this `Tracer` object will contain no actual
        tracers
        """
        # Do nothing, just initialize list of tracers
        self._trc = []

    def _dndz(self, z):
        raise NotImplementedError("`get_dndz` not implemented for "
                                  "this `Tracer` type.")

    def get_dndz(self, z):
        """Get the redshift distribution for this tracer.
        Only available for some tracers (:class:`NumberCountsTracer` and
        :class:`WeakLensingTracer`).

        Args:
            z (float or array_like): redshift values.

        Returns:
            array_like: redshift distribution evaluated at the \
                input values of `z`.
        """
        return self._dndz(z)

    def get_kernel(self, chi):
        """Get the radial kernels for all tracers contained
        in this `Tracer`.

        Args:
            chi (float or array_like): values of the comoving
                radial distance in increasing order and in Mpc.

        Returns:
            array_like: list of radial kernels for each tracer. \
                The shape will be `(n_tracer, chi.size)`, where \
                `n_tracer` is the number of tracers. The last \
                dimension will be squeezed if the input is a \
                scalar.
        """
        if not hasattr(self, '_trc'):
            return []

        chi_use = np.atleast_1d(chi)
        kernels = []
        for t in self._trc:
            status = 0
            w, status = lib.cl_tracer_get_kernel(t, chi_use,
                                                 chi_use.size,
                                                 status)
            check(status)
            kernels.append(w)
        kernels = np.array(kernels)
        if np.ndim(chi) == 0:
            if kernels.shape != (0,):
                kernels = np.squeeze(kernels, axis=-1)
        return kernels

    def get_f_ell(self, ell):
        """Get the ell-dependent prefactors for all tracers
        contained in this `Tracer`.

        Args:
            ell (float or array_like): angular multipole values.

        Returns:
            array_like: list of prefactors for each tracer. \
                The shape will be `(n_tracer, ell.size)`, where \
                `n_tracer` is the number of tracers. The last \
                dimension will be squeezed if the input is a \
                scalar.
        """
        if not hasattr(self, '_trc'):
            return []

        ell_use = np.atleast_1d(ell)
        f_ells = []
        for t in self._trc:
            status = 0
            f, status = lib.cl_tracer_get_f_ell(t, ell_use,
                                                ell_use.size,
                                                status)
            check(status)
            f_ells.append(f)
        f_ells = np.array(f_ells)
        if np.ndim(ell) == 0:
            if f_ells.shape != (0,):
                f_ells = np.squeeze(f_ells, axis=-1)
        return f_ells

    def get_transfer(self, lk, a):
        """Get the transfer functions for all tracers contained
        in this `Tracer`.

        Args:
            lk (float or array_like): values of the natural logarithm of
                the wave number (in units of inverse Mpc) in increasing
                order.
            a (float or array_like): values of the scale factor.

        Returns:
            array_like: list of transfer functions for each tracer. \
                The shape will be `(n_tracer, lk.size, a.size)`, where \
                `n_tracer` is the number of tracers. The other \
                dimensions will be squeezed if the inputs are scalars.
        """
        if not hasattr(self, '_trc'):
            return []

        lk_use = np.atleast_1d(lk)
        a_use = np.atleast_1d(a)
        transfers = []
        for t in self._trc:
            status = 0
            t, status = lib.cl_tracer_get_transfer(t, lk_use, a_use,
                                                   lk_use.size * a_use.size,
                                                   status)
            check(status)
            transfers.append(t.reshape([lk_use.size, a_use.size]))
        transfers = np.array(transfers)
        if transfers.shape != (0,):
            if np.ndim(a) == 0:
                transfers = np.squeeze(transfers, axis=-1)
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-1)
            else:
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-2)
        return transfers

    def get_bessel_derivative(self):
        """Get Bessel function derivative orders for all tracers contained
        in this `Tracer`.

        Returns:
            array_like: list of Bessel derivative orders for each tracer.
        """
        if not hasattr(self, '_trc'):
            return []

        return np.array([t.der_bessel for t in self._trc])

    def _MG_add_tracer(self, cosmo, kernel, z_b, der_bessel=0, der_angles=0,
                       bias_transfer_a=None, bias_transfer_k=None):
        """ function to set mg_transfer in the right format and add MG tracers
            for different cases including different cases and biases like
            intrinsic alignements (IA) when present
        """
        # Getting MG transfer function and building a k-array
        mg_transfer = self._get_MG_transfer_function(cosmo, z_b)

        # case with no astro biases
        if ((bias_transfer_a is None) and (bias_transfer_k is None)):
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a and  k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * (bias_transfer_k[1] *
                                mg_transfer[2]).T).T)
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a but not k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * mg_transfer[2].T).T)
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on k but not a
        elif ((bias_transfer_a is None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_k[1] * mg_transfer[2]))
            self.add_tracer(cosmo, kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

    def _get_MG_transfer_function(self, cosmo, z):
        """ This function allows to obtain the function Sigma(z,k) (1 or 2D
            arrays) for an array of redshifts coming from a redshift
            distribution (defined by the user) and a single value or
            an array of k specified by the user. We obtain then Sigma(z,k) as a
            1D array for those z and k arrays and then convert it to a 2D array
            taking into consideration the given sizes of the arrays for z and k
            The MG parameter array goes then as a multiplicative factor within
            the MG transfer function. If k is not specified then only a 1D
            array for Sigma(a,k=0) is used.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): cosmology object used to
                transform redshifts into distances.
            z (float or tuple of arrays): a single z value (e.g. for CMB)
                or a tuple of arrays (z, N(z)) giving the redshift distribution
                of the objects. The units are arbitrary; N(z) will be
                normalized to unity.
            k (float or array): a single k value or an array of k for which we
                calculate the MG parameter Sigma(a,k). For now, the k range
                should be limited to linear scales.
        """
        # Sampling scale factor from a very small (at CMB for example)
        # all the way to 1 here and today for the transfer function.
        # For a < a_single it is GR (no early MG)
        if isinstance(z, float):
            a_single = 1/(1+z)
            a = np.linspace(a_single, 1, 100)
            # a_single is for example like for the CMB surface
        else:
            if z[0] != 0.0:
                stepsize = z[1]-z[0]
                samplesize = int(z[0]/stepsize)
                z_0_to_zmin = np.linspace(0.0, z[0] - stepsize, samplesize)
                z = np.concatenate((z_0_to_zmin, z))
            a = 1./(1.+z)
        a.sort()
        # Scale-dependant MG case with an array of k
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        status = 0
        lk, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
        k = np.exp(lk)
        # computing MG factor array
        mgfac_1d = 1
        mgfac_1d += _Sig_MG(cosmo, a, k)
        # converting 1D MG factor to a 2D array, so it is compatible
        # with the transfer_ka input structure in MG_add.tracer and
        # add.tracer
        mgfac_2d = mgfac_1d.reshape(len(a), -1, order='F')
        # setting transfer_ka for this case
        mg_transfer = (a, lk, mgfac_2d)

        return mg_transfer

    def add_tracer(self, cosmo, kernel=None,
                   transfer_ka=None, transfer_k=None, transfer_a=None,
                   der_bessel=0, der_angles=0,
                   is_logt=False, extrap_order_lok=0, extrap_order_hik=2):
        """Adds one more tracer to the list contained in this `Tracer`.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): cosmology object.
            kernel (tulple of arrays, optional): A tuple of arrays
                (`chi`, `w_chi`) describing the radial kernel of this
                tracer. `chi` should contain values of the comoving
                radial distance in increasing order, and `w_chi` should
                contain the values of the kernel at those values of the
                radial distance. The kernel will be assumed to be zero
                outside the range of distances covered by `chi`. If
                `kernel` is `None` a constant kernel w(chi)=1 will be
                assumed everywhere.
            transfer_ka (tuple of arrays, optional): a tuple of arrays
                (`a`,`lk`,`t_ka`) describing the most general transfer
                function for a tracer. `a` should be an array of scale
                factor values in increasing order. `lk` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                `t_ka` should be an array of shape `(na,nk)`, where
                `na` and `nk` are the sizes of `a` and `lk` respectively.
                `t_ka` should hold the values of the transfer function at
                the corresponding values of `a` and `lk`. If your transfer
                function is factorizable (i.e. T(a,k) = A(a) * K(k)), it is
                more efficient to set this to `None` and use `transfer_k`
                and `transfer_a` to describe K and A respectively. The
                transfer function will be assumed continuous and constant
                outside the range of scale factors covered by `a`. It will
                be extrapolated using polynomials of order `extrap_order_lok`
                and `extrap_order_hik` below and above the range of
                wavenumbers covered by `lk` respectively. If this argument
                is not `None`, the values of `transfer_k` and `transfer_a`
                will be ignored.
            transfer_k (tuple of arrays, optional): a tuple of arrays
                (`lk`,`t_k`) describing the scale-dependent part of a
                factorizable transfer function. `lk` should be an
                array of values of the natural logarithm of the wave
                number (in units of inverse Mpc) in increasing order.
                `t_k ` should be an array of the same size holding the
                values of the k-dependent part of the transfer function
                at those wavenumbers. It will be extrapolated using
                polynomials of order `extrap_order_lok` and `extrap_order_hik`
                below and above the range of wavenumbers covered by `lk`
                respectively. If `None`, the k-dependent part of the transfer
                function will be set to 1 everywhere.
            transfer_a (tuple of arrays, optional): a tuple of arrays
                (`a`,`t_a`) describing the time-dependent part of a
                factorizable transfer function. `a` should be an array of
                scale factor values in increasing order. `t_a` should
                contain the time-dependent part of the transfer function
                at those values of the scale factor. The time dependence
                will be assumed continuous and constant outside the range
                covered by `a`. If `None`, the time-dependent part of the
                transfer function will be set to 1 everywhere.
            der_bessel (int): order of the derivative of the Bessel
                functions with which this tracer enters the calculation
                of the power spectrum. Allowed values are -1, 0, 1 and 2.
                0, 1 and 2 correspond to the raw functions, their first
                derivatives or their second derivatives. -1 corresponds to
                the raw functions divided by the square of their argument.
                We enable this special value because this type of dependence
                is ubiquitous for many common tracers (lensing, IAs), and
                makes the corresponding transfer functions more stables
                for small k or chi.
            der_angles (int): integer describing the ell-dependent prefactor
                associated with this tracer. Allowed values are 0, 1 and 2.
                0 means no prefactor. 1 means a prefactor ell*(ell+1),
                associated with the angular laplacian and used e.g. for
                lensing convergence and magnification. 2 means a prefactor
                sqrt((ell+2)!/(ell-2)!), associated with the angular
                derivatives of spin-2 fields (e.g. cosmic shear, IAs).
            is_logt (bool): if `True`, `transfer_ka`, `transfer_k` and
                `transfer_a` will contain the natural logarithm of the
                transfer function (or their factorizable parts). Default is
                `False`.
            extrap_order_lok (int): polynomial order used to extrapolate the
                transfer functions for low wavenumbers not covered by the
                input arrays.
            extrap_order_hik (int): polynomial order used to extrapolate the
                transfer functions for high wavenumbers not covered by the
                input arrays.
        """
        is_factorizable = transfer_ka is None
        is_k_constant = (transfer_ka is None) and (transfer_k is None)
        is_a_constant = (transfer_ka is None) and (transfer_a is None)
        is_kernel_constant = kernel is None

        chi_s, wchi_s = _check_array_params(kernel, 'kernel')
        if is_factorizable:
            a_s, ta_s = _check_array_params(transfer_a, 'transfer_a')
            lk_s, tk_s = _check_array_params(transfer_k, 'transfer_k')
            tka_s = NoneArr
            if (not is_a_constant) and (a_s.shape != ta_s.shape):
                raise ValueError("Time-dependent transfer arrays "
                                 "should have the same shape")
            if (not is_k_constant) and (lk_s.shape != tk_s.shape):
                raise ValueError("Scale-dependent transfer arrays "
                                 "should have the same shape")
        else:
            a_s, lk_s, tka_s = _check_array_params(transfer_ka, 'transer_ka',
                                                   arr3=True)
            if tka_s.shape != (len(a_s), len(lk_s)):
                raise ValueError("2D transfer array has inconsistent "
                                 "shape. Should be (na,nk)")
            tka_s = tka_s.flatten()
            ta_s = NoneArr
            tk_s = NoneArr

        status = 0
        ret = lib.cl_tracer_t_new_wrapper(cosmo.cosmo,
                                          int(der_bessel),
                                          int(der_angles),
                                          chi_s, wchi_s,
                                          a_s, lk_s,
                                          tka_s, tk_s, ta_s,
                                          int(is_logt),
                                          int(is_factorizable),
                                          int(is_k_constant),
                                          int(is_a_constant),
                                          int(is_kernel_constant),
                                          int(extrap_order_lok),
                                          int(extrap_order_hik),
                                          status)
        self._trc.append(_check_returned_tracer(ret))

    def __del__(self):
        # Sometimes lib is freed before some Tracers, in which case, this
        # doesn't work.
        # So just check that lib.cl_tracer_t_free is still a real function.
        if hasattr(self, '_trc') and lib.cl_tracer_t_free is not None:
            for t in self._trc:
                lib.cl_tracer_t_free(t)


def _check_returned_tracer(return_val):
    """Wrapper to catch exceptions when tracers are spawned from C.
    """
    if (isinstance(return_val, int)):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr


class IvTracer(Tracer):
    """Specific :class:`Tracer` associated with the cosmic infrared
    background intensity at a specific frequency v (Iv).
    The radial kernel for this tracer is

    .. math::
       W(\\chi) = \\frac{\\chi^{2} S_\\nu^{eff}}{K}.

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the CIB emissivity
    density in units of
    :math:`{\\rm Jy}\\,{\\rm Mpc}^{-1}\\,{\\rm srad}^{-1}` (or
    multiples thereof).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        snu_z (array): effective source flux for one frequency in units of
            :math:`{\\rm Jy}\\,{\\rm L_{Sun}}^{-1}\\.
        z_arr (array): redshift values to compute chi_z
        z_min (float): minimum redshift down to which we define the
            kernel.
        z_max (float): maximum redshift up to which we define the
            kernel. zmax = 6 by default (reionization)
    """
    def __init__(self, cosmo, snu_z, z_arr, z_min=0., z_max=6.):
        self.chi_max = comoving_radial_distance(cosmo, 1./(1+z_max))
        self.chi_min = comoving_radial_distance(cosmo, 1./(1+z_min))
        chi_z = comoving_radial_distance(cosmo, 1./(1+z_arr))
        snu_inter = interp1d(chi_z, snu_z, kind='linear', bounds_error=False, fill_value="extrapolate")
        chi_arr = np.linspace(self.chi_min, self.chi_max, len(snu_z))
        snu_arr = snu_inter(chi_arr)
        K = 1.0e-10  # Kennicutt constant in units of M_sun/yr/L_sun
        w_arr = chi_arr**2*snu_arr/K
        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))
