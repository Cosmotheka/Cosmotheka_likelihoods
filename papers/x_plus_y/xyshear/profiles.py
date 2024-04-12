import numpy as np
import pyccl as ccl
from scipy.integrate import quad, simps
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import sici

def get_prefac_rho(kind, XH=0.76):
    """Prefactor that transforms gas mas density into
    other types of density, depending on the hydrogen mass
    fraction XH (BBN value by default).
    """
    if kind == "rho_gas":
        return 1.0
    else:
        # Transforms density in M_sun/Mpc^3 into m_p/cm^3
        MsunMpc2Mprotcm = 4.04768956e-17
        if kind == "n_baryon":
            return MsunMpc2Mprotcm * (3 * XH + 1) / 4
        if kind == "n_H":
            return MsunMpc2Mprotcm * XH
        elif kind == "n_electron":
            return MsunMpc2Mprotcm * (XH + 1) / 2
        elif kind == "n_total":
            return MsunMpc2Mprotcm * (5 * XH + 3) / 4
        else:
            raise NotImplementedError(f"Density type {kind} \
                                      not implemented")


def get_prefac_P(kind, XH=0.76):
    """Prefactor that transforms gas Pressure into
    other types of density, depending on the hydrogen mass
    fraction XH (BBN value by default).
    """
    if kind == "n_total":
        return 1.0
    elif kind == "n_H":
        return 4 * XH / (5 * XH + 3)
    elif kind == "n_baryon":
        return (3 * XH + 1) / (5 * XH + 3)
    elif kind == "n_electron":
        return (2 * (XH + 1)) / (5 * XH + 3)
    else:
        raise NotImplementedError(f"Pressure type {kind} not implemented")


def get_fb(cosmo):
    """Returns baryon fraction."""
    return cosmo["Omega_b"] / (cosmo["Omega_b"] + cosmo["Omega_c"])


class HaloProfileDensityBattaglia(ccl.halos.HaloProfile):
    """Gas density profile from Battaglia 2016. Note that there
    are several typos (both in the arXiv and published versions).
    Correct formulas in Bolliet et al. 2022 (2208.07847).

    Profile is calculated in units of M_sun Mpc^-3 if
    requesting mass density (`kind == 'rho_gas'`), or in cm^-3
    if requesting a number density. Allowed values for `kind`
    in the latter case are `'n_total'`, `'n_baryon'`, `'n_H'`,
    `'n_electron'`.

    Default values of all parameters correspond to the values
    found in Battaglia et al. 2016.
    """

    def __init__(self, *, mass_def,
                 rho0_A=4e3,
                 rho0_aM=0.29,
                 rho0_az=-0.66,
                 alpha_A=0.88,
                 alpha_aM=-0.03,
                 alpha_az=0.19,
                 beta_A=3.83,
                 beta_aM=0.04,
                 beta_az=-0.025,
                 gamma=-0.2,
                 xc=0.5,
                 alpha_interp_spacing=0.1,
                 beta_interp_spacing=0.3,
                 qrange=(1e-3, 1e3),
                 nq=128,
                 x_out=np.inf,
                 kind="rho_gas"):
        self.rho0_A = rho0_A
        self.rho0_aM = rho0_aM
        self.rho0_az = rho0_az
        self.alpha_A = alpha_A
        self.alpha_aM = alpha_aM
        self.alpha_az = alpha_az
        self.beta_A = beta_A
        self.beta_aM = beta_aM
        self.beta_az = beta_az
        self.gamma = gamma
        self.xc = xc
        self.kind = kind
        self.prefac_rho = get_prefac_rho(self.kind)

        self.alpha_interp_spacing = alpha_interp_spacing
        self.beta_interp_spacing = beta_interp_spacing
        self.qrange = qrange
        self.nq = nq
        self.x_out = x_out
        self._fourier_interp = None
        super().__init__(mass_def=mass_def)

    def _AMz(self, M, a, A, aM, az):
        return A * (M * 1e-14) ** aM / a**az

    def _alpha(self, M, a):
        return self._AMz(M, a, self.alpha_A, self.alpha_aM, self.alpha_az)

    def _beta(self, M, a):
        return self._AMz(M, a, self.beta_A, self.beta_aM, self.beta_az)

    def _rho0(self, M, a):
        return self._AMz(M, a, self.rho0_A, self.rho0_aM, self.rho0_az)

    def update_parameters(self,
                          rho0_A=None,
                          rho0_aM=None,
                          rho0_az=None,
                          alpha_A=None,
                          alpha_aM=None,
                          alpha_az=None,
                          beta_A=None,
                          beta_aM=None,
                          beta_az=None,
                          gamma=None,
                          xc=None):
        if rho0_A is not None:
            self.rho0_A = rho0_A
        if rho0_aM is not None:
            self.rho0_aM = rho0_aM
        if rho0_az is not None:
            self.rho0_az = rho0_az
        if beta_A is not None:
            self.beta_A = beta_A
        if beta_aM is not None:
            self.beta_aM = beta_aM
        if beta_az is not None:
            self.beta_az = beta_az
        if alpha_A is not None:
            self.alpha_A = alpha_A
        if alpha_aM is not None:
            self.alpha_aM = alpha_aM
        if alpha_az is not None:
            self.alpha_az = alpha_az

        if xc is not None:
            self.xc = xc
        re_fourier=False
        if gamma is not None:
            if gamma != self.gamma:
                re_fourier = True
            self.gamma = gamma

        if re_fourier and (self._fourier_interp is not None):
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()

    def _form_factor(self, x, alpha, beta):
        # Note: this deviates from the arXiv version of the Battaglia
        # paper in the sign of the second instance of gamma. This was
        # a typo in the paper (Boris Bolliet - private comm.).
        return x**self.gamma / (1 + x**alpha) ** ((beta + self.gamma) / alpha)

    def _integ_interp(self):
        qs = np.geomspace(self.qrange[0], self.qrange[1], self.nq + 1)

        def integrand(x, alpha, beta):
            return self._form_factor(x, alpha, beta) * x

        alpha0 = self._alpha(1e15, 1.0) - self.alpha_interp_spacing
        alpha1 = self._alpha(1e10, 1 / (1 + 6.0)) + self.alpha_interp_spacing
        nalpha = int((alpha1 - alpha0) / self.alpha_interp_spacing)
        alphas = np.linspace(alpha0, alpha1, nalpha)
        beta0 = self._beta(1e10, 1.0) - 1
        beta1 = self._beta(1e15, 1 / (1 + 6.0)) + 1
        nbeta = int((beta1 - beta0) / self.beta_interp_spacing)
        betas = np.linspace(beta0, beta1, nbeta)
        f_arr = np.array(
            [
                [
                    [
                        quad(
                            integrand,
                            args=(
                                alpha,
                                beta,
                            ),
                            a=1e-4,
                            b=self.x_out,  # limits of integration
                            weight="sin",  # fourier sine weight
                            wvar=q,
                        )[0]
                        / q
                        for alpha in alphas
                    ]
                    for beta in betas
                ]
                for q in qs
            ]
        )
        # Set to zero at high q, so extrapolation does the right thing.
        f_arr[-1, :, :] = 1e-100
        Fqb = RegularGridInterpolator(
            [np.log(qs), betas, alphas],
            np.log(f_arr),
            fill_value=None,
            bounds_error=False,
            method="cubic",
        )
        return Fqb

    def _norm(self, cosmo, M, a):
        # Density in Msun/Mpc^3
        # Note: this deviates from the arXiv version of the Battaglia
        # paper in the extra factor of f_b. This was
        # a typo in the paper (Boris Bolliet - private comm.).
        rho_c = ccl.rho_x(cosmo, a, self.mass_def.rho_type)
        fb = get_fb(cosmo) * self.prefac_rho
        rho0 = self._rho0(M, a) * rho_c * fb
        return rho0

    def _fourier(self, cosmo, k, M, a):
        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()

        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # R_Delta*(1+z)
        xrDelta = self.xc * self.mass_def.get_radius(cosmo, M_use, a) / a

        qs = k_use[None, :] * xrDelta[:, None]
        alphas = self._alpha(M_use, a)
        betas = self._beta(M_use, a)
        nk = len(k_use)
        ev = np.array(
            [
                np.log(qs).flatten(),
                (np.ones(nk)[None, :] * betas[:, None]).flatten(),
                (np.ones(nk)[None, :] * alphas[:, None]).flatten(),
            ]
        ).T
        ff = self._fourier_interp(ev).reshape([-1, nk])
        ff = np.exp(ff)
        nn = self._norm(cosmo, M_use, a)

        prof = (4 * np.pi * xrDelta**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        xrDelta = self.xc * self.mass_def.get_radius(cosmo, M_use, a) / a
        alphas = self._alpha(M_use, a)
        betas = self._beta(M_use, a)

        nn = self._norm(cosmo, M_use, a)
        prof = self._form_factor(
            r_use[None, :] / xrDelta[:, None], alphas[:, None], betas[:, None]
        )
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfilePressureBattaglia(ccl.halos.HaloProfile):
    """Gas pressure profile from Battaglia 2012.

    Profile is calculated in units of eV cm^-3. Allowed values
    for `kind` are `'n_total'`, `'n_baryon'`, `'n_H'`, and
    `'n_elecron'`.

    Default values of all parameters correspond to the values
    found in Battaglia et al. 2016.
    """
    def __init__(self, *, mass_def,
                 alpha=1,
                 gamma=-0.3,
                 P0_A=18.1,
                 P0_aM=0.154,
                 P0_az=-0.758,
                 xc_A=0.497,
                 xc_aM=-0.00865,
                 xc_az=0.731,
                 beta_A=4.35,
                 beta_aM=0.0393,
                 beta_az=0.415,
                 beta_interp_spacing=0.3,
                 kind="n_total",
                 qrange=(1e-2, 1e3),
                 nq=128,
                 x_out=np.inf):
        self.alpha = alpha
        self.gamma = gamma
        self.P0_A = P0_A
        self.P0_aM = P0_aM
        self.P0_az = P0_az
        self.xc_A = xc_A
        self.xc_aM = xc_aM
        self.xc_az = xc_az
        self.beta_A = beta_A
        self.beta_aM = beta_aM
        self.beta_az = beta_az
        self.kind = kind
        self.prefac_P = get_prefac_P(self.kind)

        self.beta_interp_spacing = beta_interp_spacing
        self.qrange = qrange
        self.nq = nq
        self.x_out = x_out
        self._fourier_interp = None
        super().__init__(mass_def=mass_def)

    def _AMz(self, M, a, A, aM, az):
        return A * (M * 1e-14) ** aM / a**az

    def _beta(self, M, a):
        return self._AMz(M, a, self.beta_A, self.beta_aM, self.beta_az)

    def _xc(self, M, a):
        return self._AMz(M, a, self.xc_A, self.xc_aM, self.xc_az)

    def _P0(self, M, a):
        return self._AMz(M, a, self.P0_A, self.P0_aM, self.P0_az)

    def update_parameters(self,
                          alpha=None,
                          gamma=None,
                          P0_A=None,
                          P0_aM=None,
                          P0_az=None,
                          xc_A=None,
                          xc_aM=None,
                          xc_az=None,
                          beta_A=None,
                          beta_aM=None,
                          beta_az=None):
        if P0_A is not None:
            self.P0_A = P0_A
        if P0_aM is not None:
            self.P0_aM = P0_aM
        if P0_az is not None:
            self.P0_az = P0_az
        if xc_A is not None:
            self.xc_A = xc_A
        if xc_aM is not None:
            self.xc_aM = xc_aM
        if xc_az is not None:
            self.xc_az = xc_az
        if beta_A is not None:
            self.beta_A = beta_A
        if beta_aM is not None:
            self.beta_aM = beta_aM
        if beta_az is not None:
            self.beta_az = beta_az

        re_fourier = False
        if alpha is not None:
            if alpha != self.alpha:
                re_fourier = True
            self.alpha = alpha
        if gamma is not None:
            if gamma != self.gamma:
                re_fourier = True
            self.gamma = gamma

        if re_fourier and (self._fourier_interp is not None):
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()

    def _form_factor(self, x, beta):
        return x**self.gamma / (1 + x**self.alpha) ** beta

    def _integ_interp(self):
        qs = np.geomspace(self.qrange[0], self.qrange[1], self.nq + 1)

        def integrand(x, beta):
            return self._form_factor(x, beta) * x

        beta0 = self._beta(1e10, 1.0) - 1
        beta1 = self._beta(1e15, 1 / (1 + 6.0)) + 1
        nbeta = int((beta1 - beta0) / self.beta_interp_spacing)
        betas = np.linspace(beta0, beta1, nbeta)
        f_arr = np.array(
            [
                [
                    quad(
                        integrand,
                        args=(beta,),
                        a=1e-4,
                        b=self.x_out,  # limits of integration
                        weight="sin",  # fourier sine weight
                        wvar=q,
                    )[0]
                    / q
                    for beta in betas
                ]
                for q in qs
            ]
        )
        # Set to zero at high q, so extrapolation does the right thing.
        Fqb = RegularGridInterpolator(
            [np.log(qs), betas],
            np.log(f_arr),
            fill_value=None,
            bounds_error=False,
            method="linear",
        )
        return Fqb

    def _norm(self, cosmo, M, a):
        fb = get_fb(cosmo) * self.prefac_P
        RM = self.mass_def.get_radius(cosmo, M, a)
        Delta = self.mass_def.get_Delta(cosmo, a)
        # G in units of eV*(Mpc^4)/(cm^3*Msun^2)
        G = 1.81805235e-27
        # Density in Msun/Mpc^3
        rho = ccl.rho_x(cosmo, a, self.mass_def.rho_type)
        P0 = self._P0(M, a)
        return P0 * fb * G * M * Delta * rho / (2 * RM)

    def _fourier(self, cosmo, k, M, a):
        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()

        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        xc = self._xc(M_use, a)
        # R_Delta*(1+z)
        xrDelta = xc * self.mass_def.get_radius(cosmo, M_use, a) / a

        qs = k_use[None, :] * xrDelta[:, None]
        betas = self._beta(M_use, a)
        nk = len(k_use)
        ev = np.array(
            [np.log(qs).flatten(), (np.ones(nk)[None, :] *
                                    betas[:, None]).flatten()]
        ).T
        ff = self._fourier_interp(ev).reshape([-1, nk])
        ff = np.exp(ff)
        nn = self._norm(cosmo, M_use, a)

        prof = (4 * np.pi * xrDelta**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        xc = self._xc(M_use, a)
        xrDelta = xc * self.mass_def.get_radius(cosmo, M_use, a) / a
        betas = self._beta(M_use, a)

        nn = self._norm(cosmo, M_use, a)
        prof = self._form_factor(r_use[None, :] / xrDelta[:, None],
                                 betas[:, None])
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def profile_cumul_nr(self, nr, cosmo, M, a):
        M_use = np.atleast_1d(M)
        rDelta = self.mass_def.get_radius(cosmo, M, a) / a
        rmax = nr * rDelta
        rs = np.geomspace(1e-3, rmax, 256).T
        pr = np.array([self._real(cosmo, r, m, a) for r, m
                       in zip(rs, M_use)])
        cprof = simps(pr * rs**3, x=np.log(rs), axis=-1) * 4 * np.pi
        if np.ndim(M) == 0:
            cprof = np.squeeze(cprof, axis=0)
        return cprof


class _HaloProfileHE(ccl.halos.HaloProfile):
    """Gas density profile given by the sum of the density profile for
    the bound and the ejected gas, each modelled separetely for a halo
    in hydrostatic equilibrium.

    The density and mass fraction of the bound gas as well as the mass
    fraction of the ejected gas taken from Mead 2020, and the density
    of the ejected gas taken from Schneider & Teyssier 2016.

    Profile is calculated in units of M_sun Mpc^-3 if
    requesting mass density (`kind == 'rho_gas'`), or in cm^-3
    if requesting a number density. Allowed values for `kind`
    in the latter case are `'n_total'`, `'n_baryon'`, `'n_H'`,
    `'n_electron'`.

    Default values of all parameters correspond to the values
    found in Mead et al. 2020.
    """
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0, beta=0.6, gamma=1.17,
                 gamma_T=1.0,
                 A_star=0.03, sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0, alpha_Tz=0., alpha_Tm=0.,
                 logTw0=6.5, Tw1=0.0,
                 epsilon=1.0,
                 logTAGN=None,
                 kind="rho_gas",
                 quantity="density"):
        self._Bi = None
        if logTAGN is not None:
            lMc, gamma, alpha_T, logTw0, Tw1 = self.from_logTAGN(logTAGN)
        self.logTAGN = logTAGN
        self.lMc = lMc
        self.beta = beta
        self.gamma = gamma
        self.gamma_T = gamma_T
        self.A_star = A_star
        self.eta_b = eta_b
        self.alpha_T = alpha_T
        self.alpha_Tz = alpha_Tz
        self.alpha_Tm = alpha_Tm
        self.logTw0 = logTw0
        self.Tw1 = Tw1
        self.epsilon = epsilon
        self.sigma_star = sigma_star
        self.kind = kind
        self.quantity = quantity
        self.prefac_rho = get_prefac_rho(self.kind)
        self.norm_interp = self.get_dens_norm_interp()

        super().__init__(mass_def=mass_def, concentration=concentration)
    
    def _F_bound(self, x, G):
        return (np.log(1 + x) / x)**G

    def get_dens_norm_interp(self):
        cs = np.geomspace(1E-2, 100, 64)
        gs = np.geomspace(0.1, 10, 64)
        norms = np.array([[quad(lambda x: x**2*self._F_bound(x, g), 0, c)[0]
                           for c in cs]
                          for g in gs])
        ip = RegularGridInterpolator((np.log(gs), np.log(cs)), np.log(norms))
        return ip

    def update_parameters(self,
                          lMc=None,
                          beta=None,
                          gamma=None,
                          gamma_T=None,
                          A_star=None,
                          sigma_star=None,
                          alpha_T=None,
                          alpha_Tz=None,
                          alpha_Tm=None,
                          logTw0=None,
                          Tw1=None,
                          epsilon=None,
                          eta_b=None,
                          logTw=None,
                          logTAGN=None):
        if logTAGN is not None:
            lMc, gamma, alpha_T, logTw0, Tw1 = self.from_logTAGN(logTAGN)
            self.logTAGN = logTAGN
        if lMc is not None:
            self.lMc = lMc
        if logTw0 is not None:
            self.logTw0 = logTw0
        if Tw1 is not None:
            self.Tw1 = Tw1
        if epsilon is not None:
            self.epsilon = epsilon
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if gamma_T is not None:
            self.gamma_T = gamma_T
        if A_star is not None:
            self.A_star = A_star
        if eta_b is not None:
            self.eta_b = eta_b
        if sigma_star is not None:
            self.sigma_star = sigma_star
        if alpha_T is not None:
            self.alpha_T = alpha_T
        if alpha_Tz is not None:
            self.alpha_Tz = alpha_Tz
        if alpha_Tm is not None:
            self.alpha_Tm = alpha_Tm

    def _build_BAHAMAS_interp(self):
        if self._Bi is not None:
            return
        kwargs = {'kind': 'linear',
                  'bounds_error': False,
                  'fill_value': 'extrapolate'}
        logTAGNs = np.array([7.6, 7.8, 8.0])
        self._Bi = {}
        lMci = interp1d(logTAGNs, np.array([13.1949, 13.5937, 14.2480]),
                        **kwargs)
        gammai = interp1d(logTAGNs, np.array([1.1647, 1.1770, 1.1966]),
                          **kwargs)
        alpha_Ti = interp1d(logTAGNs, np.array([0.7642, 0.8471, 1.0314]),
                            **kwargs)
        logTw0i = interp1d(logTAGNs, np.array([6.6762, 6.6545, 6.6615]),
                           **kwargs)
        Tw1i = interp1d(logTAGNs, np.array([-0.5566, -0.3652, -0.0617]),
                        **kwargs)
        self._Bi['lMc'] = lMci
        self._Bi['gamma'] = gammai
        self._Bi['alpha_T'] = alpha_Ti
        self._Bi['logTw0'] = logTw0i
        self._Bi['Tw1'] = Tw1i

    def from_logTAGN(self, logTAGN):
        self._build_BAHAMAS_interp()
        lMc = self._Bi['lMc'](logTAGN)
        gamma = self._Bi['gamma'](logTAGN)
        alpha_T = self._Bi['alpha_T'](logTAGN)
        logTw0 = self._Bi['logTw0'](logTAGN)
        Tw1 = self._Bi['Tw1'](logTAGN)
        return lMc, gamma, alpha_T, logTw0, Tw1

    def _get_fractions(self, cosmo, M):
        fb = get_fb(cosmo)
        Mbeta = (cosmo['h']*M*10**(-self.lMc))**self.beta
        f_star = self.A_star * \
            np.exp(-0.5*((np.log10(cosmo["h"]*M)-12.5)/self.sigma_star)**2)
        f_bound = (fb - f_star)*Mbeta/(1+Mbeta)
        f_ejected = fb-f_bound-f_star
        return f_bound, f_ejected, f_star

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        am3 = 1/a**3

        # Comoving virial radius
        Delta = self.mass_def.get_Delta(cosmo, a)
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        rs = rDelta/cM
        rs *= self.epsilon
        x = r_use[None, :] / rs[:, None]

        # Mass fractions
        fb, fe, _ = self._get_fractions(cosmo, M)

        # Bound gas
        G = 1./(self.gamma-1)
        xnorm = np.array([np.full_like(cM, G), cM]).T
        norm = np.exp(self.norm_interp(np.log(xnorm)))
        shape = self._F_bound(x, G)
        rho_bound = (am3*M_use*fb/(4*np.pi*rs**3*norm))[:, None]*shape

        # Ejected gas
        # Eq. (2.13) of Schneider & Teyssier 2016
        x_esc = (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
        rho_ejected = (am3*M_use*fe/rs**3)[:, None] * \
            np.exp(-0.5*(x/x_esc)**2)/(2*np.pi*x_esc**2)**1.5

        if self.quantity == 'density':
            prof = (rho_bound + rho_ejected) * self.prefac_rho
        elif self.quantity == 'pressure':
            # Boltmann constant which, when multiplied by T in Kelvin
            # gives you eV
            k_boltz = 8.61732814974493e-05
            T_ejected = k_boltz * 10**self.logTw0 * np.exp(self.Tw1*(1/a-1))

            # Gravitational constant in eV*(Mpc^4)/(cm^3*Msun^2)
            mu_p = 0.61  # See footnote 8 in arXiv:2005.00009
            G_mp = 4.49158049E-11
            # The quantity above is: G*(1 Msun)*(proton mass)/(1 Mpc)/(1 eV)
            # I.e. gravitational potential of a proton 1Mpc away from the sun
            # in eV.
            T_bound_shape = (np.log(1+x)/x) ** self.gamma_T

            alpha_T = self.alpha_T * (a**self.alpha_Tz) * (M_use * 1e-14)**self.alpha_Tm
            T_bound_num = 2 * alpha_T * G_mp * mu_p * M_use
            T_bound_den = 3 * a * rDelta
            T_bound = (T_bound_num / T_bound_den)[:, None] * T_bound_shape

            # Put them together
            prof = (rho_bound*T_bound+rho_ejected*T_ejected)*self.prefac_rho

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileDensityHE(_HaloProfileHE):
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0,
                 beta=0.6,
                 gamma=1.17,
                 gamma_T=1.0,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 epsilon=1.0,
                 alpha_T=1.0,
                 alpha_Tz=0.,
                 alpha_Tm=0.,
                 logTw0=6.5,
                 Tw1=0.,
                 kind="rho_gas"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         lMc=lMc,
                         beta=beta,
                         gamma=gamma,
                         gamma_T=gamma_T,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         epsilon=epsilon,
                         alpha_T=alpha_T,
                         alpha_Tz=alpha_Tz,
                         alpha_Tm=alpha_Tm,
                         logTw0=logTw0,
                         Tw1=Tw1,                        
                         kind=kind,
                         quantity="density")


class HaloProfilePressureHE(_HaloProfileHE):
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0,
                 beta=0.6,
                 gamma=1.17,
                 gamma_T=1.0,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 epsilon=1.0,
                 alpha_T=1.0,
                 alpha_Tz=0.,
                 alpha_Tm=0.,
                 logTw0=6.5,
                 Tw1=0.,
                 kind="rho_gas"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         lMc=lMc,
                         beta=beta,
                         gamma=gamma,
                         gamma_T=gamma_T,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         epsilon=epsilon,
                         alpha_T=alpha_T,
                         alpha_Tz=alpha_Tz,
                         alpha_Tm=alpha_Tm,
                         logTw0=logTw0,
                         Tw1=Tw1,
                         kind=kind,
                         quantity="pressure")
        
        self._fourier_interp = None


    def _Pbound_shape_interp(self):
        """
        """
        qs = np.geomspace(1e-4, 1e2, 256)
        
        #gammas = np.linspace(1.1, 3., 512)
        #Gs = 1 / (gammas - 1)
        pows = np.linspace(-4.5, 15., 256)
        #G = 1 / (self.gamma - 1)
        def integrand(x, pow):
            return x*self._F_bound(x, pow)

        f_arr = np.array(
            [
                [
                    quad(
                        integrand,
                        args=(pow,),
                        a=1e-4,
                        b=np.inf,
                        weight="sin",
                        wvar=q)[0] / q
                    
                    for pow in pows
                ]
                for q in qs
            ])

        Fqb = RegularGridInterpolator(
            (np.log(qs), pows),
            np.log(f_arr),
            fill_value=None,
            bounds_error=False,
            method="linear",
        )

        return Fqb
    

    def _fourier(self, cosmo, k, M, a):
        """
        """

        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._Pbound_shape_interp()
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        Delta = self.mass_def.get_Delta(cosmo, a)
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        rs = rDelta/cM
        rs *= self.epsilon
        am3 = 1/a**3
        qs = k_use[None, :] * rs[:, None]

        # Mass fractions
        fb, fe, _ = self._get_fractions(cosmo, M)

        # Temp. ejected gas
        k_boltz = 8.61732814974493e-05
        T_ejected = k_boltz * 10**self.logTw0 * np.exp(self.Tw1*(1/a-1))

        # Density ejected gas (fourier)
        x_esc = (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
        density_ejected_fourier = np.exp(-0.5*(qs*x_esc)**2)

        # Pressure bound gas
        Pbound_integral_interp = self._fourier_interp

        pow_pressure = (1 + self.gamma_T * (self.gamma - 1)) / (self.gamma - 1)
        pows = np.full_like(qs, pow_pressure)
        #gammas = np.full_like(qs, self.gamma)
        ev = np.array(
            [np.log(qs).flatten(), pows.flatten()]
        ).T
        Pbound_shape_fourier = np.exp(Pbound_integral_interp(ev)).reshape([-1, len(k)])

        # Normalize Pbound
        # Gravitational constant in eV*(Mpc^4)/(cm^3*Msun^2)
        mu_p = 0.61  # See footnote 8 in arXiv:2005.00009
        G_mp = 4.49158049E-11

        alpha_T = self.alpha_T * (a ** self.alpha_Tz) * (M_use * 1e-14) ** self.alpha_Tm

        T_bound_num = 2 * alpha_T * G_mp * mu_p * M_use
        T_bound_den = 3 * a * rDelta
        T_bound_norm = (T_bound_num / T_bound_den)[:, None]

        G = 1./(self.gamma-1)
        xnorm = np.array([np.full_like(cM, G), cM]).T
        norm = np.exp(self.norm_interp(np.log(xnorm)))
        rho_bound_norm = (am3*M_use*fb/norm)[:,None]

        density_ejected_norm = (am3*M_use*fe)[:,None]

        prof = (Pbound_shape_fourier * T_bound_norm * rho_bound_norm + density_ejected_fourier * T_ejected * density_ejected_norm) * self.prefac_rho
        if np.ndim(rs) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


    def _fourier_num_slow(self, cosmo, k, M, a):
        """
        """
        M = np.atleast_1d(M)
        k = np.atleast_1d(k)

        r0 = self.mass_def.get_radius(cosmo, M, a) / a
        print(f"Shape of r0 is {r0.shape}")
        print(f"Shape of k is {k.shape}")
        print(f"Shape of M is {M.shape}")
        func = lambda x, rval, mval: self._real(cosmo, rval * x, mval, a) * x
        qs = k[None, :] * r0[:, None]
        rvals = np.ones(len(k))[None, :] * r0[:, None]
        mvals = np.ones(len(k))[None, :] * M[:, None]

        qs = qs.flatten()
        rvals = rvals.flatten()
        mvals = mvals.flatten()

        print(qs.shape)
        fq = np.array(
            [quad(func, args=(rvals[i], mvals[i]), a=1e-5, b=np.inf, weight="sin", wvar=q)[0] /
            q for i,q in enumerate(qs)]
        )
        
        fq *= 4 * np.pi * rvals ** 3
        fq = fq.reshape([-1, len(k)])

        return fq 


class _HaloProfileArico2020(ccl.halos.HaloProfile):
    """Gas density profile given by the sum of the density profile for
    the bound and the ejected gas, each modelled separetely for a halo
    in hydrostatic equilibrium.

    The density and mass fraction of the bound gas as well as the mass
    fraction of the ejected gas taken from Mead 2020, and the density
    of the ejected gas taken from Schneider & Teyssier 2016.

    Profile is calculated in units of M_sun Mpc^-3 if
    requesting mass density (`kind == 'rho_gas'`), or in cm^-3
    if requesting a number density. Allowed values for `kind`
    in the latter case are `'n_total'`, `'n_baryon'`, `'n_H'`,
    `'n_electron'`.

    Default values of all parameters correspond to the values
    found in Mead et al. 2020.
    """
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0, beta=0.6, gamma=1.17,
                 A_star=0.03, sigma_star=1.2,
                 theta_in=0.05, theta_out=1.,
                 beta_in=3., beta_out=4.,
                 eta_b=0.5, alpha_T=1.0,
                 alpha_Tz=0., alpha_Tm=0.,
                 logTw0=6.5, Tw1=0.0,
                 kind="rho_gas",
                 quantity="density",
                 sampling_interp=64,
                 use_fftlog=False):
        self.lMc = lMc
        self.beta = beta
        self.gamma = gamma
        self.A_star = A_star
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.beta_in = beta_in
        self.beta_out = beta_out
        self.eta_b = eta_b
        self.alpha_T = alpha_T
        self.alpha_Tz = alpha_Tz
        self.alpha_Tm = alpha_Tm
        self.logTw0 = logTw0
        self.Tw1 = Tw1
        self.sigma_star = sigma_star
        self.kind = kind
        self.quantity = quantity
        self.sampling_interp = sampling_interp
        self.prefac_rho = get_prefac_rho(self.kind)
        self.norm_interp = self.get_dens_norm_interp()
        self._fourier_interp = None
        self.use_fft_log = use_fftlog

        super().__init__(mass_def=mass_def, concentration=concentration)

        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2)
        
        if not self.use_fft_log:
            self._fourier = self._fourier_approx

        self._init_constants()
    
    def _init_constants(self):
        """
        """
        class Const:
            pass
        self.const = Const()

        self.const.mu_p = 0.61 # See footnote 8 in arXiv:2005.00009
        self.const.G_mp = 4.49158049E-11 # Gravitational constant in eV*(Mpc^4)/(cm^3*Msun^2)
        self.const.k_boltz = 8.61732814974493e-05

    def update_parameters(self, **kwargs):
        """
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Parameter {k} not found in HaloProfileArico2020")

    def _theta_in(self, M_in, mu_in, theta_in):
        """
        """
        if M_in is not None and mu_in is not None:
            return 3. - (M_in / 1e14) ** mu_in
        elif theta_in is not None:
            return theta_in

    def _profile_shape(self, x, p_in, p_out, itheta_in, itheta_out):
        """
        """
        return 1 / ((1 + itheta_in * x)**p_in * (1 + (itheta_out * x)**2)**p_out)

    def get_dens_norm_interp(self):
        ith_ins = np.logspace(0.04, 2, self.sampling_interp)
        ith_outs = np.logspace(0, 0.5, self.sampling_interp)
        p_ins = np.linspace(-0.5, 3, self.sampling_interp)
        p_outs = np.linspace(-0.5, 3, self.sampling_interp)
        #norms = np.array([
        #    [
        #        [
        #            [
        #                quad(
        #                    lambda r: r**2 * self._profile_shape(
        #                        r, p_in, p_out, ith_in, ith_out
        #                    ), 0, 1.
        #                )[0] for ith_in in ith_ins
        #            ] for ith_out in ith_outs
        #        ] for p_in in p_ins
        #    ] for p_out in p_outs
        #])
        norms = np.array([
            [
                [quad(
                    lambda r: r**2 * self._profile_shape(
                        r, p_in, p_out, ith_in, 1.
                    ), 0., np.inf)[0] for ith_in in ith_ins
                ] for p_in in p_ins
            ] for p_out in p_outs
        ])

        #ip = RegularGridInterpolator(
        #    (p_outs, p_ins, np.log(ith_outs), np.log(ith_ins)),
        #    np.log(norms)
        #)
        ip = RegularGridInterpolator(
            (p_outs, p_ins, np.log(ith_ins)),
            np.log(norms)
        )

        return ip

    def _get_fractions(self, cosmo, M):
        fb = get_fb(cosmo)
        Mbeta = (cosmo['h']*M*10**(-self.lMc))**self.beta
        f_bound = fb*Mbeta/(1+Mbeta)
        f_star = self.A_star * \
            np.exp(-0.5*((np.log10(cosmo["h"]*M)-12.5)/self.sigma_star)**2)
        f_ejected = fb-f_bound-f_star
        return f_bound, f_ejected, f_star

    def _bound_gas_temp(self, cosmo, var, M, a, space="real"):
        """
        """
        var_use = np.atleast_1d(var)
        M_use = np.atleast_1d(M)

        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a

        ith_in = 1 / self.theta_in
        ith_out = 1 / self.theta_out

        alpha_T = self.alpha_T * (a**self.alpha_Tz) * (M_use * 1e-14)**self.alpha_Tm
        norm = (2 * alpha_T * self.const.G_mp * self.const.mu_p * M_use / (3 * a * rDelta))[:, None]

        if space == "real":
            x = var_use[None, :] / rDelta[:, None]
            shape = self._profile_shape(x, self.beta_in * (self.gamma - 1),
                                        self.beta_out * (self.gamma - 1),
                                        ith_in, ith_out)

        elif space == "fourier":
            x = var_use[None, :] * rDelta[:, None]
            pars_vec = np.transpose([
                np.log(x).flatten(),
                np.full_like(x.flatten(), self.beta_out * (self.gamma - 1)),
                np.full_like(x.flatten(), self.beta_in * (self.gamma - 1)),
                #np.full_like(x.flatten(), np.log(ith_out)),
                np.full_like(x.flatten(), np.log(ith_in))
            ])
            shape = np.exp(self._fourier_interp(pars_vec)).reshape([-1, len(var)])

        return norm * shape

    def _bound_gas_density(self, cosmo, var, M, a, space="real"):
        """
        """
        var_use = np.atleast_1d(var)
        M_use = np.atleast_1d(M)
        am3 = 1/a**3

        ith_in = 1 / self.theta_in
        ith_out = 1 / self.theta_out

        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a

        # Mass fractions
        fb, _, _ = self._get_fractions(cosmo, M)

        pars_vec = np.transpose([
            self.beta_out,
            self.beta_in,
            #np.log(ith_out),
            np.log(ith_in)
        ])
        norm_interp = np.exp(self.norm_interp(pars_vec))

        if space == "real":
            x = var_use[None, :] / rDelta[:, None]
            shape = self._profile_shape(x, self.beta_in, self.beta_out, ith_in, ith_out)
            norm = (am3*M_use*fb/(4*np.pi*rDelta**3*norm_interp))[:, None]

        elif space == "fourier":
            x = var_use[None, :] * rDelta[:, None]
            pars_vec = np.transpose([
                np.log(x).flatten(),
                np.full_like(x.flatten(), self.beta_out),
                np.full_like(x.flatten(), self.beta_in),
                #np.full_like(x.flatten(), np.log(ith_out)),
                np.full_like(x.flatten(), np.log(ith_in))
            ])
            shape = np.exp(self._fourier_interp(pars_vec)).reshape([-1, len(var)])
            norm = (am3*M_use*fb/norm_interp)[:, None]

        return norm * shape * self.prefac_rho

    def _bound_gas_pressure(self, cosmo, var, M, a, space="real"):
        """
        """
        if space == "real":
            dens = self._bound_gas_density(cosmo, var, M, a, space=space)
            temp = self._bound_gas_temp(cosmo, var, M, a, space=space)
            return dens * temp

        if space == "fourier":
            var_use = np.atleast_1d(var)
            M_use = np.atleast_1d(M)
            am3 = 1/a**3
            rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a

            # Mass fractions
            fb, _, _ = self._get_fractions(cosmo, M)

            ith_in = 1 / self.theta_in
            ith_out = 1 / self.theta_out

            pars_vec = np.transpose([
                self.beta_out,
                self.beta_in,
                #np.log(ith_out),
                np.log(ith_in)
            ])
            norm_interp = np.exp(self.norm_interp(pars_vec))

            alpha_T = self.alpha_T * (a**self.alpha_Tz) * (M_use * 1e-14)**self.alpha_Tm

            dens_norm = (am3*M_use*fb/norm_interp)[:, None] * self.prefac_rho
            temp_norm = ((2 * alpha_T * self.const.G_mp * self.const.mu_p * M_use) / (3 * a * rDelta))[:, None]

            x = var_use[None, :] * rDelta[:, None]
            pars_vec = np.transpose([
                np.log(x).flatten(),
                np.full_like(x.flatten(), self.beta_out * self.gamma),
                np.full_like(x.flatten(), self.beta_in * self.gamma),
                #np.full_like(x.flatten(), np.log(ith_out)),
                np.full_like(x.flatten(), np.log(ith_in))
            ])
            shape = np.exp(self._fourier_interp(pars_vec)).reshape([-1, len(var)])

            return dens_norm * temp_norm * shape

    def _ejected_gas_density(self, cosmo, var, M, a, space="real"):
        """ 
        """
        var_use = np.atleast_1d(var)
        M_use = np.atleast_1d(M)
        am3 = 1/a**3

        # Mass fractions
        _, fe, _ = self._get_fractions(cosmo, M)

        Delta = self.mass_def.get_Delta(cosmo, a)
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a

        # Concentation vector
        cM = self.concentration(cosmo, M_use, a)

        rs = rDelta/cM

        # Ejected gas
        # Eq. (2.13) of Schneider & Teyssier 2016
        if space == "real":
            x = var_use[None, :] / rs[:, None]
            sigma = (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
            norm = (am3*M_use*fe/rs**3)[:, None] / (2*np.pi*sigma**2)**1.5

        elif space == "fourier":
            x = var_use[None, :] * rs[:, None]
            sigma = 1 / (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
            norm = (am3*M_use*fe)[:, None]

        rho_ejected = norm * np.exp(-0.5 * (x / sigma) ** 2)
        
        return rho_ejected * self.prefac_rho

    def _ejected_gas_temp(self, a):
        """
        """
        # Boltmann constant which, when multiplied by T in Kelvin
        # gives you eV
        T_ejected = self.const.k_boltz * 10**self.logTw0 * np.exp(self.Tw1*(1/a-1))

        return T_ejected
    
    def _ejected_gas_pressure(self, cosmo, var, M, a, space="real"):
        """
        """
        dens = self._ejected_gas_density(cosmo, var, M, a, space=space)
        temp = self._ejected_gas_temp(a)

        return dens * temp

    def profile(self, cosmo, r, M, a, kind="all", space="real", quantity="density"):
        """
        """
        func = {
            "density": {
                "ejected": self._ejected_gas_density,
                "bound": self._bound_gas_density
            },
            "pressure": {
                "ejected": self._ejected_gas_pressure,
                "bound": self._bound_gas_pressure
            },
            "temperature": {
                "ejected": self._ejected_gas_temp,
                "bound": self._bound_gas_temp
            }
        }

        f_profile = func[quantity]
        P = []
        if kind in ["all", "ejected"]:
            prof_ejected = f_profile["ejected"](cosmo, r, M, a, space=space)
            P.append(prof_ejected)

        if kind in ["all", "bound"]:
            prof_bound = f_profile["bound"](cosmo, r, M, a, space=space)
            P.append(prof_bound)
        
        return np.sum(P, axis=0)

    def _real(self, cosmo, r, M, a):
        """
        """
        prof = self.profile(cosmo, r, M, a, kind="all", space="real", quantity=self.quantity)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

    def _fourier_shape_interp(self):
        """
        """
        qs = np.geomspace(1e-4, 1e2, 128)
        ith_ins = np.logspace(0.04, 2, self.sampling_interp)
        ith_outs = np.logspace(0, 0.5, self.sampling_interp)
        p_ins = np.linspace(-0.5, 3, self.sampling_interp)
        p_outs = np.linspace(-0.5, 3, self.sampling_interp)

        #def integrand(x, p_in, p_out, ith_in, ith_out):
        #    return x * self._profile_shape(x, p_in, p_out, ith_in, ith_out)
        def integrand(x, p_in, p_out, ith_in):
            return x * self._profile_shape(x, p_in, p_out, ith_in, 1.)
        
        #fourier_integ = np.array(
        #    [
        #        [
        #            [
        #                [
        #                    [quad(
        #                        integrand,
        #                        args=(p_in, p_out, ith_in, ith_out),
        #                        a=1e-4,
        #                        b=np.inf,
        #                        weight="sin",
        #                        wvar=q
        #                    )[0] / q for ith_in in ith_ins
        #                    ] for ith_out in ith_outs
        #                ] for p_in in p_ins
        #            ] for p_out in p_outs
        #        ] for q in qs
        #    ]
        #)

        fourier_integ = np.array(
            [
                [
                    [
                        [quad(
                            integrand,
                            args=(p_in, p_out, ith_in),
                            a=1e-4,
                            b=1e2,
                            weight="sin",
                            wvar=q
                        )[0] / q for ith_in in ith_ins
                        ] for p_in in p_ins
                    ] for p_out in p_outs
                ] for q in qs
            ]
        )

        #fourier_interpolator = RegularGridInterpolator(
        #    (np.log(qs), p_outs, p_ins, np.log(ith_outs), np.log(ith_ins)),
        #    np.log(fourier_integ),
        #    fill_value=None,
        #    bounds_error=False,
        #    method="linear"
        #)
        fourier_interpolator = RegularGridInterpolator(
            (np.log(qs), p_outs, p_ins, np.log(ith_ins)),
            np.log(fourier_integ),
            fill_value=None,
            bounds_error=False,
            method="linear"
        )

        return fourier_interpolator

    def _fourier_approx(self, cosmo, k, M, a):
        """
        """
        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._fourier_shape_interp()

        prof = self.profile(cosmo, k, M, a, kind="all", space="fourier", quantity=self.quantity)
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class HaloProfileArico2020_Pressure(_HaloProfileArico2020):
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0,
                 beta=0.6,
                 gamma=1.17,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0,
                 alpha_Tz=0.,
                 alpha_Tm=0.,
                 theta_in=0.05, theta_out=1.,
                 beta_in=3., beta_out=4.,
                 logTw0=6.5,
                 Tw1=0.,
                 kind="rho_gas",
                 sampling_interp=64,
                 use_fftlog=False):

        super().__init__(mass_def=mass_def, 
                         concentration=concentration,
                         lMc=lMc,
                         beta=beta,
                         gamma=gamma,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         alpha_T=alpha_T,
                         alpha_Tz=alpha_Tz,
                         alpha_Tm=alpha_Tm,
                         theta_in=theta_in,
                         theta_out=theta_out,
                         beta_in=beta_in,
                         beta_out=beta_out,
                         logTw0=logTw0,
                         Tw1=Tw1,
                         kind=kind,
                         quantity="pressure",
                         sampling_interp=sampling_interp,
                         use_fftlog=use_fftlog)
        
class HaloProfileArico2020_Density(_HaloProfileArico2020):
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0,
                 beta=0.6,
                 gamma=1.17,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0,
                 alpha_Tz=0.,
                 alpha_Tm=0.,
                 theta_in=0.05, theta_out=1.,
                 beta_in=3., beta_out=4.,
                 logTw0=6.5,
                 Tw1=0.,
                 kind="rho_gas",
                 sampling_interp=64,
                 use_fftlog=False):

        super().__init__(mass_def=mass_def, 
                         concentration=concentration,
                         lMc=lMc,
                         beta=beta,
                         gamma=gamma,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         alpha_T=alpha_T,
                         alpha_Tz=alpha_Tz,
                         alpha_Tm=alpha_Tm,
                         theta_in=theta_in,
                         theta_out=theta_out,
                         beta_in=beta_in,
                         beta_out=beta_out,
                         logTw0=logTw0,
                         Tw1=Tw1,
                         kind=kind,
                         quantity="density",
                         sampling_interp=sampling_interp,
                         use_fftlog=use_fftlog)

class _HaloProfileNFW(ccl.halos.HaloProfile):
    """Simple gas density profile assuming NFW (times cosmic
    baryon fraction).
    """
    def __init__(self, *, mass_def, concentration,
                 truncated=True,
                 par_A=4.295,
                 par_B=0.514,
                 par_C=-0.039,
                 m_fid=3e14,
                 kind="rho_gas",
                 quantity="density"):
        self.nfw = ccl.halos.HaloProfileNFW(
            mass_def=mass_def,
            concentration=concentration,
            fourier_analytic=True,
            projected_analytic=False,
            cumul2d_analytic=False,
            truncated=truncated)
        self.kind = kind
        self.prefac_rho = get_prefac_rho(self.kind)
        self.par_A = par_A
        self.par_B = par_B
        self.par_C = par_C
        self.m_fid = m_fid
        self.quantity = quantity
        super().__init__(mass_def=mass_def, concentration=concentration)

    def _norm(self, cosmo):
        return get_fb(cosmo) * self.prefac_rho

    def _get_T(self, cosmo, M, a):
        m_ratio = M / self.m_fid
        exponent = self.par_B + self.par_C * np.log10(m_ratio)
        cosmo_model = ccl.h_over_h0(cosmo, a) ** (2 / 3)

        return 1E3*cosmo_model * self.par_A * m_ratio ** exponent

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        norm = self._norm(cosmo)
        prof = norm * self.nfw._real(cosmo, r_use, M_use, a)

        if self.quantity == 'pressure':
            T = self._get_T(cosmo, M_use, a)
            prof *= T[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        norm = self._norm(cosmo)
        prof = norm * self.nfw._fourier(cosmo, k_use, M_use, a)

        if self.quantity == 'pressure':
            T = self._get_T(cosmo, M_use, a)
            prof *= T[:, None]

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileDensityNFW(_HaloProfileNFW):
    def __init__(self, *, mass_def, concentration,
                 truncated=True,
                 par_A=4.295,
                 par_B=0.514,
                 par_C=-0.039,
                 m_fid=3e14,
                 kind="rho_gas"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         truncated=truncated,
                         par_A=par_A, par_B=par_B, par_C=par_C,
                         m_fid=m_fid, kind=kind, quantity='density')


class HaloProfilePressureNFW(_HaloProfileNFW):
    def __init__(self, *, mass_def, concentration,
                 truncated=True,
                 par_A=4.295,
                 par_B=0.514,
                 par_C=-0.039,
                 m_fid=3e14,
                 kind="rho_gas"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         truncated=truncated,
                         par_A=par_A, par_B=par_B, par_C=par_C,
                         m_fid=m_fid, kind=kind, quantity='pressure')


class HaloProfileXray(ccl.halos.HaloProfile):
    """Profie corresponding to the X-ray emission. Specifically
    it computes the combination:

    X(r) = n_H(r) * n_e(r) * J(T(r))

    where n_H and n_e are the hydrogen and electron number
    densities (in units of cm^-3), and J is the integrated
    spectrum (in units of cm^-5 s^-1). The final quantity has
    units of cm^-1 s^-1, which we then transform to Mpc^-1 s^-1
    (since the Limber integrals are performed with distance in
    units of Mpc).

    dens -> a gas density profile
    pres -> a gas pressure profile
    temp -> a gas temperature profile
    Jinterp -> an interpolator containing J(ln(kT),z),
               with kT in keV.

    Calculating the Fourier transform of this profile is rather
    slow, but we have verified that the following empirical form
    works quite well:

    X(k) = X_0 /(1+(k/k0)^gamma)^(alpha/gamma)

    where X_0 is given by the integral over volume of the real-
    space profile, and alpha = 3+beta, where beta is the
    logarithmic tilt of the real-space spectrum on small scales.
    The pivot scale k_0 can be found by evaluating the Fourier
    transform numerically at very high k, and the intermediate
    slope gamma can be found by evaluating it around the pivot
    scale. `_get_fourier_params` precomputes all these parameters
    as a function of mass and redshift for a given cos mology. The
    function `_get_fourier_num` can be used to calculate the
    Fourier transform exactly (but slowly) in order to check this
    approximation.
    """

    def __init__(self, *, mass_def,
                 Jinterp,
                 dens,
                 pres=None,
                 lMmin_fit=11,
                 lMmax_fit=15.5,
                 nlM_fit=16,
                 zmax_fit=2.0,
                 nz_fit=8,
                 with_clumping=False,
                 fourier_approx=True,
                 plaw_fourier=-2.0,
                 truncated=False):

        self.dens = dens
        self.pres = pres
        self.truncated = truncated
        # Check the density and pressure are for the
        # same quantity (otherwise recovered temperature
        # won't make sense).
        if self.dens.kind != self.pres.kind:
            raise ValueError("Density and pressure profiles must "
                             "correspond to the same species "
                             f"{self.dens.kind} != {self.pres.kind}")
        self.Jinterp = Jinterp
        self.lkT_max = Jinterp.grid[0][-1]
        self.lkT_min = Jinterp.grid[0][0]
        self.with_clumping = with_clumping
        self.fourier_approx = fourier_approx
        self.lMmin_fit = lMmin_fit
        self.lMmax_fit = lMmax_fit
        self.nlM_fit = nlM_fit
        self.zmax_fit = zmax_fit
        self.nz_fit = nz_fit
        self.lf0 = None
        self.sl_hi = None
        self.q_piv = None
        self.sl_mid = None
        self.cosmo = None
        self.mass_def = None
        if fourier_approx:
            self._fourier = self._fourier_approx
        # Transforms to n_H * n_e
        pref_dens = get_prefac_rho(self.dens.kind)
        pref_H = get_prefac_rho("n_H")
        pref_e = get_prefac_rho("n_electron")
        self.pref_nHne = pref_H * pref_e / pref_dens**2
        super().__init__(mass_def=mass_def)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=plaw_fourier)

    def _get_fourier_params(self, cosmo):
        xmax = 100.0
        if self.truncated:
            xmax = 1.0
        l10Ms = np.linspace(self.lMmin_fit, self.lMmax_fit, self.nlM_fit)
        Ms = 10.0**l10Ms
        zs = np.linspace(0.0, self.zmax_fit, self.nz_fit)
        x = np.geomspace(1e-3, xmax, 32)
        lx = np.log(x)
        qhi = 50.0
        f0s = np.zeros([self.nz_fit, self.nlM_fit])
        sl_hik = np.zeros([self.nz_fit, self.nlM_fit])
        q_piv = np.zeros([self.nz_fit, self.nlM_fit])
        sl_midk = np.zeros([self.nz_fit, self.nlM_fit])
        for iz, z in enumerate(zs):
            a = 1.0 / (1 + z)
            rDelta = self.mass_def.get_radius(cosmo, Ms, a) / a
            pr = np.array(
                [self._real(cosmo, r0 * x, M, a)
                 for r0, M in zip(rDelta, Ms)]
            )
            f0 = simps(pr * x[None, :] ** 3, x=lx, axis=-1)
            tilt_lowr = np.log(pr[:, 1] / pr[:, 0]) / (lx[1] - lx[0])
            slope_hik = 3 + tilt_lowr
            fint = [
                interp1d(
                    lx,
                    np.log(p),
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                for p in pr
            ]
            fhi = np.array(
                [
                    quad(
                        lambda y: np.exp(f(np.log(y))) * y,
                        a=1e-5,
                        b=xmax,
                        weight="sin",
                        wvar=qhi,
                    )[0]
                    / qhi
                    for f in fint
                ]
            )
            q_pivot = qhi / (f0 / fhi - 1) ** (1 / slope_hik)
            fmid = np.array(
                [
                    quad(
                        lambda y: np.exp(f(np.log(y))) * y,
                        a=1e-5,
                        b=xmax,
                        weight="sin",
                        wvar=qp,
                    )[0]
                    / qp
                    for f, qp in zip(fint, q_pivot)
                ]
            )
            slope_midk = slope_hik * np.log(2) / np.log(f0 / fmid)
            f0s[iz, :] = f0
            sl_hik[iz, :] = slope_hik
            q_piv[iz, :] = q_pivot
            sl_midk[iz, :] = slope_midk
        self.cosmo = cosmo
        self.lf0 = RegularGridInterpolator(
            [zs, l10Ms],
            np.log(f0s),
            fill_value=None,
            bounds_error=False,
            method="linear")
        self.sl_hi = RegularGridInterpolator(
            [zs, l10Ms], sl_hik,
            fill_value=None,
            bounds_error=False,
            method="linear"
        )
        self.q_piv = RegularGridInterpolator(
            [zs, l10Ms], q_piv,
            fill_value=None,
            bounds_error=False,
            method="linear"
        )
        self.sl_mid = RegularGridInterpolator(
            [zs, l10Ms], sl_midk,
            fill_value=None,
            bounds_error=False,
            method="linear"
        )

    def _get_fourier_num(self, k, cosmo, M, a):
        r0 = self.mass_def.get_radius(cosmo, M, a) / a
        func = lambda x: self._real(cosmo, r0 * x, M, a) * x
        qs = r0 * k
        fq = np.array(
            [quad(func, a=1e-5, b=np.inf, weight="sin", wvar=q)[0] /
             q for q in qs]
        )
        return fq * 4 * np.pi * r0**3

    def _should_recompute(self, cosmo):
        if self.lf0 is None:
            return True
        if cosmo is not self.cosmo:
            return True

    def _fourier_approx(self, cosmo, k, M, a):
        if self._should_recompute(cosmo):
            with ccl.UnlockInstance(self):
                self._get_fourier_params(cosmo)
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        # r0
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        # f0
        ev = np.array([np.full_like(M_use, 1 / a - 1.0), np.log10(M_use)]).T
        f0 = np.exp(self.lf0(ev)) * 4 * np.pi * rDelta**3
        # high-k slope
        sl_hi = self.sl_hi(ev)
        # pivot scale
        q_piv = self.q_piv(ev)
        # mid-k slope
        sl_mid = self.sl_mid(ev)
        # Put everything together
        y = k_use[None, :] * (rDelta / q_piv)[:, None]
        prof = f0[:, None] / (1 + y ** sl_mid[:, None]) **\
            ((sl_hi / sl_mid)[:, None])

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        nr = len(r_use)
        nM = len(M_use)

        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        # Number density in cm^-3
        ndens = self.dens.real(cosmo, r_use, M_use, a).flatten()
        # Pressure in eV * cm^-3
        P = self.pres.real(cosmo, r_use, M_use, a).flatten()
        # log-Temperature in keV
        lkT = np.zeros(nM*nr)
        good = ndens > 0
        lkT[good] = np.log(1e-3 * P[good] / ndens[good])
        if not np.all(good):  # Set temperature to a tiny number where n=0
            lkT[~good] = -30.0
        # Integrated spectrum in cm^-5 s^-1
        z = 1.0 / a - 1
        ev = np.array([lkT, np.full_like(lkT, z)]).T
        J = np.exp(self.Jinterp(ev))
        # Clumping factor
        c2r = np.ones([nM, nr])
        if self.with_clumping:
            M_c2r = M_use.copy()
            M_c2r[M_c2r < 7e13] = 7e13
            M_c2r[M_c2r > 1e15] = 1e15
            xc = 9.91e5 * (1e-14 * M_c2r) ** (-4.87)
            beta = (0.185 * (1e-14 * M_c2r) ** 0.547)
            gamma = (1.16e6 * (1e-14 * M_c2r) ** (-4.86))
            for im in range(nM):
                rD = rDelta[im]
                # Only use fitting function up to 3 times virial radius
                # Constant afterwards
                x = np.minimum(r_use/rDelta[im], 5.0)
                xxc = x / xc[im]
                c2r[im, :] = 1 + xxc**beta[im] * (1+xxc)**(gamma[im]-beta[im])
        # Final profile in cm^-1
        prof = c2r * (ndens**2 * J).reshape([nM, nr])
        # Transform to Mpc^-1
        prof *= 3.08567758e24 * self.pref_nHne
        if self.truncated:
            for iM, rD in enumerate(rDelta):
                prof[iM, r_use > rD] = 0.0

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


def XrayTracer(cosmo, z_min=0.0, z_max=2.0, n_chi=1024):
    """Specific :class:`Tracer` associated with X-ray flux.
    The radial kernel for this tracer is simply

    .. math::
       W(\\chi) = \\frac{1}{4\\pi(1+z)^3}.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        zmin (float): minimum redshift down to which we define the
            kernel.
        zmax (float): maximum redshift up to which we define the
            kernel.
        n_chi (float): number of intervals in the radial comoving
            distance on which we sample the kernel.
    """
    tracer = ccl.Tracer()

    chi_max = ccl.comoving_radial_distance(cosmo, 1.0 / (1 + z_max))
    chi_min = ccl.comoving_radial_distance(cosmo, 1.0 / (1 + z_min))
    chi_arr = np.linspace(chi_min, chi_max, n_chi)
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_arr)

    tracer.add_tracer(cosmo, kernel=(chi_arr, a_arr**3 / (4 * np.pi)))
    return tracer


class HaloProfileNFWBaryon(ccl.halos.HaloProfileMatter):
    """Gas density profile given by the sum of the density profile for
    the bound and the ejected gas, each modelled separetely for a halo
    in hydrostatic equilibrium.

    The density and mass fraction of the bound gas as well as the mass
    fraction of the ejected gas taken from Mead 2020, and the density
    of the ejected gas taken from Schneider & Teyssier 2016.

    Profile is calculated in units of M_sun Mpc^-3 if
    requesting mass density (`kind == 'rho_gas'`), or in cm^-3
    if requesting a number density. Allowed values for `kind`
    in the latter case are `'n_total'`, `'n_baryon'`, `'n_H'`,
    `'n_electron'`.

    Default values of all parameters correspond to the values
    found in Mead et al. 2020.
    """
    def __init__(self, *, mass_def, concentration,
                 lMc=14.0, beta=0.6, gamma=1.17,
                 A_star=0.03, sigma_star=1.2,
                 eta_b=0.5,
                 logTAGN=None,
                 quantity="density"):
        self._Bi = None
        if logTAGN is not None:
            lMc, gamma, _, _, _ = self.from_logTAGN(logTAGN)
        self.logTAGN = logTAGN
        self.lMc = lMc
        self.beta = beta
        self.gamma = gamma
        self.A_star = A_star
        self.eta_b = eta_b
        self.sigma_star = sigma_star
        self.quantity = quantity
        self.norm_interp = self.get_bound_norm_interp(self.gamma)
        self.fourier_interp = self.get_bound_fourier_interp(self.gamma)
        super().__init__(mass_def=mass_def, concentration=concentration)

    def get_bound_fourier_interp(self, gamma, get_fq=False):
        qs = np.geomspace(1E-3, 1E3, 128)
        fq = np.array([quad(lambda x: x*self._F_bound(x, 1/(gamma-1)),
                            1E-4, np.inf,
                            weight='sin', wvar=q)[0] / q
                       for q in qs])
        # Divide by value at q -> 0
        norm = quad(lambda x: x**2*self._F_bound(x, 1/(gamma-1)),
                    1E-4, np.inf)[0]
        fq /= norm
        ip = interp1d(np.log(qs), np.log(fq),
                      fill_value='extrapolate',
                      bounds_error=False)
        return ip


    def get_bound_norm_interp(self, gamma):
        cs = np.geomspace(1E-2, 100, 64)
        norms = np.array([quad(lambda x: x**2*self._F_bound(x, 1/(gamma-1)), 0, c)[0]
                          for c in cs])
        ip = interp1d(np.log(cs), np.log(norms),
                      fill_value='extrapolate', bounds_error=False)
        return ip

    def update_parameters(self,
                          lMc=None,
                          beta=None,
                          gamma=None,
                          A_star=None,
                          sigma_star=None,
                          eta_b=None,
                          logTAGN=None):
        if logTAGN is not None:
            lMc, gamma, _, _, _ = self.from_logTAGN(logTAGN)
            self.logTAGN = logTAGN
        if lMc is not None:
            self.lMc = lMc
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
            self.norm_interp = self.get_bound_norm_interp(self.gamma)
            self.fourier_interp = self.get_bound_fourier_interp(self.gamma)
        if A_star is not None:
            self.A_star = A_star
        if eta_b is not None:
            self.eta_b = eta_b
        if sigma_star is not None:
            self.sigma_star = sigma_star

    def _build_BAHAMAS_interp(self):
        if self._Bi is not None:
            return
        kwargs = {'kind': 'linear',
                  'bounds_error': False,
                  'fill_value': 'extrapolate'}
        logTAGNs = np.array([7.6, 7.8, 8.0])
        self._Bi = {}
        lMci = interp1d(logTAGNs, np.array([13.1949, 13.5937, 14.2480]),
                        **kwargs)
        gammai = interp1d(logTAGNs, np.array([1.1647, 1.1770, 1.1966]),
                          **kwargs)
        alpha_Ti = interp1d(logTAGNs, np.array([0.7642, 0.8471, 1.0314]),
                            **kwargs)
        logTw0i = interp1d(logTAGNs, np.array([6.6762, 6.6545, 6.6615]),
                           **kwargs)
        Tw1i = interp1d(logTAGNs, np.array([-0.5566, -0.3652, -0.0617]),
                        **kwargs)
        self._Bi['lMc'] = lMci
        self._Bi['gamma'] = gammai
        self._Bi['alpha_T'] = alpha_Ti
        self._Bi['logTw0'] = logTw0i
        self._Bi['Tw1'] = Tw1i

    def from_logTAGN(self, logTAGN):
        self._build_BAHAMAS_interp()
        lMc = self._Bi['lMc'](logTAGN)
        gamma = self._Bi['gamma'](logTAGN)
        alpha_T = self._Bi['alpha_T'](logTAGN)
        logTw0 = self._Bi['logTw0'](logTAGN)
        Tw1 = self._Bi['Tw1'](logTAGN)
        return lMc, gamma, alpha_T, logTw0, Tw1

    def _get_fractions(self, cosmo, M):
        fb = get_fb(cosmo)
        f_cold = 1-fb
        Mbeta = (cosmo['h']*M*10**(-self.lMc))**self.beta
        f_bound = fb*Mbeta/(1+Mbeta)
        f_star = self.A_star * \
            np.exp(-0.5*((np.log10(cosmo["h"]*M)-12.5)/self.sigma_star)**2)
        f_ejected = fb-f_bound-f_star
        return f_cold, f_bound, f_ejected, f_star

    def _F_bound(self, x, G):
        return (np.log(1 + x) / x)**G

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        Delta = self.mass_def.get_Delta(cosmo, a)
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        rs = rDelta/cM
        x = r_use[None, :] / rs[:, None]

        # Mass fractions
        fc, fb, fe, _ = self._get_fractions(cosmo, M)

        # Cold dark matter
        norm_cold = np.log(1+cM)-cM/(1+cM)
        shape_cold = 1/(x*(1+x)**2)
        rho_cold = (M_use*fc/(4*np.pi*rs**3*norm_cold))[:, None]*shape_cold

        # Bound gas
        G = 1./(self.gamma-1)
        norm_bound = np.exp(self.norm_interp(np.log(cM)))
        shape_bound = self._F_bound(x, G)
        rho_bound = (M_use*fb/(4*np.pi*rs**3*norm_bound))[:, None]*shape_bound

        # Ejected gas
        # Eq. (2.13) of Schneider & Teyssier 2016
        x_esc = (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
        rho_ejected = (M_use*fe/rs**3)[:, None] * \
            np.exp(-0.5*(x/x_esc)**2)/(2*np.pi*x_esc**2)**1.5

        prof = rho_cold + rho_bound + rho_ejected

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a):
        # Real-space profile.
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        Delta = self.mass_def.get_Delta(cosmo, a)
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        rs = rDelta/cM
        x = k_use[None, :] * rs[:, None]

        # Mass fractions
        fc, fb, fe, fs = self._get_fractions(cosmo, M)

        # Cold dark matter
        norm_cold = np.log(1+cM)-cM/(1+cM)
        Si1, Ci1 = sici((1+cM)[:, None]*x)
        Si2, Ci2 = sici(x)
        p1 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        p2 = np.sin(cM[:, None] * x) / ((1 + cM[:, None]) * x)
        shape_cold = p1-p2
        rho_cold = (M_use*fc/norm_cold)[:, None]*shape_cold

        # Bound gas
        # Already normalised!
        norm_bound = 1.0
        shape_bound = np.exp(self.fourier_interp(np.log(x)))
        rho_bound = (M_use*fb/norm_bound)[:, None]*shape_bound

        # Ejected gas
        # Eq. (2.13) of Schneider & Teyssier 2016
        x_esc = (self.eta_b * 0.375 * np.sqrt(Delta) * cM)[:, None]
        rho_ejected = (M_use*fe)[:, None] * np.exp(-0.5*(x*x_esc)**2)

        # Stars
        rho_stars = (M_use*fs)[:, None] * np.ones_like(k_use)[None, :]

        prof = rho_cold + rho_bound + rho_ejected + rho_stars

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
