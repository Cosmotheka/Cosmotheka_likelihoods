import numpy as np
import pyccl as ccl
from scipy.integrate import quad, simps
from scipy.interpolate import RegularGridInterpolator, interp1d


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


def get_prefac_T(T_kind, XH=0.76):
    """Prefactor that transforms a quantity in Mpc^3/M_sun
    into cm^3/m_p, and provides the correct value for the
    mean gas particle mass (divided by the proton mass),
    depending on the hydrogen mass fraction XH (BBN value
    by default), or the simulation.
    """
    Mprotcm2MsunMpc2 = 2.47054519e16
    if T_kind == "bahamas":
        return Mprotcm2MsunMpc2 * 0.61
    elif T_kind == "T_total":
        return Mprotcm2MsunMpc2 * 4 / (5 * XH + 3)
    else:
        raise NotImplementedError(f"Temperature type {T_kind} not implemented")


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

        re_fourier = False
        if xc is not None:
            if xc != self.xc:
                re_fourier = True
            self.xc = xc
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
                 qrange=(1e-3, 1e3),
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
        f_arr[-1, :] = 1e-100
        Fqb = RegularGridInterpolator(
            [np.log(qs), betas],
            np.log(f_arr),
            fill_value=None,
            bounds_error=False,
            method="cubic",
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
                 beta=0.6,
                 gamma=1.17,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0,
                 kind="rho_gas",
                 kind_T="T_total",
                 quantity="density"):
        self.beta = beta
        self.gamma = gamma
        self.A_star = A_star
        self.eta_b = eta_b
        self.alpha_T = alpha_T
        self.sigma_star = sigma_star
        self.kind = kind
        self.quantity = quantity
        self.prefac_rho = get_prefac_rho(self.kind)
        self.prefac_T = get_prefac_T(kind_T)

        super().__init__(mass_def=mass_def, concentration=concentration)

    def update_parameters(self,
                          beta=None,
                          gamma=None,
                          A_star=None,
                          sigma_star=None,
                          alpha_T=None,
                          eta_b=None):
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if A_star is not None:
            self.A_star = A_star
        if eta_b is not None:
            self.eta_b = eta_b
        if sigma_star is not None:
            self.sigma_star = sigma_star
        if alpha_T is not None:
            self.alpha_T = alpha_T

    def _fb_bound(self, cosmo, M):
        part1 = get_fb(cosmo)
        part2 = (cosmo["h"] * M * 1e-14) ** self.beta
        part3 = part2 / (1 + part2)
        return part1 * part3

    def _fb_ejected(self, cosmo, M):
        part1 = get_fb(cosmo)
        part2 = self._fb_bound(cosmo, M)
        part3 = self.A_star * np.exp(
            -(
                (np.log10(cosmo["h"] * M * 10 ** (-12.5)) ** 2)
                / (2 * self.sigma_star**2)
            )
        )
        return part1 - part2 - part3

    def _rho_bound(self, x):
        return (np.log(1 + x) / x) ** (1 / (self.gamma - 1))

    def _rho_ejected(self, x, cosmo, M, a):
        # Eq. (2.13) of Schneider & Teyssier 2016
        r200 = self.mass_def.get_radius(cosmo, M, a) / a
        delta200 = self.mass_def.get_Delta(cosmo, a)
        r_esc = 0.5 * np.sqrt(delta200) * r200
        eta_a = 0.75 * self.eta_b
        Re2 = ((eta_a * r_esc)**2)[:, None]
        return M[:, None] / (2*np.pi*Re2)**1.5 * np.exp(-(x**2/(2*Re2)))

    def _get_rho0(self, cosmo, M, r, c_M):
        # This integral can be precomputed if it's too slow
        integral = np.array(
            [quad(lambda x: self._rho_bound(x) * x**2, 0, c)[0] for c in c_M]
        )
        fb = self._fb_bound(cosmo, M)
        rho0 = M * fb / (4 * np.pi * r**3 * integral)
        return rho0

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        rho0 = self._get_rho0(cosmo, M_use, R_s, c_M)
        rho_bound = self._rho_bound(x) * rho0[:, None]
        rho_ejected = (self._rho_ejected(r_use[None, :], cosmo, M_use, a) *
                       self._fb_ejected(cosmo, M_use)[:, None])

        if self.quantity == 'density':
            prof = (rho_bound + rho_ejected) * self.prefac_rho
        elif self.quantity == 'pressure':
            # Boltmann constant which, when multiplied by T in Kelvin
            # gives you eV
            k_boltz = 8.61732814974493e-05
            T_ejected = 10**6.5*k_boltz

            # Gravitational constant in eV*(Mpc^4)/(cm^3*Msun^2)
            G = 1.81805235e-27
            factor = np.log(1+x)/x
            T_bound = factor * self.alpha_T * 2 * G * M_use[:, None]
            T_bound *= self.prefac_T / (3 * a * R_M[:, None])

            # Put them together
            prof = (rho_bound*T_bound+rho_ejected*T_ejected)*self.prefac_rho

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileDensityHE(_HaloProfileHE):
    def __init__(self, *, mass_def, concentration,
                 beta=0.6,
                 gamma=1.17,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0,
                 kind="rho_gas",
                 kind_T="T_total"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         beta=beta,
                         gamma=gamma,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         alpha_T=alpha_T,
                         kind=kind,
                         kind_T=kind_T,
                         quantity="density")


class HaloProfilePressureHE(_HaloProfileHE):
    def __init__(self, *, mass_def, concentration,
                 beta=0.6,
                 gamma=1.17,
                 A_star=0.03,
                 sigma_star=1.2,
                 eta_b=0.5,
                 alpha_T=1.0,
                 kind="rho_gas",
                 kind_T="T_total"):
        super().__init__(mass_def=mass_def, concentration=concentration,
                         beta=beta,
                         gamma=gamma,
                         A_star=A_star,
                         sigma_star=sigma_star,
                         eta_b=eta_b,
                         alpha_T=alpha_T,
                         kind=kind,
                         kind_T=kind_T,
                         quantity="pressure")


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
    as a function of mass and redshift for a given cosmology. The
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
        if self.pres is not None:
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
                x = np.minimum(r_use/rDelta[im], 3.0)
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
