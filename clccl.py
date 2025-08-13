"""
Simple CCL theory wrapper that returns the cosmology object and
 optionally a number of methods depending only on that object.

This is based on an earlier implementation by Antony Lewis:
 https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/ccl.py

This version is like the `CCL` class implemented in `ccl.py`, but
 this one is cosmology-less in the sense that the primary
 cosmological quantities (distances and P(k)) are taken externally.
"""

import numpy as np
import pyccl as ccl
from typing import Sequence, Union
from cobaya.theory import Theory


class CLCCL(Theory):
    """
    This implements CCL as a `Theory` object in a Cosmology-Less way.
    CCL is used downstream from camb/CLASS, which provide distances and
    (optionally) power spectra.
    """
    # Maximum wavenumber for camb/CLASS
    kmax: float = 0
    # Redshifts at which P(k) will be sampled
    z_pk: Union[Sequence, np.ndarray] = []
    # Redshifts at which distances will be sampled
    z_bg: Union[Sequence, np.ndarray] = []
    # CCL options
    transfer_function: str = 'boltzmann_camb'
    matter_pk: str = 'halofit'
    baryons_pk: str = 'nobaryons'
    # If True, CCL will take P(k)s from an upstream camb/CLASS object.
    external_nonlin_pk: bool = True

    # Default redshift samplings
    _default_z_pk_sampling = np.linspace(0, 5, 100)
    _default_z_bg_sampling = np.concatenate((np.linspace(0, 10, 100),
                                             np.geomspace(10, 1500, 50)))

    namedir = {'delta_tot': 'delta_matter'}

    def _translate_camb(self, pair):
        v1, v2 = pair
        return self.namedir[v1]+':'+self.namedir[v2]

    def initialize(self):
        # Pairs of quantities for which we want P(k)
        self._var_pairs = set()
        self._required_results = {}

    def get_requirements(self):
        return {'omch2', 'ombh2', 'ns', 'As', 'mnu'}

    def must_provide(self, **requirements):
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        # Sampling
        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z_pk = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z_pk", self._default_z_pk_sampling)),
             np.atleast_1d(self.z_pk))))
        self.z_bg = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z_bg", self._default_z_bg_sampling)),
             np.atleast_1d(self.z_bg))))

        # Dictionary of the things CCL needs from CAMB/CLASS
        needs = {}

        # Power spectra
        if self.kmax:
            self.external_nonlin_pk = (self.external_nonlin_pk or
                                       options.get('external_nonlin_pk',
                                                   False))
            # CCL currently only supports ('delta_tot', 'delta_tot'), but call
            # allow general as placeholder
            self._var_pairs.update(
                set((x, y) for x, y in
                    options.get('vars_pairs', [('delta_tot', 'delta_tot')])))

            if self.external_nonlin_pk:
                nonlin = (True, False)
            else:
                nonlin = False
            needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': nonlin,
                'z': self.z_pk,
                'k_max': self.kmax}

        # Background arrays
        needs['Hubble'] = {'z': self.z_bg}
        needs['comoving_radial_distance'] = {'z': self.z_bg}

        assert len(self._var_pairs) < 2, "CCL doesn't support other Pks yet"
        return needs

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['sigma8']

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        prov = self.provider
        # Get background from upstream
        distance = prov.get_comoving_radial_distance(self.z_bg)
        hubble_z = prov.get_Hubble(self.z_bg)
        H0 = hubble_z[0]
        E_of_z = hubble_z / H0
        distance = np.flip(distance)
        E_of_z = np.flip(E_of_z)

        # Translate into CCL parameters
        h = H0 * 0.01
        Omega_c = prov.get_param('omch2') / h**2
        Omega_b = prov.get_param('ombh2') / h**2

        # Generate cosmology and populate background
        a_bg = 1. / (1+self.z_bg[::-1])

        if self.kmax:
            pkln = {}
            if self.external_nonlin_pk:
                pknl = {}
            for pair in self._var_pairs:
                name = self._translate_camb(pair)
                k, z, Pk_lin = prov.get_Pk_grid(var_pair=pair,
                                                nonlinear=False)
                Pk_lin = np.flip(Pk_lin, axis=0)
                a = 1./(1+np.flip(z))
                pkln[name] = Pk_lin
                pkln['a'] = a
                pkln['k'] = k

                if self.external_nonlin_pk:
                    k, z, Pk_nl = prov.get_Pk_grid(var_pair=pair,
                                                   nonlinear=True)
                    Pk_nl = np.flip(Pk_nl, axis=0)
                    a = 1./(1+np.flip(z))
                    pknl[name] = Pk_nl
                    pknl['a'] = a
                    pknl['k'] = k
            cosmo = ccl.CosmologyCalculator(Omega_c=Omega_c,
                                            Omega_b=Omega_b, h=h,
                                            n_s=prov.get_param('ns'),
                                            A_s=prov.get_param('As'),
                                            T_CMB=2.7255,
                                            m_nu=prov.get_param('mnu'),
                                            background={'a': a_bg,
                                                        'chi': distance,
                                                        'h_over_h0': E_of_z},
                                            pk_linear=pkln,
                                            pk_nonlin=pknl)
        else:
            cosmo = ccl.CosmologyCalculator(Omega_c=Omega_c,
                                            Omega_b=Omega_b, h=h,
                                            n_s=prov.get_param('ns'),
                                            A_s=prov.get_param('As'),
                                            T_CMB=2.7255,
                                            m_nu=prov.get_param('mnu'),
                                            background={'a': a_bg,
                                                        'chi': distance,
                                                        'h_over_h0': E_of_z})

        state['CCL'] = {'cosmo': cosmo}
        # Compute sigma8 (we should actually only do this if required -- TODO)
        state['sigma8'] = ccl.sigma8(cosmo)
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']
