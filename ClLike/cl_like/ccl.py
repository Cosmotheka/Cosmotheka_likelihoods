"""
Simple CCL theory wrapper that returns the cosmology object
and optionally a number of methods depending only on that
object.

This is based on an earlier implementation by Antony Lewis:
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/ccl.py

`get_CCL` results a dictionary of results, where `results['cosmo']`
is the CCL cosmology object.

Classes that need other CCL-computed results (without additional
free parameters), should pass them in the requirements list.

e.g. a `Likelihood` with `get_requirements()` returning
`{'CCL': {'methods:{'name': self.method}}}`
[where self is the Likelihood instance] will have
`results['name']` set to the result
of `self.method(cosmo)` being called with the CCL cosmo
object.

The `Likelihood` class can therefore handle for itself which
results specifically it needs from CCL, and just give the
method to return them (to be called and cached by Cobaya with
the right parameters at the appropriate time).

Alternatively the `Likelihood` can compute what it needs from
`results['cosmo']`, however in this case it will be up to the
`Likelihood` to cache the results appropriately itself.

Note that this approach precludes sharing results other than
the cosmo object itself between different likelihoods.

Also note lots of things still cannot be done consistently
in CCL, so this is far from general.
"""
import numpy as np
import pyccl as ccl
import numpy as np
from packaging import version
from cobaya.theory import Theory


class CCL(Theory):
    """
    This implements CCL as a `Theory` object that takes in
    cosmological parameters directly (i.e. cannot be used
    downstream from camb/CLASS.
    """
    # CCL options
    transfer_function: str = 'boltzmann_camb'
    matter_pk: str = 'halofit'
    baryons_pk: str = 'nobaryons'
    ccl_arguments: dict = {}

    def initialize(self):
        self._required_results = {}

    def initialize_with_params(self):
        if ('A_sE9' not in self.input_params) and \
            ('sigma8' not in self.input_params) and \
                ('S8' not in self.input_params):
            raise ValueError("One of A_sE9, sigma8 or S8 must be set")
        if ('Omega_m' not in self.input_params) and \
                ('Omega_c' not in self.input_params):
            raise ValueError("One of Omega_c or Omega_m must be set")
        elif ('Omega_m' in self.input_params) and \
                ('Omega_c' in self.input_params):
            raise ValueError("Only one of Omega_c or Omega_m must be set")

    def get_allow_agnostic(self):
        # Pass all parameters without unknown prefix to this class
        return True

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from
        # the required outputs
        # So a lot of this is fixed
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        return {}

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['S8', 'sigma8', "Omega_m", "Omega_nu", "Omega_c"]

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return ["sigma8", "A_sE9", "Omega_m", "Omega_c", "S8"]

    def _get_ccl_param_or_arg(self, param_name, default):
        if param_name in self.ccl_arguments:
            value = self.ccl_arguments[param_name]
        elif param_name in self.input_params:
            value = self.provider.get_param(param_name)
        else:
            value = default

        return value

    def _get_Onuh2(self, m_nu, T_CMB, T_ncdm):
        # Onu = np.sum(self.provider.get_param('m_nu')) / 93.14 / self.provider.get_param('h')**2 # check consistency with ccl
        if version.parse(ccl.__version__) < version.parse("2.8.0"):
            Onu = ccl.neutrinos.Omeganuh2(1, m_nu, T_CMB=T_CMB)
        else:
            raise NotImplementedError("_get_Onuh2 needs testing for "
                                      "pyccl>2.8.0")
            # Copied from ccl/cosmology.py
            import pyccl.physical_constants as const
            c = const
            g = (4/11)**(1/3)
            T_nu = g * T_CMB
            massless_limit = T_nu * c.KBOlTZ / c.EV_IN_J

            from pyccl.neutrinos import nu_masses
            mass_split = self._get_ccl_param_or_arg('mass_split', 'normal')
            mnu_list = nu_masses(m_nu=m_nu, mass_split=mass_split)
            nu_mass = mnu_list[mnu_list > massless_limit]
            Onu = pyccl.cosmology.Cosmology._OmNuh2(m_nu, len(m_nu), T_CMB, T_ncdm)

        return Onu

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Generate the CCL cosmology object which can then be used downstream

        # Copy input_params. We will be removing elements that we extract from
        # it to pass to ccl.Cosmology all the remaining parameters.
        input_params = list(self.input_params)

        #
        h = self.provider.get_param('h')
        input_params.remove('h')
        Ob = self.provider.get_param('Omega_b')
        input_params.remove('Omega_b')
        m_nu = self.provider.get_param('m_nu')
        input_params.remove('m_nu')
        T_CMB = self._get_ccl_param_or_arg('T_CMB',
                                           ccl.physical_constants.T_CMB)
        T_ncdm = self._get_ccl_param_or_arg('T_ncdm', 0.71611)

        Onu = self._get_Onuh2(m_nu, T_CMB, T_ncdm) / h**2 if m_nu != 0 else 0
        if 'Omega_c' in self.input_params:
            Oc = self.provider.get_param('Omega_c')
            input_params.remove('Omega_c')
            Om = Ob + Oc + Onu
        else:
            Om = self.provider.get_param('Omega_m')
            input_params.remove('Omega_m')
            Oc = Om - Ob - Onu

        # Parameters accepted by CCL
        params = {'Omega_c': Oc,
                  'Omega_b': Ob,
                  'h': h,
                  'n_s': self.provider.get_param('n_s'),
                  'm_nu': m_nu}

        if 'A_sE9' in self.input_params:
            params.update({'A_s': self.provider.get_param('A_sE9')*1E-9})
            input_params.remove('A_sE9')
        elif 'S8' in self.input_params:
            sigma8 = self.provider.get_param('S8') * np.sqrt(0.3 / Om)
            input_params.remove('S8')
            params.update({'sigma8': sigma8})
        else:
            params.update({'sigma8': self.provider.get_param('sigma8')})
            input_params.remove('sigma8')

        for p in input_params:
            params[p] = self.provider.get_param(p)

        cosmo = ccl.Cosmology(**params,
                              T_CMB=T_CMB,
                              # T_ncdm=T_ncdm, v2.8.0
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_pk,
                              baryons_power_spectrum=self.baryons_pk,
                              **self.ccl_arguments)

        state['CCL'] = {'cosmo': cosmo}

        # Compute derived parameters
        # (we should actually only do this if required -- TODO)
        # Compute sigma8 if it is not an input parameter
        state['derived'] = {}
        if 'sigma8' not in self.input_params:
            sigma8 = ccl.sigma8(cosmo)
            state['derived']['sigma8'] = sigma8
        else:
            sigma8 = cosmo['sigma8']

        if 'S8' not in self.input_params:
            state['derived']['S8'] = sigma8*np.sqrt(Om/0.3)

        state['derived'].update({'Omega_m': Om,
                                 'Omega_nu': Onu,
                                 })
        if 'Omega_c' not in self.input_params:
            state['derived']['Omega_c'] = Oc

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
