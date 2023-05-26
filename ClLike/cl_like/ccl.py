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
            ('sigma8' not in self.input_params):
            raise ValueError("One of A_sE9 or sigma8 must be set")

    def get_requirements(self):
        # Specify A_sE9 and sigma8 in get_can_support_params to allow both
        # inputs.
        params = {'Omega_m': None,
                  'omega_b': None,
                  'n_s': None,
                  'm_nu': None}

        return params

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
        return ['S8', 'sigma8', "Omega_c", "Omega_b", "h"]

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return ["sigma8", "A_sE9", "h", "Omh3"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Generate the CCL cosmology object which can then be used downstream
        Om = self.provider.get_param('Omega_m')
        if 'Omh3' in self.input_params:
            Omh3 = self.provider.get_param('Omh3')
            h = (Omh3/Om)**(1./3.)
        else:
            h = self.provider.get_param('h')
        ob = self.provider.get_param('omega_b')
        Ob = ob/h**2
        Oc = Om-Ob

        params = {'Omega_c': Oc, 'Omega_b': Ob, 'h': h,
                  'n_s': self.provider.get_param('n_s'),
                  'm_nu': self.provider.get_param('m_nu')}

        if 'A_sE9' in self.input_params:
            params.update({'A_s': self.provider.get_param('A_sE9')*1E-9})
        else:
            params.update({'sigma8': self.provider.get_param('sigma8')})

        cosmo = ccl.Cosmology(**params,
                              T_CMB=2.7255,
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_pk,
                              baryons_power_spectrum=self.baryons_pk,
                              **self.ccl_arguments)

        state['CCL'] = {'cosmo': cosmo}

        # Compute derived parameters
        # (we should actually only do this if required -- TODO)
        # Compute sigma8 if it is not an input parameter
        state['derived'] = {}
        if 'A_sE9' in self.input_params:
            sigma8 = ccl.sigma8(cosmo)
            state['derived']['sigma8'] = sigma8
        else:
            sigma8 = cosmo['sigma8']

        if 'Omh3' in self.input_params:
            state['derived']['h'] = h

        state['derived'].update({'S8': sigma8*np.sqrt(Om/0.3),
                                 'Omega_c': Oc, 'Omega_b': Ob})

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
