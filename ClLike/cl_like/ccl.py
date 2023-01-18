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

    def initialize(self):
        self._required_results = {}
        self.pk_options = None

    def initialize_with_params(self):
        if ('A_sE9' not in self.input_params) and \
            ('sigma8' not in self.input_params):
            raise ValueError("One of A_sE9 or sigma8 must be set")

    def get_requirements(self):
        # Specify A_sE9 and sigma8 in get_can_support_params to allow both
        # inputs.
        params = {'Omega_c': None,
                  'Omega_b': None,
                  'h': None,
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

        self.pk_options = options.get("pk_options")
        return {}

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['S8', 'sigma8', "Omega_m"]

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return ["sigma8", "A_sE9"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Generate the CCL cosmology object which can then be used downstream
        Ob = self.provider.get_param('Omega_b')
        Oc = self.provider.get_param('Omega_c')
        Om = Ob + Oc

        params = {'Omega_c': Oc,
                  'Omega_b': Ob,
                  'h': self.provider.get_param('h'),
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
                              baryons_power_spectrum=self.baryons_pk)

        state['CCL'] = {'cosmo': cosmo,
                        "pk_data": self._get_pk_data(cosmo)}

        # Compute derived parameters
        # (we should actually only do this if required -- TODO)
        # Compute sigma8 if it is not an input parameter
        if 'A_sE9' in self.input_params:
            sigma8 = ccl.sigma8(cosmo)
            state['derived'] = {'sigma8': sigma8}
        else:
            sigma8 = cosmo['sigma8']

        state['derived'].update({'S8': sigma8*np.sqrt(Om/0.3),
                                 'Omega_m': Om})

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

    def get_cosmo(self):
        "Get the current CCL Cosmology object"
        return self._current_state['CCL']["cosmo"]

    def get_pk_data(self):
        "Get the current pk_data dictionary"
        return self._current_state['CCL']["pk_data"]

    def _get_pk_data(self, cosmo):
        # TODO: I don't like it reading the pk_options dictionary. I think it'd
        # be better if one could pass the options as argument. Kept like this
        # for now because it's less work
        bias_model = self.pk_options["bias_model"]
        is_PT_bias = self.pk_options["is_PT_bias"]

        cosmo.compute_nonlin_power()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        if bias_model == 'Linear':
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_d1m'] = pkmm
            pkd['pk_d1d1'] = pkmm
        elif is_PT_bias:
            k_SN_suppress = self.pk_options["k_SN_suppress "]
            if k_SN_suppress > 0:
                k_filter = k_SN_suppress
            else:
                k_filter = None
            if bias_model == 'EulerianPT':
                from .ept import EPTCalculator
                EPTkwargs = self.pk_options["EPTkwargs"]
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=EPTkwargs["l10k_min_pks"],
                                    log10k_max=EPTkwargs["l10k_max_pks"],
                                    nk_per_decade=EPTkwargs["nk_per_dex_pks"],
                                    a_arr=EPTkwargs["a_s_pks"],
                                    k_filter=k_filter)
            else:
                raise NotImplementedError("Not yet: " + bias_model)
            pk_lin_z0 = ccl.linear_matter_power(cosmo, ptc.ks, 1.)
            Dz = ccl.growth_factor(cosmo, ptc.a_s)
            ptc.update_pk(pk_lin_z0, Dz)
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_md2'] = ptc.get_pk('d1d2')
            pkd['pk_ms2'] = ptc.get_pk('d1s2')
            pkd['pk_mk2'] = ptc.get_pk('d1k2', pgrad=pkmm, cosmo=cosmo)
            pkd['pk_d1m'] = pkd['pk_md1']
            pkd['pk_d1d1'] = pkmm
            pkd['pk_d1d2'] = pkd['pk_md2']
            pkd['pk_d1s2'] = pkd['pk_ms2']
            pkd['pk_d1k2'] = pkd['pk_mk2']
            pkd['pk_d2m'] = pkd['pk_md2']
            pkd['pk_d2d1'] = pkd['pk_d1d2']
            pkd['pk_d2d2'] = ptc.get_pk('d2d2')
            pkd['pk_d2s2'] = ptc.get_pk('d2s2')
            pkd['pk_d2k2'] = None
            pkd['pk_s2m'] = pkd['pk_ms2']
            pkd['pk_s2d1'] = pkd['pk_d1s2']
            pkd['pk_s2d2'] = pkd['pk_d2s2']
            pkd['pk_s2s2'] = ptc.get_pk('s2s2')
            pkd['pk_s2k2'] = None
            pkd['pk_k2m'] = pkd['pk_mk2']
            pkd['pk_k2d1'] = pkd['pk_d1k2']
            pkd['pk_k2d2'] = pkd['pk_d2k2']
            pkd['pk_k2s2'] = pkd['pk_s2k2']
            pkd['pk_k2k2'] = None
        return pkd
