import pyccl as ccl
import numpy as np
import sacc
from . import common as co
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.interpolate import interp1d

class xCell_lkl(Likelihood):

    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = "xCell"
    # Input sacc file
    input_file: str = ""
    # Interpolate cls
    interpolate_cls: bool = True
    # List of bin names
    tracers: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # Dict of two-point functions that make up the data vector.
    #  - Keys are tuples with the tracer names that make the different
    #    combinations as given by sacc e.g. ('DESgc__0', 'DESgc__0')
    #  - YAML read ('DESgc__0', 'DESgc__0') as a string. We will first change
    #    it to lists in the initialize method
    #  - Values can be lmin, lmax
    tracer_combinations: dict = {}

    def initialize(self):
        # Rewrite tracer_combinations with keys as tuples not string
        self._rewrite_tracer_combinations()
        # Read SACC file
        self.scovG = self._load_sacc_file(self.input_file)
        # Store tracers info in a metadata dictionary. This is specially
        # useful to avoid computing the bandpower function in each call.
        self.xCell_metadata = self._extract_tracers_metadata(self.scovG)
        # Load data vector
        self.data_vec = self.scovG.mean
        # Load covariance matrix
        self.cov = self.scovG.covariance.covmat
        # Store inverse covariance
        self.icov = np.linalg.inv(self.cov)

    def _rewrite_tracer_combinations(self):
        # Check if the dictionary keys are already tuple
        k0 =  list(self.tracer_combinations.keys())[0]
        if type(k0) is not tuple:
            # If they're not, change str -> tuple
            d = {}
            for k, v in self.tracer_combinations.items():
                k1 = tuple(k.replace('(', '').replace(')', '').replace(',', '').split())
                d[k1] = v

            self.tracer_combinations = d.copy()
        return

    def _load_sacc_file(self, sacc_file):
        print(f'Loading {sacc_file}')
        s = sacc.Sacc.load_fits(sacc_file)
        # Check used tracers are in the sacc file
        tracers_sacc = [trd for trd in s.tracers]
        # Check all tracers are in the sacc file
        for tr in self.tracers:
            if tr not in tracers_sacc:
                raise ValueError('The tracer {} is not present in {}'.format(tr, sacc_file))

        # Remove not needed tracers and corresponding Cls
        for tr in tracers_sacc:
            if tr not in self.tracers:
                # Loop through the tracer combinatios to spot those with the
                # unused tracer
                for trs_sacc in s.get_tracer_combinations():
                    if tr in trs_sacc:
                        s.remove_selection(tracers=trs_sacc)
                del s.tracers[tr]

        # Remove B modes
        for dt in s.get_data_types():
            if 'b' in dt:
                s.remove_selection(data_type=dt)

        for trs in s.get_tracer_combinations():
            # Check if trs is in tracer_combinations
            if trs not in self.tracer_combinations:
                trs = trs[::-1]
            # Check if trs ordered the other way round is in
            # tracer_combinations and remove it if not
            if trs not in self.tracer_combinations:
                s.remove_selection(tracers=trs)
                continue

            trs_val = self.tracer_combinations[trs]
            lmin = trs_val.get('lmin', self.defaults['lmin'])
            lmax = trs_val.get('lmax', self.defaults['lmax'])
            print(trs, lmin, lmax)
            s.remove_selection(ell__lt=lmin, tracers=trs)
            s.remove_selection(ell__gt=lmax, tracers=trs)

        print()

        return s

    def _get_dtype_for_trs(self, tr1, tr2):
        dt1 = self.scovG.get_tracer(tr1).quantity
        dt2 = self.scovG.get_tracer(tr2).quantity

        dt_to_suffix = {'galaxy_density': '0', 'cmb_convergence': '0',
                        'galaxy_shear': 'e'}

        dtype = 'cl_'

        dtype += dt_to_suffix[dt1]
        dtype += dt_to_suffix[dt2]

        if dtype == 'cl_e0':
            dtype = 'cl_0e'

        return dtype

    def _extract_tracers_metadata(self, sacc):
        metadata = {}

        for tr1, tr2 in self.scovG.get_tracer_combinations():
            dtype = self._get_dtype_for_trs(tr1, tr2)
            ell, cl, cov, ind = self.scovG.get_ell_cl(dtype, tr1, tr2,
                                                      return_cov=True,
                                                      return_ind=True)
            bpw = self.scovG.get_bandpower_windows(ind)
            metadata[(tr1, tr2)] = {'ell_eff': ell, 'cl': cl,
                                    'cov': cov, 'ind': ind,
                                    'ell_bpw': bpw.values,
                                    'w_bpw': bpw.weight.T}

        return metadata

    def _get_dndz(self, trname, sacc_tr, **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname])
        z = sacc_tr.z
        dz = pars.get(pars_prefix + '_dz', 0)
        z -= dz
        z_sel = z>=0
        z = z[z_sel]
        nz = sacc_tr.nz[z_sel]

        return z, nz

    def _get_ccl_tracer_gc(self, cosmo, trname, sacc_tr,  **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname])

        # Shift the redshift mean
        z, nz = self._get_dndz(trname, sacc_tr, **pars)

        # Galaxy bias
        b = pars.get(pars_prefix + '_gc_b')
        bz = b * np.ones_like(z)

        # Magnification bias
        s = pars.get(pars_prefix + '_gc_s', None)
        mag_bias = None
        if s is not None:
            mag_bias = (z, s * np.ones_like(z))

        return ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz),
                                      bias=(z, bz), mag_bias=mag_bias)

    def _get_ccl_tracer_sh(self, cosmo, trname, sacc_tr,  **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname])

        # Shift the redshift mean
        # Shift the redshift mean
        z, nz = self._get_dndz(trname, sacc_tr, **pars)

        # Intrinsic Alignments (same for all wl tracers)
        A = pars.get(self.input_params_prefix + '_' +  'wl_ia_A')
        eta = pars.get(self.input_params_prefix + '_' +  'wl_ia_eta')
        z0 = pars.get(self.input_params_prefix + '_' +  'wl_ia_z0')
        # pyccl2 -> has already the factor inside. Only needed bz
        IAz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474
        # Get tracer
        return ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=(z, IAz))

    def _get_ccl_tracer_cv(self, cosmo, trname, sacc_tr,  **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname, 'cv'])
        return ccl.CMBLensingTracer(cosmo, z_source=1100)

    def _get_ccl_tracers(self, **pars):
        res = self.provider.get_result('CCL')
        cosmo = res['cosmo']
        ccl_tracers = {}

        # Get Tracers
        for trname, sacc_tr in self.scovG.tracers.items():
            if sacc_tr.quantity == 'galaxy_density':
                tr = self._get_ccl_tracer_gc(cosmo, trname, sacc_tr, **pars)
            elif sacc_tr.quantity == 'galaxy_shear':
                tr = self._get_ccl_tracer_sh(cosmo, trname, sacc_tr, **pars)
            elif sacc_tr.quantity == 'cmb_convergence':
                tr = self._get_ccl_tracer_cv(cosmo, trname, sacc_tr, **pars)
            else:
                raise ValueError(f'Tracer type {sacc_tr.quantity} not implemented')
            ccl_tracers[trname] = tr

        return ccl_tracers

    def _get_cl_theory(self, **pars):
        res = self.provider.get_result('CCL')
        cosmo = res['cosmo']
        ccl_tracers = self._get_ccl_tracers(**pars)
        cl_th = np.array([])
        for tr1, tr2 in self.scovG.get_tracer_combinations():
            ell_bpw = self.xCell_metadata[(tr1, tr2)]['ell_bpw']
            w_bpw = self.xCell_metadata[(tr1, tr2)]['w_bpw']
            cl_trs = co.get_binned_cl(cosmo, ccl_tracers[tr1],
                                      ccl_tracers[tr2], ell_bpw, w_bpw,
                                      interp=self.interpolate_cls)
            # Multiplicative bias here because it is absourdly cheap to
            # compute when varying
            m1 = pars.get(self.input_params_prefix + '_' + tr1 + '_wl_m', 0)
            m2 = pars.get(self.input_params_prefix + '_' + tr2 + '_wl_m', 0)
            cl_trs *= (1 + m1) * (1 + m2)
            cl_th = np.concatenate([cl_th, cl_trs])
        cl_th = np.array(cl_th)
        return cl_th

    def get_requirements(self):
        return {'CCL': {'cosmo': None}}


    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_cl_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, self.icov.dot(r))
        return -0.5 * chi2
