import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import copy
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.optimize import minimize
import sacc


class ClLike(Likelihood):
    # Input sacc file
    input_file: str = ""
    # Angular resolution
    nside: int = -1
    # List of bin names
    bins: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # List of two-point functions that make up the data vector
    twopoints: list = []
    # Jeffreys prior for bias params?
    jeffrey_bias: bool = False
    # Null negative covariance eigenvalues when computing inverse cov?
    null_negative_cov_eigvals_in_icov: bool = False

    def initialize(self):
        # Deep copy defaults to avoid modifying the input yaml
        self.defaults = copy.deepcopy(self.defaults)
        # Read SACC file
        self._read_data()

    def _get_bin_info_extra(self):
        pass

    def _read_data(self):
        """
        Reads sacc file
        Selects relevant data.
        Applies scale cuts
        Reads tracer metadata (N(z))
        Reads covariance
        """

        def get_cl_type(tr1, tr2):
            cltyp = 'cl_'
            for tr in [tr1, tr2]:
                q = tr.quantity
                if (q == 'galaxy_density') or (q == 'cmb_convergence'):
                    cltyp += '0'
                elif q == 'galaxy_shear':
                    cltyp += 'e'
                else:
                    raise ValueError(f'dtype not found for quantity {q}')
            if cltyp == 'cl_e0':  # sacc doesn't like this one
                cltyp = 'cl_0e'
            return cltyp

        def get_lmax_from_kmax(cosmo, kmax, zmid):
            chi = ccl.comoving_radial_distance(cosmo, 1./(1+zmid))
            lmax = np.max([10., kmax * chi - 0.5])
            return lmax

        self.sacc_file = s = sacc.Sacc.load_fits(self.input_file)

        # 1. Iterate through tracers and collect properties
        self.bin_properties = {}
        # We use a default cosmology to map k_max into ell_max
        cosmo_lcdm = ccl.CosmologyVanillaLCDM()
        kmax_default = self.defaults.get('kmax', 0.1)
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            # Default redshift distributions
            if t.quantity in ['galaxy_density', 'galaxy_shear']:
                zmid = np.average(t.z, weights=t.nz)
                self.bin_properties[b['name']] = {'z_fid': t.z,
                                                  'nz_fid': t.nz,
                                                  'zmean_fid': zmid}
            else:
                self.bin_properties[b['name']] = {}

            # Scale cuts
            # Ensure all tracers have ell_min
            if b['name'] not in self.defaults:
                self.defaults[b['name']] = {}
                self.defaults[b['name']]['lmin'] = self.defaults['lmin']

            # Give galaxy clustering an ell_max
            if t.quantity == 'galaxy_density':
                # Get lmax from kmax for galaxy clustering
                if 'lmax' not in self.defaults[b['name']]:
                    if 'kmax' in self.defaults[b['name']]:
                        kmax = self.defaults[b['name']]['kmax']
                    else:
                        kmax = kmax_default
                    lmax = get_lmax_from_kmax(cosmo_lcdm,
                                              kmax, zmid)
                    self.defaults[b['name']]['lmax'] = lmax

                # Do we want magnification bias for this tracer?
                self.bin_properties[b['name']]['mag_bias'] = \
                    self.defaults[b['name']].get("mag_bias", False)

            else:
                # Make sure everything else has an ell_max
                if 'lmax' not in self.defaults[b['name']]:
                    self.defaults[b['name']]['lmax'] = self.defaults['lmax']

        # Additional information specific for this likelihood
        self._get_bin_info_extra()

        # 2. Iterate through two-point functions, apply scale cuts and collect
        # information about them (tracer names, bandpower windows etc.), and
        # put all C_ells in the right order
        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.tracer_qs = {}
        nsides = {b['name']: b.get('nside', None) for b in self.bins}
        for cl in self.twopoints:
            # Get the suffix for both tracers
            tn1, tn2 = cl['bins']
            cltyp = get_cl_type(s.tracers[tn1], s.tracers[tn2])
            # Get data
            l, c_ell, cov, ind = s.get_ell_cl(cltyp, tn1, tn2,
                                              return_cov=True,
                                              return_ind=True)
            # Check it is not empty
            if c_ell.size == 0:
                continue

            # Scale cuts
            if 'lmin' in cl:
                lmin = cl['lmin']
            else:
                lmin = np.max([self.defaults[tn1].get('lmin', 2),
                               self.defaults[tn2].get('lmin', 2)])
            if 'lmax' in cl:
                lmax = cl['lmax']
            else:
                lmax = np.min([self.defaults[tn1].get('lmax', 1E30),
                               self.defaults[tn2].get('lmax', 1E30)])
            sel = (l > lmin) * (l < lmax)
            l = l[sel]
            c_ell = c_ell[sel]
            cov = cov[sel][:, sel]
            ind = ind[sel]

            if tn1 not in self.tracer_qs:
                self.tracer_qs[tn1] = s.tracers[tn1].quantity
            if tn2 not in self.tracer_qs:
                self.tracer_qs[tn2] = s.tracers[tn2].quantity

            bpw = s.get_bandpower_windows(ind)
            l_bpw = bpw.values
            w_bpw = bpw.weight.T

            self.cl_meta.append({'bin_1': tn1,
                                 'bin_2': tn2,
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size,
                                                    dtype=int)),
                                 'l_bpw': l_bpw,
                                 'w_bpw': w_bpw,
                                 'nside_1': nsides[tn1],
                                 'nside_2': nsides[tn2]})
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)
        # Reorder data vector and covariance
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.dense[indices][:, indices]
        # Invert covariance
        self.inv_cov = self.get_inv_cov(self.cov)
        self.ndata = len(self.data_vec)
        # Keep indices in case we want so slice the original sacc file
        self.indices = indices

    def get_inv_cov(self, cov):
        if self.null_negative_cov_eigvals_in_icov:
            evals, evecs = np.linalg.eigh(cov)
            inv_evals = 1/evals
            sel = evals<0
            print(f'Nulling {np.sum(sel)} negative eigenvalues:', evals[sel])
            inv_evals[sel] = 0
            inv_cov = evecs.dot(np.diag(inv_evals).dot(evecs.T))
        else:
            inv_cov = np.linalg.inv(cov)
        return inv_cov

    def _get_jeffrey_bias_dchi2(self):
        g = self.provider.get_cl_theory_deriv()
        ic_g = np.dot(self.inv_cov, g) # (ndata, ndata) , (ndata, nbias)
        F = np.sum(g[:, None, :]*ic_g[:, :, None], axis=0) # (ndata, _, nbias) , (ndata, nbias, _) -> (nbias, nbias)
        return -np.log(np.linalg.det(F))

    def get_requirements(self):
        return {"cl_theory": {"cl_meta": self.cl_meta,
                              "tracer_qs": self.tracer_qs,
                              "bin_properties": self.bin_properties,
                             },
                }

    def _get_chi2(self, **pars):
        t = self.provider.get_cl_theory()
        r = t - self.data_vec
        chi2 = np.dot(r, np.dot(self.inv_cov, r)) # (ndata) , (ndata, ndata) , (ndata)

        # Jeffreys prior for bias?
        dchi2_jeffrey = 0
        if self.jeffrey_bias:
            dchi2_jeffrey = self._get_jeffrey_bias_dchi2()
        return chi2, dchi2_jeffrey

    def get_can_provide_params(self):
        return ['dchi2_jeffrey']

    def calculate(self, state, want_derived=True, **pars):
        # Calculate chi2
        chi2, dchi2_jeffrey = self._get_chi2(**pars)
        state['logp'] = -0.5*(chi2+dchi2_jeffrey)
        state['derived'] = {'dchi2_jeffrey': dchi2_jeffrey}

    def get_cl_theory_sacc(self):
        # Create empty file
        s = sacc.Sacc()

        # Add tracers
        for n, p in self.bin_properties.items():
            if n not in self.tracer_qs:
                continue
            q = self.tracer_qs[n]
            spin = 2 if q == 'galaxy_shear' else 0
            if q in ['galaxy_density', 'galaxy_shear']:
                # TODO: These would better be the shifted Nz
                s.add_tracer('NZ', n, quantity=q, spin=spin,
                             z=p['z_fid'], nz=p['nz_fid'])
            else:
                s.add_tracer('Map', n, quantity=q, spin=spin,
                             ell=np.arange(10), beam=np.ones(10))

        # Calculate power spectra
        cl = self.provider.get_cl_theory()
        for clm in self.cl_meta:
            p1 = 'e' if self.tracer_qs[clm['bin_1']] == 'galaxy_shear' else '0'
            p2 = 'e' if self.tracer_qs[clm['bin_2']] == 'galaxy_shear' else '0'
            cltyp = f'cl_{p1}{p2}'
            if cltyp == 'cl_e0':
                cltyp = 'cl_0e'
            bpw = sacc.BandpowerWindow(clm['l_bpw'], clm['w_bpw'].T)
            s.add_ell_cl(cltyp, clm['bin_1'], clm['bin_2'],
                         clm['l_eff'], cl[clm['inds']], window=bpw)

        return s

    def get_cl_data_sacc(self):
        s = self.sacc_file.copy()
        s.keep_indices(self.indices)

        # Reorder
        indices = []
        for clm in self.cl_meta:
            indices.extend(list(s.indices(tracers=(clm['bin_1'], clm['bin_2']))))
        s.reorder(indices)

        tracers = []
        for trs in s.get_tracer_combinations():
            tracers.extend(trs)

        # Use list(set()) to remove duplicates
        s.keep_tracers(list(set(tracers)))

        return s


class ClLikeFastBias(ClLike):
    bias_params: dict = {}
    # 2nd-order term in bias marginalization?
    bias_laplace: bool = True
    # 2nd-derivative in Fisher term?
    bias_fisher_deriv2: bool = False
    # Update start point every time?
    bias_update_every: bool = False

    def _get_bin_info_extra(self):
        self.bias_names = list(self.bias_params.keys())
        self.bias0 = np.array([v['value'] for k, v in self.bias_params.items()])
        self.bias_pr_mean = np.array([v['prior']['mean'] for k, v in self.bias_params.items()])
        self.bias_pr_isigma2 = np.array([1/v['prior']['sigma']**2 for k, v in self.bias_params.items()])
        self.updated_bias0 = False

    def get_requirements(self):
        return {"bias_info": {},
                "cl_with_bias": {"cl_meta": self.cl_meta,
                                 "tracer_qs": self.tracer_qs,
                                 "bin_properties": self.bin_properties,
                             },
                "cl_theory_dderiv": {},
                }
    
    def calculate(self, state, want_derived=True, **pars):
        # Calculate chi2
        chi2, dchi2, p = self._get_chi2(**pars)

        if self.bias_update_every or (not self.updated_bias0):
            self.bias0 = p.x.copy()
            self.updated_bias0 = True

        if self.bias_laplace:
            state['logp'] = -0.5*(chi2+dchi2)
        else:
            state['logp'] = -0.5*chi2

        # - Best-fit biases
        state['derived'] = dict(zip(self.bias_names, p.x))
        # - Contribution from Laplace marginalization
        state['derived']['dchi2_marg'] = dchi2
        # - Number of function evaluations
        state['derived']['nfev'] = p.nfev

    def _get_chi2(self, **pars):
        bias_names, bias_info = self.provider.get_bias_info()
        assert bias_names == self.bias_names
        global_bias = {name: 1 for name in self.bin_properties.keys()}

        def chi2_f(bias):
            t, g = self.provider.get_cl_with_bias(bias, global_bias)
            r = t - self.data_vec
            ic_r = np.dot(self.inv_cov, r)
            chi2 = np.dot(r, ic_r) # (ndata) , (ndata, ndata) , (ndata)

            dchi2 = 2*np.dot(ic_r, g) # (ndata, ndata) , (ndata) , (ndata, nbias)
            # Bias prior
            rb = bias - self.bias_pr_mean
            chi2 += np.sum(rb**2*self.bias_pr_isigma2)
            dchi2 += 2*rb*self.bias_pr_isigma2
            return chi2, dchi2

        def hessian_chi2_f(bias, include_DF=False):
            t, g = self.provider.get_cl_with_bias(bias, global_bias)
            ic_g = np.dot(self.inv_cov, g) # (ndata, ndata) , (ndata, nbias)
            ddchi2 = 2*np.sum(g[:, None, :]*ic_g[:, :, None], axis=0) # (ndata, _, nbias) , (ndata, nbias, _) -> (nbias, nbias)
            # Bias prior
            ddchi2 += 2*np.diag(self.bias_pr_isigma2)
            if include_DF:
                ddt = self.provider.get_cl_theory_dderiv(bias, global_bias)
                r = t-self.data_vec
                ic_r = np.dot(self.inv_cov, r)
                ddchi2 += 2*np.sum(ic_r[:, None, None]*ddt, axis=0) # (ndata), (ndata, nbias, nbias)
            return ddchi2

        p = minimize(chi2_f, self.bias0, method='Newton-CG', jac=True,
                     hess=lambda b: hessian_chi2_f(b))
        chi2 = p.fun

        H = hessian_chi2_f(p.x, include_DF=self.bias_fisher_deriv2)
        ev = np.linalg.eigvals(0.5*H)
        if np.any(ev <= 0):  # Use positive-definite version if needed
            H = hessian_chi2_f(p.x, include_DF=False)
            ev = np.linalg.eigvals(0.5*H)
        dchi2 = np.sum(np.log(ev))
        if np.isnan(dchi2):
            print(dchi2)
            print(ev)
            print(H)
            exit(1)

        return chi2, dchi2, p

    def get_cl_theory_sacc(self,bias,global_bias):
        # Create empty file
        s = sacc.Sacc()

        # Add tracers
        for n, p in self.bin_properties.items():
            if n not in self.tracer_qs:
                continue
            q = self.tracer_qs[n]
            spin = 2 if q == 'galaxy_shear' else 0
            if q in ['galaxy_density', 'galaxy_shear']:
                # TODO: These would better be the shifted Nz
                s.add_tracer('NZ', n, quantity=q, spin=spin,
                             z=p['z_fid'], nz=p['nz_fid'])
            else:
                s.add_tracer('Map', n, quantity=q, spin=spin,
                             ell=np.arange(10), beam=np.ones(10))

        # Calculate power spectra
        cl, dcl = self.provider.get_cl_with_bias(bias, global_bias)#self.provider.get_cl_theory()
        for clm in self.cl_meta:
            p1 = 'e' if self.tracer_qs[clm['bin_1']] == 'galaxy_shear' else '0'
            p2 = 'e' if self.tracer_qs[clm['bin_2']] == 'galaxy_shear' else '0'
            cltyp = f'cl_{p1}{p2}'
            if cltyp == 'cl_e0':
                cltyp = 'cl_0e'
            bpw = sacc.BandpowerWindow(clm['l_bpw'], clm['w_bpw'].T)
            s.add_ell_cl(cltyp, clm['bin_1'], clm['bin_2'],
                         clm['l_eff'], cl[clm['inds']], window=bpw)
        return s
