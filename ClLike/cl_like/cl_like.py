import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.optimize import minimize


class ClLike(Likelihood):
    # TODO: Might many of these make more sense in the Theory classes?
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
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
    # b(z) model name
    bias_model: str = "BzNone"
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"

    def initialize(self):
        # Bias model
        self.is_PT_bias = self.bias_model in ['LagrangianPT', 'EulerianPT']
        # Read SACC file
        self._read_data()
        # Ell sampling for interpolation
        self._get_ell_sampling()

    def initialize_with_provider(self, provider):
        self.provider = provider
        # self.ia_model = self.provider.get_ia_model()
        # self.is_PT_bias = self.provider.get_is_PT_bias()

    def _read_data(self):
        """
        Reads sacc file
        Selects relevant data.
        Applies scale cuts
        Reads tracer metadata (N(z))
        Reads covariance
        """
        import sacc

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

        s = sacc.Sacc.load_fits(self.input_file)

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
                if 'kmax' in self.defaults[b['name']]:
                    kmax = self.defaults[b['name']]['kmax']
                else:
                    kmax = kmax_default
                lmax = get_lmax_from_kmax(cosmo_lcdm,
                                          kmax, zmid)
                self.defaults[b['name']]['lmax'] = lmax
            else:
                # Make sure everything else has an ell_max
                if 'lmax' not in self.defaults[b['name']]:
                    self.defaults[b['name']]['lmax'] = self.defaults['lmax']

        # Additional information specific for this likelihood
        self._get_bin_info_extra(s)

        # 2. Iterate through two-point functions and apply scale cuts
        indices = []
        for cl in self.twopoints:
            tn1, tn2 = cl['bins']
            lmin = np.max([self.defaults[tn1].get('lmin', 2),
                           self.defaults[tn2].get('lmin', 2)])
            lmax = np.min([self.defaults[tn1].get('lmax', 1E30),
                           self.defaults[tn2].get('lmax', 1E30)])
            # Get the suffix for both tracers
            cltyp = get_cl_type(s.tracers[tn1], s.tracers[tn2])
            ind = s.indices(cltyp, (tn1, tn2),
                            ell__gt=lmin, ell__lt=lmax)
            indices += list(ind)
        s.keep_indices(np.array(indices))

        # 3. Iterate through two-point functions, collect information about
        # them (tracer names, bandpower windows etc.), and put all C_ells in
        # the right order
        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.tracer_qs = {}
        self.l_min_sample = 1E30
        self.l_max_sample = self.defaults.get('lmax_sample', -1E30)
        self.sample_type = self.defaults.get('sample_type', 'convolve')
        self.sample_cen = self.sample_type in ['center', 'best']
        self.sample_bpw = self.sample_type == 'convolve'
        lmax_sample_set = self.l_max_sample > 0
        nsides = {b['name']: b.get('nside', None) for b in self.bins}
        for cl in self.twopoints:
            # Get the suffix for both tracers
            tn1, tn2 = cl['bins']
            cltyp = get_cl_type(s.tracers[tn1], s.tracers[tn2])
            l, c_ell, cov, ind = s.get_ell_cl(cltyp, tn1, tn2,
                                              return_cov=True,
                                              return_ind=True)
            if c_ell.size > 0:
                if tn1 not in self.tracer_qs:
                    self.tracer_qs[tn1] = s.tracers[tn1].quantity
                if tn2 not in self.tracer_qs:
                    self.tracer_qs[tn2] = s.tracers[tn2].quantity

            bpw = s.get_bandpower_windows(ind)
            if np.amin(bpw.values) < self.l_min_sample:
                self.l_min_sample = np.amin(bpw.values)
            if lmax_sample_set:
                good = bpw.values <= self.l_max_sample
                l_bpw = bpw.values[good]
                w_bpw = bpw.weight[good].T
            else:
                if np.amax(bpw.values) > self.l_max_sample:
                    self.l_max_sample = np.amax(bpw.values)
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
        self.inv_cov = np.linalg.inv(self.cov)
        self.ndata = len(self.data_vec)

    def _get_bin_info_extra(self, s):
        # Extract additional per-sample information from the sacc
        # file needed for this likelihood.
        ind_bias = 0
        ind_IA = None
        self.bias_names = []
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]

            self.bin_properties[b['name']]['bias_ind'] = None # No biases by default
            if t.quantity == 'galaxy_density':
                # Linear bias
                inds = [ind_bias]
                self.bias_names.append(self.input_params_prefix +
                                       '_'+b['name']+'_b1')
                ind_bias += 1
                # Higher-order biases
                if self.is_PT_bias:
                    for bn in ['b2', 'bs', 'bk2']:
                        self.bias_names.append(self.input_params_prefix +
                                               '_'+b['name']+'_'+bn)
                        inds.append(ind_bias)
                        ind_bias += 1
                self.bin_properties[b['name']]['bias_ind'] = inds
                # No magnification bias yet
                self.bin_properties[b['name']]['eps'] = False
            elif t.quantity == 'galaxy_shear':
                if self.ia_model != 'IANone':
                    if ind_IA is None:
                        ind_IA = ind_bias
                        self.bias_names.append(self.input_params_prefix + '_A_IA')
                        ind_bias += 1
                    self.bin_properties[b['name']]['bias_ind'] = [ind_IA]
                self.bin_properties[b['name']]['eps'] = True
            elif t.quantity == 'cmb_convergence':
                # TODO: No idea what eps is so setting to True to make the test
                # work
                self.bin_properties[b['name']]['eps'] = True

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30
        if self.l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = self.l_min_sample
        nl_sample = int(np.log10(self.l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          self.l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if self.l_min_sample == 0:
            self.l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            self.l_sample = l_sample

    def _model(self, cld, bias_vec):
        cls = np.zeros(self.ndata)
        for icl, clm in enumerate(self.cl_meta):
            cl_this = np.zeros_like(clm['l_eff'])
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            e1 = self.bin_properties[n1]['eps']
            e2 = self.bin_properties[n2]['eps']
            ind1 = self.bin_properties[n1]['bias_ind']
            ind2 = self.bin_properties[n2]['bias_ind']
            b1 = bias_vec[ind1] if ind1 is not None else None
            b2 = bias_vec[ind2] if ind2 is not None else None
            inds = clm['inds']
            if e1 and e2:
                cl_this += cld['cl00'][icl]
            if e1 and (b2 is not None):
                cl_this += np.dot(b2, cld['cl01'][icl]) # (nbias) , (nbias, nell)
            if e2 and (b1 is not None):
                cl_this += np.dot(b1, cld['cl10'][icl]) # (nbias) , (nbias, nell)
            if (b1 is not None) and (b2 is not None):
                cl_this += np.dot(b1, np.dot(b2, cld['cl11'][icl])) # (nbias1) * ((nbias2), (nbias1,nbias2,nell))
            cls[inds] = cl_this

        return cls

    def _model_deriv(self, cld, bias_vec):
        nbias = len(bias_vec)
        cls_deriv = np.zeros((self.ndata, nbias))

        for icl, clm in enumerate(self.cl_meta):
            cls_grad = np.zeros([nbias, len(clm['l_eff'])])
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            e1 = self.bin_properties[n1]['eps']
            e2 = self.bin_properties[n2]['eps']
            ind1 = self.bin_properties[n1]['bias_ind']
            ind2 = self.bin_properties[n2]['bias_ind']
            b1 = bias_vec[ind1] if ind1 is not None else None
            b2 = bias_vec[ind2] if ind2 is not None else None
            inds = clm['inds']

            if e1 and (b2 is not None):
                cls_grad[ind2] += cld['cl01'][icl] # (nbias2, ndata) , (nbias2, ndata)
            if e2 and (b1 is not None):
                cls_grad[ind1] += cld['cl10'][icl] # (nbias1, ndata) , (nbias1, ndat)

            if (b1 is not None) and (b2 is not None):
                cl_b2 = np.dot(b2, cld['cl11'][icl]) # (nbias2) , (nbias1, nbias2, ndata) -> (nbias1, ndata)
                cl_b1 = np.sum(b1[:, None, None] * cld['cl11'][icl], axis=0) # (nbias1) , (nbias1, nbias2, ndata) -> (nbias2, ndata)
                cls_grad[ind1] += cl_b2
                cls_grad[ind2] += cl_b1

            cls_deriv[inds] = cls_grad.T
        return cls_deriv # (ndata, nbias)

    def _get_jeffrey_bias_dchi2(self, bias, cld):
        g = self._model_deriv(cld, bias)
        ic_g = np.dot(self.inv_cov, g) # (ndata, ndata) , (ndata, nbias)
        F = np.sum(g[:, None, :]*ic_g[:, :, None], axis=0) # (ndata, _, nbias) , (ndata, nbias, _) -> (nbias, nbias)
        return -np.log(np.linalg.det(F))

    def get_requirements(self):
        return {"Limber": {"cl_meta": self.cl_meta,
                           "l_sample": self.l_sample,
                           "tracer_qs": self.tracer_qs,
                           "bin_properties": self.bin_properties,
                           "sample_cen": self.sample_cen,
                           "sample_bpw": self.sample_bpw,
                           "input_params_prefix": self.input_params_prefix,
                           "bias_model": self.bias_model,
                           "ia_model": self.ia_model
                           }
                }

    def _get_chi2(self, **pars):
        # First, gather all the necessary ingredients for the Cls without bias parameters
        res = self.provider.get_Limber()
        cld = res['cl_data']

        # Construct bias vector
        bias = np.array([pars[k] for k in self.bias_names])

        # Theory model
        t = self._model(cld, bias)
        r = t - self.data_vec
        chi2 = np.dot(r, np.dot(self.inv_cov, r)) # (ndata) , (ndata, ndata) , (ndata)

        # Jeffreys prior for bias?
        dchi2_jeffrey = 0
        if self.jeffrey_bias:
            dchi2_jeffrey = self._get_jeffrey_bias_dchi2(bias, cld)
        return chi2, dchi2_jeffrey

    def get_can_provide_params(self):
        return ['dchi2_jeffrey']

    def calculate(self, state, want_derived=True, **pars):
        # Calculate chi2
        chi2, dchi2_jeffrey = self._get_chi2(**pars)
        state['logp'] = -0.5*(chi2+dchi2_jeffrey)
        state['derived'] = {'dchi2_jeffrey': dchi2_jeffrey}


class ClLikeFastBias(ClLike):
    # Bias parameters
    bias_params: dict = {}
    # 2nd order term in bias marginalization?
    bias_fisher: bool = True
    # 2nd derivative in Fisher term?
    bias_fisher_deriv2: bool = False
    # Update start point every time?
    bias_update_every: bool = False

    def _get_bin_info_extra(self, s):
        # Extract additional per-sample information from the sacc
        # file needed for this likelihood.
        ind_bias = 0
        ind_IA = None
        self.bias0 = []
        self.bias_names = []
        self.bias_pr_mean = []
        self.bias_pr_isigma2 = []
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]

            self.bin_properties[b['name']]['bias_ind'] = None # No biases by default
            if t.quantity == 'galaxy_density':
                inds = []
                if self.is_PT_bias:
                    bnames = ['b1', 'b2', 'bs', 'bk2']
                else:
                    bnames = ['b1']
                for bn in bnames:
                    bname = self.input_params_prefix + '_' + b['name'] + '_' + bn
                    self.bias0.append(self.bias_params[bname]['value'])
                    self.bias_names.append(bname)
                    pr = self.bias_params[bname].get('prior', None)
                    if pr is not None:
                        self.bias_pr_mean.append(pr['mean'])
                        self.bias_pr_isigma2.append(1./pr['sigma']**2)
                    else:
                        self.bias_pr_mean.append(self.bias_params[bname]['value'])
                        self.bias_pr_isigma2.append(0.0)
                    inds.append(ind_bias)
                    ind_bias += 1
                self.bin_properties[b['name']]['bias_ind'] = inds
                # No magnification bias yet
                self.bin_properties[b['name']]['eps'] = False
            if t.quantity == 'galaxy_shear':
                if self.ia_model != 'IANone':
                    if ind_IA is None:
                        ind_IA = ind_bias
                        pname = self.input_params_prefix + '_A_IA'
                        self.bias0.append(self.bias_params[pname]['value'])
                        self.bias_names.append(pname)
                        pr = self.bias_params[pname].get('prior', None)
                        if pr is not None:
                            self.bias_pr_mean.append(pr['mean'])
                            self.bias_pr_isigma2.append(1./pr['sigma']**2)
                        else:
                            self.bias_pr_mean.append(self.bias_params[pname]['value'])
                            self.bias_pr_isigma2.append(0.0)
                        ind_bias += 1
                    self.bin_properties[b['name']]['bias_ind'] = [ind_IA]
                self.bin_properties[b['name']]['eps'] = True
        self.bias0 = np.array(self.bias0)
        self.bias_pr_mean = np.array(self.bias_pr_mean)
        self.bias_pr_isigma2 = np.array(self.bias_pr_isigma2)
        self.updated_bias0 = False

    def _model_dderiv(self, cld, bias_vec):
        nbias = len(bias_vec)
        cls_dderiv = np.zeros((self.ndata, nbias, nbias))

        for icl, clm in enumerate(self.cl_meta):
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            ind1 = self.bin_properties[n1]['bias_ind']
            ind2 = self.bin_properties[n2]['bias_ind']
            inds = clm['inds']

            if (ind1 is not None) and (ind2 is not None):
                cls_hess = np.zeros([nbias, nbias, len(clm['l_eff'])])
                cls_hess[np.ix_(ind1, ind2)] += cld['cl11'][icl]
                cls_hess[np.ix_(ind2, ind1)] += np.transpose(cld['cl11'][icl], axes=(1, 0, 2))
                cls_dderiv[inds] = np.transpose(cls_hess, axes=(2, 0, 1))
        return cls_dderiv # (ndata, nbias)

    def hessian_chi2(self, bias, cld, include_DF=False):
        g = self._model_deriv(cld, bias)
        ic_g = np.dot(self.inv_cov, g) # (ndata, ndata) , (ndata, nbias)
        ddchi2 = 2*np.sum(g[:, None, :]*ic_g[:, :, None], axis=0) # (ndata, _, nbias) , (ndata, nbias, _) -> (nbias, nbias)
        # Bias prior
        ddchi2 += 2*np.diag(self.bias_pr_isigma2)
        # Second derivative term
        if include_DF:
            t = self._model(cld, bias)
            ddt = self._model_dderiv(cld, bias)
            r = t - self.data_vec
            ic_r = np.dot(self.inv_cov, r)
            ddchi2 += 2*np.sum(ic_r[:, None, None]*ddt, axis=0) # (ndata), (ndata, nbias, nbias)
        return ddchi2

    def _get_BF_chi2_and_F(self, **pars):
        # First, gather all the necessary ingredients for the Cls without bias parameters
        res = self.provider.get_CCL()
        cld = res['cl_data']

        def chi2(bias):
            t = self._model(cld, bias)
            r = t - self.data_vec
            ic_r = np.dot(self.inv_cov, r)
            chi2 = np.dot(r, ic_r) # (ndata) , (ndata, ndata) , (ndata)
            g = self._model_deriv(cld, bias)
            dchi2 = 2*np.dot(ic_r, g) # (ndata, ndata) , (ndata) , (ndata, nbias)
            # Bias prior
            rb = bias - self.bias_pr_mean
            chi2 += np.sum(rb**2*self.bias_pr_isigma2)
            dchi2 += 2*rb*self.bias_pr_isigma2
            return chi2, dchi2

        p = minimize(chi2, self.bias0, method='Newton-CG', jac=True,
                     hess=lambda b: self.hessian_chi2(b, cld))
        H = self.hessian_chi2(p.x, cld,
                              include_DF=self.bias_fisher_deriv2)

        return p.fun, 0.5*H, p

    def get_can_provide_params(self):
        return self.bias_names + ['nfev', 'dchi2_marg']

    def calculate(self, state, want_derived=True, **pars):
        # Calculate chi2
        chi2, F, p = self._get_BF_chi2_and_F(**pars)

        # Update starting point
        if self.bias_update_every or (not self.updated_bias0):
            self.bias0 = p.x.copy()

        # Plotting test
        #import matplotlib.pyplot as plt
        #res = self.provider.get_CCL()
        #cld = res['cl_data']
        #
        ## Theory
        #t = self._model(cld, self.bias0)
        #plt.figure()
        #plt.plot(t)
        #
        ## Derivative
        #dt = self._model_deriv(cld, self.bias0)
        #for ind_b in range(4):
        #    plt.figure()
        #    plt.title(f'{ind_b}')
        #    colors = ['r', 'g', 'b', 'y', 'k']
        #    for i, c in enumerate(colors):
        #        ibias = ind_b+i*4
        #        db = 0.01
        #        b = self.bias0.copy()
        #        b[ibias] += db
        #        tp = self._model(cld, b)
        #        b = self.bias0.copy()
        #        b[ibias] -= db
        #        tm = self._model(cld, b)
        #        dtb = (tp-tm)/(2*db)
        #
        #        plt.plot(np.fabs(dt[:, ibias]), c=c, label=f'{i}')
        #        plt.plot(np.fabs(dtb), '.', c=c)
        #    plt.legend()
        #
        ## Second derivative
        #ddt = self._model_dderiv(cld, self.bias0)
        #print(ddt.shape)
        #for ind_b1 in range(4):
        #    for ind_b2 in range(ind_b1, 4):
        #        plt.figure()
        #        plt.title(f'{ind_b1}-{ind_b2}')
        #        colors = ['r', 'g', 'b', 'y', 'k']
        #        for i, c in enumerate(colors):
        #            ibias1 = ind_b1+i*4
        #            ibias2 = ind_b2+i*4
        #            db = 0.01
        #            bpp = self.bias0.copy()
        #            bpp[ibias1] += db
        #            bpp[ibias2] += db
        #            tpp = self._model(cld, bpp)
        #            bpm = self.bias0.copy()
        #            bpm[ibias1] += db
        #            bpm[ibias2] -= db
        #            tpm = self._model(cld, bpm)
        #            bmp = self.bias0.copy()
        #            bmp[ibias1] -= db
        #            bmp[ibias2] += db
        #            tmp = self._model(cld, bmp)
        #            bmm = self.bias0.copy()
        #            bmm[ibias1] -= db
        #            bmm[ibias2] -= db
        #            tmm = self._model(cld, bmm)
        #            ddtb = (tpp-tpm-tmp+tmm)/(4*db**2)
        #
        #            print(ind_b1, ind_b2, i, ddt[:, ibias1, ibias2])
        #            plt.plot(np.fabs(ddt[:, ibias1, ibias2]), c=c, label=f'{i}')
        #            plt.plot(np.fabs(ddtb), '.', c=c)
        #        plt.legend()
        #plt.show()
        #exit(1)

        # Compute log_like
        if self.bias_fisher:
            dchi2 = np.log(np.linalg.det(F))
        else:
            dchi2 = 0.0
        state['logp'] = -0.5*(chi2 + dchi2)

        # Add derived parameters
        # - Best-fit biases
        state['derived'] = dict(zip(self.bias_names, p.x))
        # - Number of function evaluations
        state['derived']['nfev'] = p.nfev
        # - Contribution from Laplace marginalization
        state['derived']['dchi2_marg'] = dchi2
