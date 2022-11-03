import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.nl_pt as pt
from .hm_extra import HalomodCorrection
from .pixwin import beam_hpix
from .lpt import LPTCalculator, get_lpt_pk2d
from .ept import EPTCalculator, get_ept_pk2d
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


class ClLike(Likelihood):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
    # Input sacc file
    input_file: str = ""
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"
    # N(z) model name
    nz_model: str = "NzNone"
    # b(z) model name
    bias_model: str = "BzNone"
    # zmax for 3D power spectra
    zmax_pks: float = 4.
    # #z for 3D power spectra
    nz_pks: int = 30
    # #k for 3D power spectra
    nk_per_dex_pks: int = 25
    # min k 3D power spectra
    l10k_min_pks: float = -4.0
    # max k 3D power spectra
    l10k_max_pks: float = 2.0
    # Angular resolution
    nside: int = -1
    # List of bin names
    bins: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # List of two-point functions that make up the data vector
    twopoints: list = []

    def initialize(self):
        # Read SACC file
        self._read_data()
        # Ell sampling for interpolation
        self._get_ell_sampling()
        # Other global parameters
        self._init_globals()

    def _init_globals(self):
        # We will need this to map parameters into tracers
        self.qabbr = {'galaxy_density': 'g',
                      'galaxy_shear': 'm',
                      'cmb_convergence': 'm'}

        # Pk sampling
        self.a_s_pks = 1./(1+np.linspace(0., self.zmax_pks, self.nz_pks)[::-1])
        self.nk_pks = int((self.l10k_max_pks - self.l10k_min_pks) *
                          self.nk_per_dex_pks)

        # Cl sampling
        if self.sample_type == 'best':
            import yaml

            # Read parameters
            with open(self.defaults['params_fid'], "r") as fpar:
                info = yaml.load(fpar, Loader=yaml.FullLoader)
            p0 = {}
            for p in info['params']:
                if isinstance(info['params'][p], dict):
                    if 'ref' in info['params'][p]:
                        p0[p] = info['params'][p]['ref']['loc']
            # Get best ells
            cosmo = ccl.CosmologyVanillaLCDM()
            self.get_best_ells(cosmo, **p0)

        # Pixel window function product for each power spectrum
        nsides = {b['name']: b.get('nside', None)
                  for b in self.bins}
        for clm in self.cl_meta:
            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample
            beam = np.ones(ls.size)
            for n in [clm['bin_1'], clm['bin_2']]:
                if nsides[n]:
                    beam *= beam_hpix(ls, nsides[n])
            clm['pixbeam'] = beam

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
                                 'w_bpw': w_bpw})
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)
        # Reorder data vector and covariance
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.dense[indices][:, indices]
        # Invert covariance
        self.inv_cov = np.linalg.inv(self.cov)
        self.ndata = len(self.data_vec)

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

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(np.log(1E-3+self.l_sample), cl_in)
        cl_unbinned = f(np.log(1E-3+l_bpw))
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, cosmo, name, **pars):
        """ Get redshift distribution for a given tracer.
        Applies shift and width nuisance parameters if needed.
        """
        z = self.bin_properties[name]['z_fid']
        nz = self.bin_properties[name]['nz_fid']
        zm = self.bin_properties[name]['zmean_fid']
        dz = 0.
        wz = 1.
        if (self.nz_model == 'NzShift') or (self.nz_model == 'NzShiftWidth'):
            dz = pars.get(self.input_params_prefix + '_' + name + '_dz', 0.)
        if (self.nz_model == 'NzShiftWidth') or (self.nz_model == 'NzWidth'):
            wz = pars.get(self.input_params_prefix + '_' + name + '_wz', 1.)
        z = zm+dz+(z-zm)/wz
        msk = z >= 0
        z = z[msk]
        nz = nz[msk]
        return (z, nz)

    def _get_bz(self, cosmo, name, **pars):
        """ Get linear galaxy bias. Unless we're using a linear bias,
        model this should be just 1."""
        z = self.bin_properties[name]['z_fid']
        zmean = self.bin_properties[name]['zmean_fid']
        b0 = pars[self.input_params_prefix + '_' + name + '_b1']
        bz = b0*np.ones_like(z)
        return (z, bz)

    def _get_ia_bias(self, cosmo, name, **pars):
        """ Intrinsic alignment amplitude.
        """
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            A = pars[self.input_params_prefix + '_' + name + '_A_IA']
            A_IA = np.ones_like(z) * A
            return (z, A_IA)

    def _get_tracers(self, cosmo, **pars):
        """ Obtains CCL tracers (and perturbation theory tracers,
        and halo profiles where needed) for all used tracers given the
        current parameters."""
        trs0 = {}
        trs1 = {}
        for name, q in self.tracer_qs.items():
            prefix = self.input_params_prefix + '_' + name

            if q == 'galaxy_density':
                nz = self._get_nz(cosmo, name, **pars)
                bz = self._get_bz(cosmo, name, **pars)
                t0 = None
                t1 = [ccl.NumberCountsTracer(cosmo, dndz=nz,
                                             bias=bz, has_rsd=False)]
            elif q == 'galaxy_shear':
                nz = self._get_nz(cosmo, name, **pars)
                ia = self._get_ia_bias(cosmo, name, **pars)
                t0 = ccl.WeakLensingTracer(cosmo, nz)
                t1 = [ccl.WeakLensingTracer(cosmo, nz, has_shear=False, ia_bias=ia)]
            elif q == 'cmb_convergence':
                # B.H. TODO: pass z_source as parameter to the YAML file
                t0 = ccl.CMBLensingTracer(cosmo, z_source=1100)
                t1 = None

            trs0[name] = t0
            trs1[name] = t1
        return trs0, trs1

    def _get_pk_data(self, cosmo):
        """ Get all cosmology-dependent ingredients to create the
        different P(k)s needed for the C_ell calculation.
        For linear bias, this is just the matter power spectrum.
        """
        # Get P(k)s from CCL
        if self.bias_model == 'Linear':
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            return {'pk_mm': pkmm}
        else:
            raise NotImplementedError(f"Bias model {self.bias_model} not implemented")

    def _get_pkxy(self, cosmo, clm, pkd, trs, **pars):
        """ Get the P(k) between two tracers. """
        q1 = self.tracer_qs[clm['bin_1']]
        q2 = self.tracer_qs[clm['bin_2']]

        return pkd['pk_mm']

    def _get_cl_data(self, cosmo, **pars):
        """ Compute all C_ells."""
        # Gather all tracers
        trs0, trs1 = self._get_tracers(cosmo, **pars)

        # Correlate all needed pairs of tracers
        cls_00 = []
        cls_01 = []
        cls_10 = []
        cls_11 = []
        for clm in self.cl_meta:
            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample

            n1 = clm['bin_1']
            n2 = clm['bin_2']
            t0_1 = trs0[n1]
            t0_2 = trs0[n2]
            t1_1 = trs1[n1]
            t1_2 = trs1[n2]
            # 00: unbiased x unbiased
            if t0_1 and t0_2:
                cl00 = ccl.angular_cl(cosmo, t0_1, t0_2, ls) * clm['pixbeam']
                cls_00.append(cl00)
            else:
                cls_00.append(None)
            # 01: unbiased x biased
            if t0_1 and t1_2:
                cl01 = []
                for t12 in t1_2:
                    cl = ccl.angular_cl(cosmo, t0_1, t12, ls) * clm['pixbeam']
                    cl01.append(cl)
                cl01 = np.array(cl01)
            else:
                cl01 = None
            cls_01.append(cl01)
            # 10: biased x unbiased
            if n1 == n2:
                cls_10.append(cl01)
            else:
                if t0_2 and t1_1:
                    cl10 = []
                    for t11 in t1_1:
                        cl = ccl.angular_cl(cosmo, t11, t0_2, ls) * clm['pixbeam']
                        cl10.append(cl)
                    cl10 = np.array(cl10)
                else:
                    cl10 = None
                cls_10.append(cl10)
            # 11: biased x biased
            if t1_1 and t1_2:
                cl11 = np.zeros([len(t1_1), len(t1_2), len(ls)])
                autocorr = n1 == n2
                for i1, t11 in enumerate(t1_1):
                    for i2, t12 in enumerate(t1_2):
                        if autocorr and i2 < i1:
                            cl11[i1, i2] = cl11[i2, i1]
                        else:
                            cl = ccl.angular_cl(cosmo, t11, t12, ls) * clm['pixbeam']
                            cl11[i1, i2, :] = cl
            else:
                cl11 = None
            cls_11.append(cl11)

        # Bandpower window convolution
        if self.sample_cen:
            clbs_00 = cls_00
            clbs_01 = cls_01
            clbs_10 = cls_10
            clbs_11 = cls_11
        elif self.sample_bpw:
            clbs_00 = []
            clbs_01 = []
            clbs_10 = []
            clbs_11 = []
            # 00: unbiased x unbiased
            for clm, cl00 in zip(self.cl_meta, cls_00):
                if cl00:
                    clb00 = self._eval_interp_cl(cl00, clm['l_bpw'], clm['w_bpw'])
                else: 
                    clb00 = None
                clbs_00.append(clb00)
            for clm, cl01, cl10 in zip(self.cl_meta, cls_01, cls_10):
                # 01: unbiased x biased
                if cl01:
                    clb01 = []
                    for cl in cl01:
                        clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                        clb01.append(clb)
                    clb01 = np.array(clb01)
                else: 
                    clb01 = None
                clbs_01.append(clb01)
                # 10: biased x unbiased
                if clm['bin1'] == clm['bin2']:
                    clbs_10.append(clb01)
                else:
                    if cl10:
                        clb10 = []
                        for cl in cl10:
                            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                            clb10.append(clb)
                        clb10 = np.array(clb10)
                    else: 
                        clb10 = None
                clbs_10.append(clb10)
                # 11: biased x biased
                for clm, cl11 in zip(self.cl_meta, cls_11):
                    if cl11:
                        clb11 = np.zeros_like(cl11)
                        autocorr = clm['bin_1'] == clm['bin_2']
                        for i1 in range(np.shape(cl11)[0]):
                            for i2 in range(np.shape(cl11)[1]):
                                if autocorr and i2 < i1:
                                    clb11[i1, i2] = clb11[i2, i1]
                                else:
                                    cl = cl11[i1,i2,:]
                                    clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                                    clb11[i1,i2,:] = clb 
                    else: 
                        clb11 = None
                clbs_11.append(clb11)
                
        return clbs_00, clbs_01, clbs_10, clbs_11

    
    def get_cls_theory(self, **pars):
        # Get cosmological model
        res = self.provider.get_CCL()
        cosmo = res['cosmo']

        # First, gather all the necessary ingredients for the different P(k)
        pkd = res['pk_data']

        # Then pass them on to convert them into C_ells
        cls = self._get_cl_all(cosmo, pkd, **pars)

        return cls

    def get_best_ells(self, cosmo, **pars):
        nsides = {b['name']: b.get('nside', None)
                  for b in self.bins}
        pkd = self._get_pk_data(cosmo)
        trs = self._get_tracers(cosmo, **pars)

        # Compute all C_ells
        cls = []
        for clm in self.cl_meta:
            pkxy = self._get_pkxy(cosmo, clm, pkd, trs, **pars)
            ls = self.l_sample
            cl = ccl.angular_cl(cosmo,
                                trs[clm['bin_1']]['ccl_tracer'],
                                trs[clm['bin_2']]['ccl_tracer'],
                                ls, p_of_k_a=pkxy)
            # Pixel window function
            beam = np.ones(ls.size)
            for n in [clm['bin_1'], clm['bin_2']]:
                if nsides[n]:
                    beam *= beam_hpix(ls, nsides[n])
            cl *= beam
            f = interp1d(np.log(1E-3+self.l_sample), cl)
            cls.append(f(np.log(1E-3+clm['l_bpw'])))

        # Bandpower window convolution
        clbs = []
        for clm, cl in zip(self.cl_meta, cls):
            clb = np.dot(clm['w_bpw'], cl)
            clbs.append(clb)

        # Find best ells and replace 'l_eff'
        for clm, cl, clb in zip(self.cl_meta, cls, clbs):
            lbfs = []
            for i, ll in enumerate(clm['l_eff']):
                l0 = ll*0.5
                l1 = ll*1.5
                ids = (clm['l_bpw'] >= l0) & (clm['l_bpw'] <= l1)
                ibf = np.argmin(np.fabs(clb[i]-cl[ids]))
                lbf = clm['l_bpw'][ids][ibf]
                lbfs.append(lbf)
            lbfs = np.array(lbfs)
            if np.any(np.fabs(lbfs/clm['l_eff']-1) > 0.2):
                print(clm['bin_1'], clm['bin_1'])
                print(lbfs)
                print(clm['l_eff'])
                raise ValueError("There might be something wrong with "
                                 "the effective ell assignment.")
            clm['l_eff'] = lbfs

    def get_sacc_file(self, **pars):
        import sacc

        # Create empty file
        s = sacc.Sacc()

        # Add tracers
        for n, p in self.bin_properties.items():
            if n not in self.tracer_qs:
                continue
            q = self.tracer_qs[n]
            spin = 2 if q == 'galaxy_shear' else 0
            if q in ['galaxy_density', 'galaxy_shear']:
                s.add_tracer('NZ', n, quantity=q, spin=spin,
                             z=p['z_fid'], nz=p['nz_fid'])
            else:
                s.add_tracer('Map', n, quantity=q, spin=spin,
                             ell=np.arange(10), beam=np.ones(10))

        # Calculate power spectra
        cls = self.get_cls_theory(**pars)
        for clm, cl in zip(self.cl_meta, cls):
            p1 = 'e' if self.tracer_qs[clm['bin_1']] == 'galaxy_shear' else '0'
            p2 = 'e' if self.tracer_qs[clm['bin_2']] == 'galaxy_shear' else '0'
            cltyp = f'cl_{p1}{p2}'
            if cltyp == 'cl_e0':
                cltyp = 'cl_0e'
            bpw = sacc.BandpowerWindow(clm['l_bpw'], clm['w_bpw'].T)
            s.add_ell_cl(cltyp, clm['bin_1'], clm['bin_2'],
                         clm['l_eff'], cl, window=bpw)

        s.add_covariance(self.cov)
        return s

    def _get_theory(self, **pars):
        """ Computes theory vector."""
        cls = self.get_cls_theory(**pars)

        # Flattening into a 1D array
        cl_out = np.zeros(self.ndata)
        for clm, cl in zip(self.cl_meta, cls):
            cl_out[clm['inds']] = cl

        return cl_out

    def get_requirements(self):
        # By selecting `self._get_cl_data` as a `method` of CCL here,
        # we make sure that this function is only run when the
        # cosmological parameters vary.
        return {'CCL': {'methods': {'cl_data': self._get_cl_data}}}

    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, np.dot(self.inv_cov, r))
        return -0.5*chi2
