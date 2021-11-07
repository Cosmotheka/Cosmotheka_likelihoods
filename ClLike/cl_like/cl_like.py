import time
import os

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

from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT

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
    # Baryon model
    baryon_model: str = "BNone"
    # Shape systamatics
    shape_model: str = "ShapeNone"
    # Mass function name
    mf_name: str = "Tinker08"
    # Halo bias name
    hb_name: str = "Tinker10"
    # Concentration name
    cm_name: str = "Duffy08M500c"
    # Mass definition
    mass_def_str: str = "500c"
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
    # k shot noise suppression scale
    k_SN_suppress: float = 0.01
    # HM correction
    HM_correction: str = "HMCode"
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
                      'cmb_convergence': 'm',
                      'cmb_tSZ': 'y'}

        # Pk sampling
        self.a_s_pks = 1./(1+np.linspace(0., self.zmax_pks, self.nz_pks)[::-1])
        self.nk_pks = int((self.l10k_max_pks - self.l10k_min_pks) *
                          self.nk_per_dex_pks)

        # Pixel window function for each tracer
        self.pixwin = {}
        for b in self.bins:
            nside = b.get('nside', None)
            if nside:
                pixwin = beam_hpix(self.l_sample, nside)
            else:
                pixwin = np.ones(self.l_sample.size)
            self.pixwin[b['name']] = pixwin

        # Initialize parameterless Halo model stuff
        if self.bias_model == 'HaloModel':
            # Mass definition
            if self.mass_def_str == 'fof':
                self.massdef = ccl.halos.MassDef('fof', 'critical')
            else:
                rt = self.mass_def_str[-1]
                if rt == 'c':
                    rhotyp = 'critical'
                elif rt == 'm':
                    rhotyp = 'matter'
                else:
                    raise ValueError(f"Unknown density type {rt}")
                if self.mass_def_str[:-1] == 'Vir':
                    Delta = 'vir'
                else:
                    Delta = float(self.mass_def_str[:-1])
                self.massdef = ccl.halos.MassDef(Delta, rhotyp)
            # Mass function
            self.mfc = ccl.halos.mass_function_from_name(self.mf_name)
            # Halo bias
            self.hbc = ccl.halos.halo_bias_from_name(self.hb_name)
            # Concentration
            cmc = ccl.halos.concentration_from_name(self.cm_name)
            self.cm = cmc(mdef=self.massdef)
            # Default profiles for different quantities
            self.profs = {'galaxy_density': None,
                          'galaxy_shear': ccl.halos.HaloProfileNFW(self.cm),
                          'cmb_convergence': ccl.halos.HaloProfileNFW(self.cm),
                          'cmb_tSZ': ccl.halos.HaloProfilePressureGNFW()}
            # Profile 2-point function for HOD
            self.p2pt_HOD = ccl.halos.Profile2ptHOD()
            # Halo model correction for the transition regime
            if self.HM_correction == 'halofit':
                self.hmcorr = HalomodCorrection()
            else:
                self.hmcorr = None
                
        elif self.bias_model == 'BACCO':
            # load the emulators
            import baccoemu_beta as baccoemu
            self.lbias = baccoemu.Lbias_expansion()

            # set k values (TODO: could also keep k=None) # tuks l10k_max is too big
            #self.k_s = np.logspace(self.l10k_min_pks,
            #                        self.l10k_max_pks,
            #                        self.nk_pks)
            # set k values in units of h/Mpc; smoothing scale is 0.75 h/Mpc so between 0.45 and 0.675 for h = 0.6 and 0.9
            #self.k_s = np.logspace(-2, np.log10(0.75), 1000)
            self.k_s = np.logspace(-1.999, np.log10(0.75), 1000)
            
            # redshifts for creating the Pk-2d object
            #z = np.array([3.0, 2.0, 1.5, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0]) # BACCO goes to z = 1.5
            z = np.array([1.5, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
            self.a_s = 1./(1. + z)
            
            # labels of the bias parameters
            self.bias_labels = ['b0', 'b1', 'b2', 'bs', 'bn']

            # create a dictionary for the emulator outputs, following the bacco outputs
            self.bacco_dict = {}
            counter = 0
            for i in range(len(self.bias_labels)):
                for j in range(len(self.bias_labels)):
                    if i > j: continue # !!!!!!!!!!!!!! notice different convention
                    self.bacco_dict[self.bias_labels[i] + '_' + self.bias_labels[j]] = counter
                    self.bacco_dict[self.bias_labels[j] + '_' + self.bias_labels[i]] = counter
                    counter += 1
            # bacco: 0: <1,1>, 1: <1,d>, 2: <1,d^2>, 3: <1,s^2>, 4: <1,nabla^2 d>, 5: <d,d>, 6: <d,d^2>, 7: <d,s^2>, 8: <d,nabla^2 d>, 9: <d^2,d^2> (!), 10: <d^2,s^2> (!), 11: <d^2,nabla^2 d>, 12: <s^2,s^2> (!), 13: <s^2,nabla^2 d>, 14: <nabla^2 d,nabla^2 d>
            
        elif self.bias_model == 'anzu':
            from anzu.emu_funcs import LPTEmulator
            
            # Initialize Hybrid EFT emulator
            self.emu = LPTEmulator(use_sigma_8=True, kecleft=True)

            # k values over which templates are computed # h/Mpc
            self.k_s = np.logspace(-2, 0, 1000)
            self.k_lpt = np.logspace(-4, 0, 1000)

            # If we don't pass redshift (scale factor), emu assumes:
            # [3.0, 2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0] but zmax = 2.0!, so 9
            #z = np.array([2.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
            z = np.linspace(0., 4., 30)[::-1]
            self.a_s = 1./(1. + z)

            # Available fields for the emulator # B.H. bn is not available
            self.b_emu = ['b0', 'b1', 'b2', 'bs']

            # Create a dictionary for the emulator outputs
            self.emu_dict = {}
            counter = 0
            for i in range(len(self.b_emu)):
                for j in range(len(self.b_emu)):
                    if j > i: continue # !!!!!!!!!!!!!!!! notice different convention
                    self.emu_dict[self.b_emu[i] + '_' + self.b_emu[j]] = counter
                    self.emu_dict[self.b_emu[j] + '_' + self.b_emu[i]] = counter
                    counter += 1

            # cleft: 0: k, 1: <1,1>, 2: 2*<1,d>, 3: <d,d>, 4: 2*<1,d^2/2>, 5: 2*<d,d^2/2>, 6: <d^2/2,d^2/2> (!), 7: 2*<1,s^2>, 8: 2*<d,s^2>, 9: 2*<d^2/2,s^2> (!), 10: <s^2,s^2> (!), 11: 2*<1, O3>, 12: 2*<d, O3>
            # anzu: 0: k, 1: <1,1>, 2: <1,d>, 3: <d,d>, 4: <1,d^2>, 5: <d,d^2>, 6: <d^2,d^2>, 7: <1,s^2>, 8: <d,s^2> (!), 9: <d^2,s^2> (!), 10: <s^2,s^2> (!),

            
                    
    def _read_data(self):
        """
        Reads sacc file
        Selects relevant data.
        Applies scale cuts
        Reads tracer metadata (N(z))
        Reads covariance
        """
        import sacc

        def get_suffix_for_tr(tr):
            q = tr.quantity
            if (q == 'galaxy_density') or (q == 'cmb_convergence'):
                return '0'
            elif q == 'galaxy_shear':
                return 'e'
            else:
                raise ValueError(f'dtype not found for quantity {q}')

        def get_lmax_from_kmax(cosmo, kmax, zmid):
            chi = ccl.comoving_radial_distance(cosmo, 1./(1+zmid))
            lmax = np.max([10., kmax * chi - 0.5])
            return lmax

        s = sacc.Sacc.load_fits(self.input_file)

        # 1. Iterate through tracers and collect properties
        self.bin_properties = {}
        # We use a default cosmology to map k_max into ell_max
        cosmo_lcdm = ccl.CosmologyVanillaLCDM()
        # We give it a generous default value
        kmax_default = self.defaults.get('kmax', 1.0)
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
                #self.defaults[b['name']]['lmax'] = lmax
                # If lmax cut is available for the tracer, take that instead (but only if it is more conservative)
                self.defaults[b['name']]['lmax'] = np.min([lmax, self.defaults[b['name']].get('lmax', 1E30)])
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
            cl_name1 = get_suffix_for_tr(s.tracers[tn1])
            cl_name2 = get_suffix_for_tr(s.tracers[tn2])
            # TODO: temporary solution cause sacc doesn't like cl_e0
            if cl_name1 == 'e' and cl_name2 == '0': # get rid of cl_e0 case
                ind = s.indices('cl_%s%s' % (cl_name2, cl_name1), (tn1, tn2),
                                ell__gt=lmin, ell__lt=lmax)
            else:
                ind = s.indices('cl_%s%s' % (cl_name1, cl_name2), (tn1, tn2),
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
        self.l_max_sample = -1E30
        for cl in self.twopoints:
            # Get the suffix for both tracers
            tn1, tn2 = cl['bins']
            cl_name1 = get_suffix_for_tr(s.tracers[tn1])
            cl_name2 = get_suffix_for_tr(s.tracers[tn2])
            # TODO: temporary solution cause sacc doesn't like cl_e0
            if cl_name1 == 'e' and cl_name2 == '0': # get rid of cl_e0 case
                l, c_ell, cov, ind = s.get_ell_cl('cl_%s%s' % (cl_name2, cl_name1),
                                                  tn1,
                                                  tn2,
                                                  return_cov=True,
                                                  return_ind=True)
            else:
                l, c_ell, cov, ind = s.get_ell_cl('cl_%s%s' % (cl_name1, cl_name2),
                                                  tn1,
                                                  tn2,
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
            if np.amax(bpw.values) > self.l_max_sample:
                self.l_max_sample = np.amax(bpw.values)

            self.cl_meta.append({'bin_1': tn1,
                                 'bin_2': tn2,
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size,
                                                    dtype=int)),
                                 'l_bpw': bpw.values,
                                 'w_bpw': bpw.weight.T})
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)
        # Reorder data vector and covariance
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.covmat[indices][:, indices]
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
        f = interp1d(self.l_sample, cl_in)
        cl_unbinned = f(l_bpw)
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
        bz = np.ones_like(z)
        if self.bias_model == 'Linear':
            b0 = pars[self.input_params_prefix + '_' + name + '_b0']
            bp = pars.get(self.input_params_prefix + '_' + name + '_b0p', 0.)
            bz = b0 + bp * (z - zmean)
        elif self.bias_model == 'BACCO':
            pass
        elif self.bias_model == 'anzu':
            pass
        return (z, bz)

    def _get_ia_bias(self, cosmo, name, **pars):
        """ Intrinsic alignment amplitude.
        """
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            if self.ia_model == 'IAPerBin':
                A = pars[self.input_params_prefix + '_' + name + '_A_IA']
                A_IA = np.ones_like(z) * A
            elif self.ia_model == 'IADESY1':
                A0 = pars[self.input_params_prefix + '_A_IA']
                eta = pars[self.input_params_prefix + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            elif self.ia_model == 'IADESY1_PerSurvey': # TODO: make pretty
                # This assumes that name = survey__zbin
                survey = name.split('__')[0]
                A0 = pars[self.input_params_prefix + '_' + survey + '_A_IA']
                eta = pars[self.input_params_prefix + '_' + survey + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            else:
                raise LoggedError(self.log, "Unknown IA model %s" %
                                  self.ia_model)
            return (z, A_IA)

    def _get_tracers(self, cosmo, **pars):
        """ Obtains CCL tracers (and perturbation theory tracers,
        and halo profiles where needed) for all used tracers given the
        current parameters."""
        trs = {}
        is_PT_bias = self.bias_model in ['LagrangianPT', 'EulerianPT']
        for name, q in self.tracer_qs.items():
            prefix = self.input_params_prefix + '_' + name
            if self.bias_model == 'HaloModel':
                prof = self.profs[q]
                normed = True

            if q == 'galaxy_density':
                nz = self._get_nz(cosmo, name, **pars)
                bz = self._get_bz(cosmo, name, **pars)
                t = ccl.NumberCountsTracer(cosmo, dndz=nz,
                                           bias=bz, has_rsd=False)
                if is_PT_bias:
                    z = self.bin_properties[name]['z_fid']
                    zmean = self.bin_properties[name]['zmean_fid']
                    b1 = pars[self.input_params_prefix + '_' + name + '_b1']
                    b1p = pars.get(self.input_params_prefix + '_' +
                                   name + '_b1p', 0.)
                    bz = b1 + b1p * (z - zmean)
                    b2 = pars[self.input_params_prefix + '_' + name + '_b2']
                    bs = pars[self.input_params_prefix + '_' + name + '_bs']
                    ptt = pt.PTNumberCountsTracer(b1=(z, bz), b2=b2, bs=bs)
                if self.bias_model == 'HaloModel':
                    hod_pars = {k: pars[prefix + '_' + k]
                                for k in ['lMmin_0', 'lM1_0']}
                    #if self.M0_track:
                    #    hod_pars['lM0_0'] = hod_pars['lMmin_0']
                    #else:
                    hod_pars['lM0_0'] = pars[prefix + '_lM0_0']
                    slM = pars.get(prefix + '_siglM_0', None)
                    if slM is None:
                        slM = pars[self.input_params_prefix + '_siglM_0']
                    hod_pars['siglM_0'] = slM
                    prof = ccl.halos.HaloProfileHOD(self.cm)
                    prof.update_parameters(**hod_pars)
                if self.bias_model == 'BACCO':
                    bias = {}
                    for b in self.bias_labels:
                        bias[b] = pars.get(self.input_params_prefix + '_' + name + '_' + b, 0.)
                    bias['b0'] = 1. # TODO: can also just feed it as a fixed parameter in the yaml
                if self.bias_model == 'anzu':
                    # note that anzu is missing bn
                    bias = {}
                    for b in self.b_emu:
                        bias[b] = pars.get(self.input_params_prefix + '_' + name + '_' + b, 0.)
                    bias['b0'] = 1. # TODO: can also just feed it as a fixed parameter in the yaml
            elif q == 'galaxy_shear':
                nz = self._get_nz(cosmo, name, **pars)
                ia = self._get_ia_bias(cosmo, name, **pars)
                t = ccl.WeakLensingTracer(cosmo, nz, ia_bias=ia)
                if is_PT_bias:
                    ptt = pt.PTMatterTracer()
                if self.bias_model == 'BACCO':
                    bias = None
                if self.bias_model == 'anzu':
                    bias = None
            elif q == 'cmb_convergence':
                # B.H. TODO: pass z_source as parameter to the YAML file
                t = ccl.CMBLensingTracer(cosmo, z_source=1100)
                if is_PT_bias:
                    ptt = pt.PTMatterTracer()
                if self.bias_model == 'BACCO':
                    bias = None
                if self.bias_model == 'anzu':
                    bias = None
            elif q == 'cmb_tSZ':
                t = ccl.tSZTracer(cosmo, z_max=3.)
                if self.bias_model == 'HaloModel':
                    o_m_b = pars.get(self.input_params_prefix +
                                     '_mass_bias', 1.)
                    prof.update_parameters(mass_bias=o_m_b)
                    normed = False
                if self.bias_model == 'BACCO':
                    bias = None
                if self.bias_model == 'anzu':
                    bias = None
                else:
                    raise NotImplementedError("Can't do tSZ without"
                                              " the halo model.")

            trs[name] = {}
            trs[name]['ccl_tracer'] = t
            if is_PT_bias:
                trs[name]['PT_tracer'] = ptt
            if self.bias_model == 'HaloModel':
                trs[name]['Profile'] = prof
                trs[name]['Normed'] = normed
            if self.bias_model == 'BACCO':
                trs[name]['bias'] = bias # if not galaxy tracer, this is None
            if self.bias_model == 'anzu':
                trs[name]['bias'] = bias # if not galaxy tracer, this is None
        return trs

    def _cosmo_to_bacco(self, cosmo):
        # TODO: add neutrinos to omega_matter
        pars = {
            'omega_matter' : cosmo['Omega_b']+cosmo['Omega_c'], # correct
            'omega_baryon' : cosmo['Omega_b'],
            'hubble' : cosmo['H0']/100.,
            'ns' : cosmo['n_s'],
            'sigma8' : ccl.sigma8(cosmo), # tuks should be just CDM + baryons
            'neutrino_mass' : np.sum(cosmo['m_nu']), # tuks summing the masses of the neutrino species (should probs be selecting the single neutrino species in pyccl)
            'w0' : cosmo['w0'],
            'wa' : cosmo['wa'],
            'expfactor' : 1.  # random scale factor just to initialize
        }
        return pars

        
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
        elif self.bias_model in ['EulerianPT', 'LagrangianPT']:
            if self.k_SN_suppress > 0:
                k_filter = self.k_SN_suppress
            else:
                k_filter = None
            if self.bias_model == 'EulerianPT':
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=self.l10k_min_pks,
                                    log10k_max=self.l10k_max_pks,
                                    nk_per_decade=self.nk_per_dex_pks,
                                    a_arr=self.a_s_pks, k_filter=k_filter)
            else:
                ptc = LPTCalculator(log10k_min=self.l10k_min_pks,
                                    log10k_max=self.l10k_max_pks,
                                    nk_per_decade=self.nk_per_dex_pks,
                                    a_arr=self.a_s_pks, h=cosmo['h'],
                                    k_filter=k_filter)
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            pk_lin_z0 = ccl.linear_matter_power(cosmo, ptc.ks, 1.)
            Dz = ccl.growth_factor(cosmo, ptc.a_s)
            ptc.update_pk(pk_lin_z0, Dz)
            return {'ptc': ptc, 'pk_mm': pkmm}
        elif self.bias_model == 'HaloModel':
            cosmo.compute_linear_power()
            cosmo.compute_nonlin_power()
            cosmo.compute_sigma()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            mf = self.mfc(cosmo, mass_def=self.massdef)
            hb = self.hbc(cosmo, mass_def=self.massdef)
            hmc = ccl.halos.HMCalculator(cosmo, mf, hb, self.massdef)
            return {'hmc': hmc, 'pk_mm': pkmm}
        elif self.bias_model == 'BACCO':
            pk2d_bacco, pkmm = self._compute_bacco(cosmo)
            return {'pk2d_bacco': pk2d_bacco, 'pk_mm': pkmm}
        elif self.bias_model == 'anzu':
            pk2d_anzu, pkmm = self._compute_anzu(cosmo)
            return {'pk2d_anzu': pk2d_anzu, 'pk_mm': pkmm}
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bias_model)

    def _cosmo_to_anzu(self, cosmo):
        # Initialize cosmology vector
        cosmovec = np.zeros(8)
        cosmovec[0] = cosmo['Omega_b'] * (cosmo['H0'] / 100)**2
        cosmovec[1] = cosmo['Omega_c'] * (cosmo['H0'] / 100)**2
        cosmovec[2] = cosmo['w0']
        cosmovec[3] = cosmo['n_s']
        cosmovec[4] = ccl.sigma8(cosmo) #np.log(cosmo['A_s'] * 1.e10)
        cosmovec[5] = cosmo['H0']
        cosmovec[6] = cosmo['Neff']
        cosmovec[7] = 1. # random scale factor just to initialize
                                    
        # Vector of cosmological parameters
        cosmovec = np.atleast_2d(cosmovec)

        return cosmovec


    def _compute_anzu_old(self, cosmo):
        # 9 redshifts, 10 combinations between bias params, and the rest are the ks
        num_comb = int(len(self.b_emu)*(len(self.b_emu)-1)/2 + len(self.b_emu))
        emu_spec = np.zeros((len(self.a), num_comb, len(self.k_s)))
        
        # Get the emulator prediction for this cosmology
        cosmovec = self._cosmo_to_anzu(cosmo) # tuks might need to do sigma8 rather than A_s

        t1 = time.time()
        for i in range(len(self.a)):
            cosmovec[-1, -1] = self.a[i]
            emu_spec[i] = self.emu.predict(self.k_s, cosmovec) # tuks might need to add another argument
        print("time = ", time.time()-t1)
        # tuks I convert the units in the P(k,a) function
        
        cosmo.compute_nonlin_power()
        #cosmo.compute_sigma()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        return emu_spec, pkmm


    def _compute_anzu(self, cosmo):
        # 9 redshifts, 10 combinations between bias params, and the rest are the ks
        num_comb = int(len(self.b_emu)*(len(self.b_emu)-1)/2 + len(self.b_emu))
        lpt_spec = np.zeros((len(self.a_s), num_comb, len(self.k_lpt)))
        
        #if self.cleftobj is None:
        # Do the full calculation again, as the cosmology changed.
        pk = ccl.linear_matter_power(
            cosmo, self.k_lpt * cosmo['h'], 1) * (cosmo['h'])**3 # Mpc/h^3
            
        # Function to obtain the no-wiggle spectrum.
        # Not implemented yet, maybe Wallisch maybe B-Splines?
        # pnw = p_nwify(pk)
        # For now just use Stephen's standard savgol implementation.
        cleftobj = RKECLEFT(self.k_lpt, pk)

        # Adjust growth factors
        Dz = ccl.background.growth_factor(cosmo, self.a_s)
        for i, D in enumerate(Dz):

            cleftobj.make_ptable(D=D, kmin=self.k_lpt[0], kmax=self.k_lpt[-1], nk=1000)
            cleftpk = cleftobj.pktable.T # shape after transposing should be [10, k]
    
            # Adjust normalizations to match anzu measurements. B.H. Factors come from dividing anzu by cleft in the initialization comments
            #cleftpk[3:, :] = cleftobj.pktable.T[3:, :] # B.H. TODO:  is this not meaningless?
            cleftpk[2, :] /= 2. # dmd1
            cleftpk[6, :] *= 4. # d2d2
            cleftpk[7, :] /= 2. # dms2
            cleftpk[8, :] /= 2. # d1s2
            #Do we have to spline every time? nevertheless # B.H. TODO: I think unnecessary  but need to check make_ptable
            cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')
            lpt_spec[i] = cleftspline(self.k_lpt)[1:num_comb+1, :]
            
        # Computed the relevant lpt predictions, plug into emu
        # 1. Set up cosmovec for anzu from CCL cosmo object
        # Final array should be [Nz, Nparam]
        # Parameter order is as specified in anzu documentation.
        # 2. Update ptc for units that are h-ful for anzu 
        # 3. Compute emulator basis functions
        # NOTE: Check ptc.lpt_table has the proper normalization for Anzu LPT spectra.

        # Get the emulator prediction for this cosmology
        cosmovec = self._cosmo_to_anzu(cosmo)
        anzu_cosmo = np.tile(cosmovec, len(self.a_s)).reshape(-1, cosmovec.shape[-1]) # cause at least 2d;  becomes len(a_s), len(cosmovec)
        anzu_cosmo[:, -1] = self.a_s

        # Convert units back to Mpc^3
        emu_spec = self.emu.predict(self.k_s, anzu_cosmo, spec_lpt=lpt_spec, k_lpt=self.k_lpt)
        emu_spec /= cosmo['h']**3
        # kinda dumb B.H. please change
        self.h = cosmo['h']
        #self.ks = k_emu*cosmo['h'] # we actually do this when getting the pk2d object
        
        # TODO: missing k^2 terms
        
        # normalize bacco following LPT convention (see Andrina)
        for i in range(num_comb):
            if i in [4, 5, 7, 8]: # dmd2, d1d2, dms2, d1s2
                emu_spec[:, i, :] *= 0.5
            elif i in [6, 9, 10]: # d2d2, d2s2, s2s2
                emu_spec[:, i, :] *= 0.25
        cosmo.compute_nonlin_power()
        #cosmo.compute_sigma()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        return emu_spec, pkmm


    def _compute_bacco(self, cosmo):
        
        # translate the pyccl cosmology parameters into bacco notation
        pars = self._cosmo_to_bacco(cosmo)

        # convert k_s [Mpc^-1] into h Mpc^-1 units just for the calculation
        #k = self.k_s/pars['hubble']
        # self.k_s already in h/Mpc
        k = self.k_s
        
        # 10 redshifts, 15 combinations between bias params, and the ks (added b0)
        num_comb = int((len(self.bias_labels))*(len(self.bias_labels)-1)/2 + len(self.bias_labels))
        pk2d_bacco = np.zeros((len(self.a_s), num_comb, len(self.k_s)))

        # compute the power for each redshift
        #t1 = time.time()
        for i in range(len(self.a_s)):
            pars['expfactor'] = self.a_s[i]
            if self.a_s[i] >= 0.4: # if within BACCO, emulator
                # call the emulator of the nonlinear 15 lagrangian bias expansion terms, shape is (15, len(k))
                k, pnn = self.lbias.get_nonlinear_pnn(pars, k=k)
            else: # if outside, normal lpt
                pnn = self.lbias.get_lpt_pk(pars, k=k)
            pk2d_bacco[i, :, :] = pnn

        # normalize bacco following LPT convention (see Andrina)
        for i in range(num_comb):
            if i in [2, 6, 3, 7, 4, 8]: #dmd2, d1d2, dms2, d1s2, dmn2, d1n2
                pk2d_bacco[:, i, :] *= 0.5
            elif i in [9, 10, 12, 14, 11, 13]: # d2d2, d2s2, s2s2, n2n2, d2n2, s2n2
                pk2d_bacco[:, i, :] *= 0.25
        #print("time = ", time.time()-t1)
        
        # convert the spit out result from (Mpc/h)^3 to Mpc^3 (bacco uses h units, but pyccl doesn't)
        pk2d_bacco /= pars['hubble']**3
        # kinda dumb B.H. please change
        self.h = cosmo['h']
        
        # call the emulator of the LPT-predicted 15 lagrangian bias expansion terms
        #_, plpt = lbias.get_lpt_pk(pars, k=k) # TODO: check if both calls are necessary

        #labels = lbias.lb_term_labels # just so I don't forget this is an option
        #print(labels)

        # get the matter-matter power spectrum in units of Mpc^3
        #pkmm = pnn[0]

        # TODO: is this the best way to handle the matter-matter power spectrum? Or:
        #cosmo.compute_linear_power()
        cosmo.compute_nonlin_power()
        #cosmo.compute_sigma()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')

        #pkmm_ccl = pkmm.eval(self.k_s, self.a_s[0], cosmo) # Mpc
        #pkmm_bacco = pk2d_bacco[0, 0, :] # Mpc
        #np.save("pkmm_ccl.npy", pkmm_ccl) # Mpc
        #np.save("pkmm_bacco.npy", pkmm_bacco) # Mpc
        #np.save("k_s.npy", self.k_s) # Mpc
        
        return pk2d_bacco, pkmm
        
    def _get_pkxy(self, cosmo, clm, pkd, trs, **pars):
        """ Get the P(k) between two tracers. """
        q1 = self.tracer_qs[clm['bin_1']]
        q2 = self.tracer_qs[clm['bin_2']]        
        
        if (self.bias_model == 'Linear') or (self.bias_model == 'BzNone'):
            if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
                return pkd['pk_mm']  # galaxy-galaxy
            elif ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                return pkd['pk_mm']  # galaxy-matter
        elif (self.bias_model == 'EulerianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[clm['bin_1']]['PT_tracer']
                ptt2 = trs[clm['bin_2']]['PT_tracer']
                pk_pt = get_ept_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'], sub_lowk=False)
                return pk_pt
        elif (self.bias_model == 'LagrangianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[clm['bin_1']]['PT_tracer']
                ptt2 = trs[clm['bin_2']]['PT_tracer']
                pk_pt = get_lpt_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'])
                return pk_pt
        elif self.bias_model == 'HaloModel':
            k_s = np.logspace(self.l10k_min_pks,
                              self.l10k_max_pks,
                              self.nk_pks)
            comb = self.qabbr[q1]+self.qabbr[q2]
            p1 = trs[clm['bin_1']]['Profile']
            p2 = trs[clm['bin_2']]['Profile']
            norm1 = trs[clm['bin_1']]['Normed']
            norm2 = trs[clm['bin_2']]['Normed']

            if q1 == q2 == 'galaxy_density':
                prof2pt = self.p2pt_HOD
            else:
                r = pars.get(self.input_params_prefix + '_rho' + comb, 0.)
                prof2pt = ccl.halos.Profile2pt()#r_corr=r)

            if self.HM_correction == "HMCode":
                alpha = pars.get(self.input_params_prefix +
                                 '_alpha' + comb, None)
                if alpha is None:
                    alpha = pars.get(self.input_params_prefix + '_alpha', 1.)

                def fsmooth(a): return alpha
            else:
                fsmooth = None

            def fsuppress(a): return self.k_SN_suppress

            pkt = ccl.halos.halomod_power_spectrum(cosmo, pkd['hmc'], k_s,
                                                   self.a_s_pks, p1,
                                                   prof_2pt=prof2pt, prof2=p2,
                                                   normprof1=norm1,
                                                   normprof2=norm2)
            #smooth_transition=fsmooth,
            #supress_1h=fsuppress)
            if self.HM_correction == 'halofit':
                A = pars.get(self.input_params_prefix +
                             '_Ahmc' + comb, None)
                if A is None:
                    A = pars.get(self.input_params_prefix + '_Ahmc', 1.)
                ratio = np.array([1+A*self.hmcorr.rk_interp(k_s, a)
                                  for a in self.a_s_pks])
                pkt *= ratio
            pk = ccl.Pk2D(a_arr=self.a_s_pks, lk_arr=np.log(k_s),
                          pk_arr=np.log(pkt),
                          extrap_order_lok=1, extrap_order_hik=2,
                          cosmo=cosmo, is_logp=True)
            return pk
        elif self.bias_model == 'BACCO':
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')): # TODO: is this fine?
                return pkd['pk_mm']  # matter-matter
                #return pkd['pk2d_bacco'][:, 0, :]  # matter-matter from bacco emulator # apparently this is not total matter but just CDM + baryons
            else:
                # bias is a dictionary holding the value for each bias parameter or None if not galaxy tracer
                bias_eft1 = trs[clm['bin_1']]['bias']
                bias_eft2 = trs[clm['bin_2']]['bias']
                pk2d = self._get_pk_2d_bacco(pkd['pk2d_bacco'], bias_eft1, bias_eft2, q1, q2)
                return pk2d
            
        elif self.bias_model == 'anzu':
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')): # TODO: is this fine?
                return pkd['pk_mm']  # matter-matter
            else:
                # bias is a dictionary holding the value for each bias parameter or None if not galaxy tracer
                bias_eft1 = trs[clm['bin_1']]['bias']
                bias_eft2 = trs[clm['bin_2']]['bias']
                pk2d = self._get_pk_2d_anzu(pkd['pk2d_anzu'], bias_eft1, bias_eft2, q1, q2)
                return pk2d
        
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bias_model)


    def _get_pk_2d_bacco(self, pk2d_bacco, bias_eft1, bias_eft2, q1, q2):
        # TODO: the calculation could be sped up, but need to figure out how to do for two sets of biases
        #bias = [0.75, 0.25, 0.1, 1.4] # b1, b2, bs2, blaplacian
        #kgal, pgalauto, pgalcross = lbias.get_galaxy_real_pk(pars, bias, k=None)
        # in this current version b_trs has b0 = 1 as well
        
        # Initialize power spectrum Pk_a(as, ks)
        Pk_a = np.zeros_like(pk2d_bacco[:, 0, :])
        
        # If both tracers are galaxies, Pk^{tr1,tr2} = f_i^bin1 * f_j^bin2 * Pk_ij
        if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]
                for key2 in bias_eft2.keys():
                    bias2 = bias_eft2[key2]
                    if key1+'_'+key2 in self.bacco_dict.keys():
                        comb = self.bacco_dict[key1+'_'+key2]
                    else:
                        comb = self.bacco_dict[key2+'_'+key1]
                    Pk_a += bias1*bias2*pk2d_bacco[:, comb, :]

        # If first tracer is galaxies and second is matter, Pk^{tr1,tr2} = f_i^bin1 * 1. * Pk_0i
        elif (q1 == 'galaxy_density') and (q2 != 'galaxy_density'):
            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]
                comb = self.bacco_dict['b0'+'_'+key1]
                Pk_a += bias1*pk2d_bacco[:, comb, :]

        # If second tracer is galaxies and first is matter, Pk^{tr1,tr2} = f_j^bin2 * 1. * Pk_0j
        elif (q1 != 'galaxy_density') and (q2 == 'galaxy_density'):
            for key2 in bias_eft2.keys():
                bias2 = bias_eft2[key2]
                comb = self.bacco_dict['b0'+'_'+key2]
                Pk_a += bias2*pk2d_bacco[:, comb, :]

        # Convert ks from [Mpc/h]^-1 to [Mpc]^-1
        lk_arr = np.log(self.k_s*self.h)
                
        # Compute the 2D power spectrum
        pk_2d_bacco = ccl.Pk2D(a_arr=self.a_s, lk_arr=lk_arr, pk_arr=Pk_a, is_logp=False)
        return pk_2d_bacco

    def _get_pk_2d_anzu(self, pk2d_anzu, bias_eft1, bias_eft2, q1, q2):
        
        # Initialize power spectrum Pk_a(as, ks)
        Pk_a = np.zeros_like(pk2d_anzu[:, 0, :])
        
        # If both tracers are galaxies, Pk^{tr1,tr2} = f_i^bin1 * f_j^bin2 * Pk_ij
        if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]
                for key2 in bias_eft2.keys():
                    bias2 = bias_eft2[key2]
                    if key1+'_'+key2 in self.emu_dict.keys():
                        comb = self.emu_dict[key1+'_'+key2]
                    else:
                        comb = self.emu_dict[key2+'_'+key1]
                    Pk_a += bias1*bias2*pk2d_anzu[:, comb, :]

        # If first tracer is galaxies and second is matter, Pk^{tr1,tr2} = f_i^bin1 * 1. * Pk_0i
        elif (q1 == 'galaxy_density') and (q2 != 'galaxy_density'):
            for key1 in bias_eft1.keys():
                bias1 = bias_eft1[key1]
                comb = self.emu_dict['b0'+'_'+key1]
                Pk_a += bias1*pk2d_anzu[:, comb, :]

        # If second tracer is galaxies and first is matter, Pk^{tr1,tr2} = f_j^bin2 * 1. * Pk_0j
        elif (q1 != 'galaxy_density') and (q2 == 'galaxy_density'):
            for key2 in bias_eft2.keys():
                bias2 = bias_eft2[key2]
                comb = self.emu_dict['b0'+'_'+key2]
                Pk_a += bias2*pk2d_anzu[:, comb, :]

        # Convert ks from [Mpc/h]^-1 to [Mpc]^-1
        lk_arr = np.log(self.k_s*self.h)
                          
        # Already in Mpc^3 Same for the power spectrum: convert to Mpc^3
        #Pk_a /= (cosmo['H0']/100.)**3.
                
        # Compute the 2D power spectrum
        pk_2d_anzu = ccl.Pk2D(a_arr=self.a_s, lk_arr=lk_arr, pk_arr=Pk_a, is_logp=False)
        return pk_2d_anzu
        
    def _get_pixel_window(self, clm):
        pix1 = self.pixwin[clm['bin_1']]
        pix2 = self.pixwin[clm['bin_2']]
        return pix1*pix2

    def _get_cl_all(self, cosmo, pk, **pars):
        """ Compute all C_ells."""
        # Gather all tracers
        trs = self._get_tracers(cosmo, **pars)

        # baryon correction model should be oche
        if self.baryon_model == 'BCM':
            pk2d = pk['pk_mm']
            # kopele is for testing
            #k_s = np.logspace(-3, np.log10(0.45), 1000)
            #kopele1 = pk2d.eval(k_s, 1., cosmo)
            ccl.bcm_correct_pk2d(cosmo, pk2d)
            #kopele2 = pk2d.eval(k_s, 1., cosmo)
            #print("fractional difference = ", (kopele1-kopele2)/kopele1) # on the order of 2% for smallest scales and tiny for larger scales
            pk['pk_mm'] = pk2d
        
        # Correlate all needed pairs of tracers
        cls = []
        for clm in self.cl_meta:
            pkxy = self._get_pkxy(cosmo, clm, pk, trs, **pars)
            cl = ccl.angular_cl(cosmo,
                                trs[clm['bin_1']]['ccl_tracer'],
                                trs[clm['bin_2']]['ccl_tracer'],
                                self.l_sample, p_of_k_a=pkxy)
            # Pixel window function
            cl *= self._get_pixel_window(clm)
            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            cls.append(clb)
        return cls

    def _apply_shape_systematics(self, cls, **pars):
        if self.shape_model == 'ShapeMultiplicative':
            # Multiplicative shear bias
            for i, clm in enumerate(self.cl_meta):
                q1 = self.tracer_qs[clm['bin_1']]
                q2 = self.tracer_qs[clm['bin_2']]
                if q1 == 'galaxy_shear':
                    m1 = pars[self.input_params_prefix + '_' +
                              clm['bin_1'] + '_m']
                else:
                    m1 = 0.
                if q2 == 'galaxy_shear':
                    m2 = pars[self.input_params_prefix + '_' +
                              clm['bin_2'] + '_m']
                else:
                    m2 = 0.
                prefac = (1+m1) * (1+m2)
                cls[i] *= prefac

    def get_cls_theory(self, **pars):
        # Get cosmological model
        res = self.provider.get_CCL()
        cosmo = res['cosmo']

        # First, gather all the necessary ingredients for the different P(k)
        pkd = res['pk_data']

        # Then pass them on to convert them into C_ells
        cls = self._get_cl_all(cosmo, pkd, **pars)

        # Multiplicative bias if needed
        self._apply_shape_systematics(cls, **pars)
        return cls

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
            bpw = sacc.BandpowerWindow(clm['l_bpw'], clm['w_bpw'].T)
            s.add_ell_cl(f'cl_{p1}{p2}', clm['bin_1'], clm['bin_2'],
                         clm['l_eff'], cl, window=bpw)

        s.add_covariance(self.cov)
        return s

    def _get_theory(self, **pars):
        """ Computes theory vector."""
        cls = self.get_cls_theory(**pars)

        # Flattening into a 1D array
        cl_out = np.zeros(self.ndata)
        ell = np.zeros(self.ndata)
        bins = []
        for clm, cl in zip(self.cl_meta, cls):
            cl_out[clm['inds']] = cl
            # B.H. saving file
            ell[clm['inds']] = clm['l_eff'] #
            bins.append(f"{clm['bin_1']:s}_!_{clm['bin_2']:s}") #
        #return cl_out # og
        return ell, cl_out, bins #
    #return cl_out # og
        

    def get_requirements(self):
        # By selecting `self._get_pk_data` as a `method` of CCL here,
        # we make sure that this function is only run when the
        # cosmological parameters vary.
        return {'CCL': {'methods': {'pk_data': self._get_pk_data}}}
        
    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        #t1 = time.time()
        #t = self._get_theory(**pars) # og
        # B.H. saving file
        ell, t, bins = self._get_theory(**pars) #
        r = t - self.data_vec
        chi2 = np.dot(r, np.dot(self.inv_cov, r))
        #print("time = ", time.time()-t1)
        # B.H. saving file
        np.savez_compressed(os.path.join('/users/boryanah/repos/xCell-likelihoods/analysis/data/', f'cl_cross_corr_{self.bias_model:s}.npz'), chi2=chi2, chi2_dof=chi2/self.ndata, cls=t, ells=ell, tracers=bins) # 
        return -0.5*chi2

    
