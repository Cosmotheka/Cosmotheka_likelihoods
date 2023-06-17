"""
Theory class that computes the angular power spectra. It uses the CCL theory
class and the nuisance parameters that cannot be factored out (such as the
galaxy bias parameters).
"""
from cobaya.theory import Theory
from cobaya.log import LoggedError
from scipy.interpolate import interp1d
from .pixwin import beam_hpix
import pyccl as ccl
import numpy as np


class Limber(Theory):
    """ Computes the angular power spectra
    """
    input_params_prefix: str = ""
    # N(z) model name
    nz_model: str = "NzNone"
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"
    # Limber integration method
    limber_integ: str = "qag_quad"

    # Sample type
    sample_type: str = "convolve"
    # Magnification bias selected per tracer in defaults
    # with_magnification_bias: bool = False

    def initialize(self):
        self.cl_meta = None
        self.l_sample = None
        self.tracer_qs = None
        self.bin_properties = None
        self.is_PT_bias = None
        self.bias_model = None
        self.provider = None

        self.sample_cen = self.sample_type in ['center', 'best']
        self.sample_bpw = self.sample_type == 'convolve'

    def initialize_with_provider(self, provider):
        self.provider = provider
        self.l_sample = self._get_ell_sampling()
        self._add_pixbeam_to_cl_meta()
        self.is_PT_bias = self.provider.get_is_PT_bias()
        self.bias_model = self.provider.get_bias_model()

    def get_requirements(self):
        return {'bias_model': None, 'is_PT_bias': None}

    def must_provide(self, **requirements):
        if "Limber" not in requirements:
            return {}

        options = requirements.get('Limber') or {}
        self.cl_meta = options.get("cl_meta")
        self.tracer_qs = options.get("tracer_qs")
        self.bin_properties = options.get("bin_properties")

        return {"CCL": None, "Pk": None}

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = self.provider.get_CCL()["cosmo"]
        state["Limber"] = {"cl_data": self._get_cl_data(cosmo,
                                                        **params_values_dict)}

    def get_Limber(self):
        """Get dictionary of Limber computed quantities.
        results['Limber'] contains the computed angular power spectra without
        the parameters that can be factored out.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['Limber']

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(np.log(1E-3+self.l_sample), cl_in)
        cl_unbinned = f(np.log(1E-3+l_bpw))
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_tracers(self, cosmo, **pars):
        """ Obtains CCL tracers (and perturbation theory tracers,
        and halo profiles where needed) for all used tracers given the
        current parameters."""
        trs0 = {}
        trs0_dnames = {}
        trs1 = {}
        trs1_dnames = {}
        for name, q in self.tracer_qs.items():
            if q == 'galaxy_density':
                dndz = self._get_nz(cosmo, name, **pars)
                z = dndz[0]
                oz = np.ones_like(z)
                if self.bias_model=='QuasarEvo':
                    # Laurent et al. 1705.04718
                    bz = 0.278*((1+z)**2-6.565)+2.393
                elif self.bias_model=='QuasarEvo1':
                    # Shen et al. 0810.4144
                    bz = 1.254+0.521*z**2
                elif self.bias_model=='QuasarEvo2':
                    # Croom et al. /astro-ph/0409314
                    bz = 0.53 + 0.289*(1 + z)**2
                elif self.bias_model=='QuasarEvo3':
                    # Chehade et al. 1603.04849
                    bz = 0.59 + 0.23*(1 + z)**2
                elif self.bias_model=='QuasarEvo4':
                    # Leistedt et al. 
                    bz = 1+((1+z)/2.5)**5
                else:
                    bz = oz

                tr = ccl.NumberCountsTracer(cosmo, dndz=dndz, bias=(z, bz),
                                            has_rsd=False)
                # Tracer for the unbiased component
                t0 = None
                t0n = [None]
                if self.bias_model in ['LagrangianPT', 'BaccoPT']:
                    t0 = tr
                    t0n = ["m"]
                # Tracers for the biased components
                t1 = [tr]
                t1n = ['d1']
                if self.is_PT_bias:
                    for bn, dn in zip(['b2', 'bs', 'bk2'], ['d2', 's2', 'k2']):
                        t1.append(tr)
                        t1n.append(dn)
                # Magnification
                if self.bin_properties[name]['mag_bias']:
                    # We use s = 1/5 here so that (2-5s)=1 and we multiply by
                    # (2-5s) in cl_total
                    tr = ccl.NumberCountsTracer(cosmo, dndz=dndz,
                                                bias=(z, z*0), has_rsd=False,
                                                mag_bias=(z, oz/5.))
                    t1.append(tr)
                    t1n.append("w")
            elif q == 'galaxy_shear':
                dndz = self._get_nz(cosmo, name, **pars)
                t0 = ccl.WeakLensingTracer(cosmo, dndz=dndz)
                t0n = ["w"]
                if self.ia_model == 'IANone':
                    t1 = None
                    t1n = [None]
                else:
                    ia_bias = self._get_ia_bias(cosmo, name, **pars)
                    t1 = [ccl.WeakLensingTracer(cosmo, dndz=dndz,
                                                has_shear=False,
                                                ia_bias=ia_bias)]
                    t1n = ['w']
            elif q == 'cmb_convergence':
                # B.H. TODO: pass z_source as parameter to the YAML file
                t0 = ccl.CMBLensingTracer(cosmo, z_source=1100)
                t0n = ['w']
                t1 = None
                t1n = [None]

            trs0[name] = t0
            trs1[name] = t1
            trs0_dnames[name] = t0n
            trs1_dnames[name] = t1n
        return trs0, trs1, trs0_dnames, trs1_dnames

    def _get_cl_data(self, cosmo, **pars):
        """ Compute all C_ells."""
        # Get P(k)s
        pkd = self.provider.get_Pk()["pk_data"]

        # Gather all tracers
        trs0, trs1, trs0_dnames, trs1_dnames = self._get_tracers(cosmo, **pars)

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
            else:
                raise RuntimeError("Something went wrong with the sampling!")

            n1 = clm['bin_1']
            n2 = clm['bin_2']
            t0_1 = trs0[n1]
            t0_2 = trs0[n2]
            t1_1 = trs1[n1]
            t1_2 = trs1[n2]
            # The unbiased tracers have only one element in the list "m" or
            # "w".
            t0dn_1 = trs0_dnames[n1][0]
            t0dn_2 = trs0_dnames[n2][0]
            t1dn_1 = trs1_dnames[n1]
            t1dn_2 = trs1_dnames[n2]
            # 00: unbiased x unbiased
            if t0_1 and t0_2:
                pk = pkd['pk_mm_sh_sh'] if 'pk_mm_sh_sh' in pkd.keys() else pkd[f'pk_{t0dn_1}{t0dn_2}']
                cl00 = ccl.angular_cl(
                    cosmo, t0_1, t0_2, ls, p_of_k_a=pk,
                    limber_integration_method=self.limber_integ) * clm['pixbeam']
                cls_00.append(cl00)
            else:
                cls_00.append(None)
            # 01: unbiased x biased
            if t0_1 and (t1_2 is not None):
                cl01 = []
                for t12, dn in zip(t1_2, t1dn_2):
                    pk = pkd[f'pk_{t0dn_1}{dn}']
                    if pk is not None:
                        cl = ccl.angular_cl(
                            cosmo, t0_1, t12, ls, p_of_k_a=pk,
                            limber_integration_method=self.limber_integ) * clm['pixbeam']
                    else:
                        cl = np.zeros_like(ls)
                    cl01.append(cl)
                cl01 = np.array(cl01)
            else:
                cl01 = None
            cls_01.append(cl01)
            # 10: biased x unbiased
            if n1 == n2:
                cls_10.append(cl01)
            else:
                if t0_2 and (t1_1 is not None):
                    cl10 = []
                    for t11, dn in zip(t1_1, t1dn_1):
                        pk = pkd[f'pk_{t0dn_2}{dn}']
                        if pk is not None:
                            cl = ccl.angular_cl(
                                cosmo, t11, t0_2, ls, p_of_k_a=pk,
                                limber_integration_method=self.limber_integ) * clm['pixbeam']
                        else:
                            cl = np.zeros_like(ls)
                        cl10.append(cl)
                    cl10 = np.array(cl10)
                else:
                    cl10 = None
                cls_10.append(cl10)
            # 11: biased x biased
            if (t1_1 is not None) and (t1_2 is not None):
                cl11 = np.zeros([len(t1_1), len(t1_2), len(ls)])
                autocorr = n1 == n2
                for i1, (t11, dn1) in enumerate(zip(t1_1, t1dn_1)):
                    for i2, (t12, dn2) in enumerate(zip(t1_2, t1dn_2)):
                        if autocorr and i2 < i1:
                            cl11[i1, i2] = cl11[i2, i1]
                        else:
                            pk = pkd[f'pk_{dn1}{dn2}']
                            if pk is not None:
                                cl = ccl.angular_cl(
                                    cosmo, t11, t12, ls, p_of_k_a=pk,
                                    limber_integration_method=self.limber_integ) * clm['pixbeam']
                            else:
                                cl = np.zeros_like(ls)
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
                if (cl00 is not None):
                    clb00 = self._eval_interp_cl(cl00, clm['l_bpw'], clm['w_bpw'])
                else:
                    clb00 = None
                clbs_00.append(clb00)
            for clm, cl01, cl10 in zip(self.cl_meta, cls_01, cls_10):
                # 01: unbiased x biased
                if (cl01 is not None):
                    clb01 = []
                    for cl in cl01:
                        clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                        clb01.append(clb)
                    clb01 = np.array(clb01)
                else:
                    clb01 = None
                clbs_01.append(clb01)
                # 10: biased x unbiased
                if clm['bin_1'] == clm['bin_2']:
                    clbs_10.append(clb01)
                else:
                    if (cl10 is not None):
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
                    if (cl11 is not None):
                        clb11 = np.zeros((cl11.shape[0], cl11.shape[1], len(clm['l_eff'])))
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

        return {'cl00': clbs_00, 'cl01': clbs_01, 'cl10': clbs_10, 'cl11': clbs_11}

    def _get_nz(self, cosmo, name, **pars):
        """ Get redshift distribution for a given tracer.
        Applies shift and width nuisance parameters if needed.
        """
        z = self.bin_properties[name]['z_fid']
        nz = interp1d(z, self.bin_properties[name]['nz_fid'], kind='cubic',
                      bounds_error=False, fill_value=0)
        zm = self.bin_properties[name]['zmean_fid']
        dz = 0.
        wz = 1.
        jacob = 1
        if (self.nz_model == 'NzShift') or (self.nz_model == 'NzShiftWidth'):
            dz = pars.get(self.input_params_prefix + '_' + name + '_dz', 0.)
        if (self.nz_model == 'NzShiftWidth') or (self.nz_model == 'NzWidth'):
            wz = pars.get(self.input_params_prefix + '_' + name + '_wz', 1.)
            jacob = wz
        # NzShiftParam parametrized as z_true - z_false = f(z_true)
        # These were used in 2210.13434
        if self.nz_model == 'NzShiftParamLinear':
            A = pars.get(self.input_params_prefix + '_A_Nz', 0)
            B = pars.get(self.input_params_prefix + '_B_Nz', 0)
            dz = A + B * z
            jacob = (1 - B)
        elif self.nz_model == 'NzShiftParamLinearPerSurvey':
            survey = name.split('__')[0]
            A = pars.get(self.input_params_prefix + '_' + survey + '_A_Nz', 0)
            B = pars.get(self.input_params_prefix + '_' + survey + '_B_Nz', 0)
            dz = A + B * z
            jacob = (1 - B)
        z_out = (z - dz - zm) * wz + zm
        # dn/dzt = dzf/dzt|_zt * dn/dzf|_zt
        nz_out = jacob * nz(z_out)
        return (z, nz_out)

    def _get_ia_bias(self, cosmo, name, **pars):
        """ Intrinsic alignment.
        """
        # The amplitudes are multiplied in cl_like/cl_final.py with the galaxy
        # bias parameters.
        if self.ia_model == 'IANone':
            return None

        z = self.bin_properties[name]['z_fid']
        if self.ia_model == 'IAPerBin':
            A_IA = np.ones_like(z)
        elif self.ia_model == 'IADESY1':
            A0 = 1
            eta = pars[self.input_params_prefix + '_eta_IA']
            A_IA = A0 * ((1+z)/1.62)**eta
        elif self.ia_model == 'IADESY1_PerSurvey':
            # This assumes that name = survey__zbin
            survey = name.split('__')[0]
            A0 = 1
            eta = pars[self.input_params_prefix + '_' + survey + '_eta_IA']
            A_IA = A0 * ((1+z)/1.62)**eta
        else:
            raise LoggedError(self.log, f"Unknown IA model {self.ia_model}")
        return (z, A_IA)

    def _add_pixbeam_to_cl_meta(self):
        # Pixel window function product for each power spectrum
        for clm in self.cl_meta:
            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample
            beam = np.ones(ls.size)
            for nside in [clm['nside_1'], clm['nside_2']]:
                if nside is not None:
                    beam *= beam_hpix(ls, nside)
            clm['pixbeam'] = beam

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30

        l_min_sample = []
        l_max_sample = []
        for clm in self.cl_meta:
            l_bpw = clm['l_bpw']
            l_min_sample.append(l_bpw.min())
            l_max_sample.append(l_bpw.max())

        l_min_sample = np.min(l_min_sample)
        l_max_sample = np.max(l_max_sample)

        if l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = l_min_sample
        nl_sample = int(np.log10(l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if l_min_sample == 0:
            l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            l_sample = l_sample

        return l_sample

    def get_can_provide(self):
        return ["ia_model"]

    def get_ia_model(self):
        return self.ia_model
