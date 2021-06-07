import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d


class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # Set default parameters
        # Planck 2018: Table 2 of 1807.06209
        self.pars = {
            'h':       0.6736,
            'Omega_c': 0.2640,
            'Omega_b': 0.0493,
            'sigma8': 0.8111,
            'n_s': 0.9649,
            'w0': -1.0,
            'wa':  0.0
        }

        self.cosmo_ccl_planck = self.get_cosmo_ccl()

        # Copied from ccl/pk2d.py
        # These lines are needed to compute the Pk2D array
        self.nk = ccl.ccllib.get_pk_spline_nk(self.cosmo_ccl_planck.cosmo)
        self.na = ccl.ccllib.get_pk_spline_na(self.cosmo_ccl_planck.cosmo)
        self.a_arr, _ = ccl.ccllib.get_pk_spline_a(self.cosmo_ccl_planck.cosmo,
                                                   self.na, 0)
        lk_arr, _ = ccl.ccllib.get_pk_spline_lk(self.cosmo_ccl_planck.cosmo,
                                                self.nk, 0)
        self.k_arr = np.exp(lk_arr)

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass
        if 'growth_param' in param_dict:
            param_dict.pop('growth_param')
            if 'z_anchor' in param_dict:
                param_dict.pop('z_anchor')
            if 'spline' in param_dict:
                param_dict.pop('spline')
            for k in list(param_dict.keys()):
                if 'dpk' in k:
                    param_dict.pop(k)
        if 'omega_b' in param_dict:
            omega_b = param_dict.pop('omega_b')
            param_dict['Omega_b'] = omega_b / param_dict['h']**2
        if 'omega_c' in param_dict:
            omega_c = param_dict.pop('omega_c')
            param_dict['Omega_c'] = omega_c / param_dict['h']**2

        cosmo_ccl = ccl.Cosmology(**param_dict)
        return cosmo_ccl

    def get_sigma8(self):
        return self.get_sigma8z(0)

    def get_Omegam(self):
        Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        return Omm

    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8

    def get_growth_factor(self, a):
        return ccl.background.growth_factor(self.cosmo_ccl, a)

    def struct_cleanup(self):
        return

    def empty(self):
        return

    # Set up the dictionary
    def set(self, *pars_in, **kars):
        if ('A_s' in pars_in[0].keys()) and ('sigma8' in self.pars.keys()):
            self.pars.pop('sigma8')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        ### Check for parmeters of cl_cross_corr lkl
        if 'params_dir' in self.pars.keys():
            del[self.pars['params_dir']]
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        #
        if 'tau_reio' in self.pars.keys():
            raise ValueError('CCL does not read tau_reio. Remove it.')
        # Translate w_0, w_a CLASS vars to CCL w0, wa
        if 'w_0' in self.pars.keys():
            self.pars['w0'] = self.pars.pop('w_0')
        if 'w_a' in self.pars.keys():
            self.pars['wa'] = self.pars.pop('w_a')
        # Check that sigma8 or As are fixed with "growth_param"
        if (('A_s' in pars_in) or ('sigma8' in pars_in)) and \
                ("growth_param" in self.pars):
            raise RuntimeError("Remove 'A_s' and 'sigma8' when modifying \
                               growth")

        self.pars.update(kars)
        return True

    def compute(self, level=[]):
        self.cosmo_ccl = self.get_cosmo_ccl()
        # Modified growth part
        if 'growth_param' in self.pars:
            pkln = self.pk2D_arr(self.cosmo_ccl)
            pk_linear = {'a': self.a_arr, 'k': self.k_arr,
                         'delta_matter:delta_matter': pkln}
            p = self.cosmo_ccl.cosmo.params
            self.cosmo_ccl = ccl.CosmologyCalculator(Omega_c=p.Omega_c,
                                                     Omega_b=p.Omega_b, h=p.h,
                                                     sigma8=p.sigma8,
                                                     n_s=p.n_s,
                                                     pk_linear=pk_linear,
                                                     nonlinear_model='halofit')

        # ccl.sigma8(self.cosmo_ccl)  # David's suggestion
        return

    def get_current_derived_parameters(self, names):
        derived = {}
        for name in names:
            if name == 'sigma_8':
                value = self.get_sigma8()
            elif name == 'Omega_m':
                value = self.get_Omegam()
            elif name == 'S_8':
                value = self.get_S8()
            elif 'S8z_' in name:
                z = float(name.split('_')[-1])
                value = self.get_S8z(z)
            elif 'sigma8z_' in name:
                z = float(name.split('_')[-1])
                value = self.get_sigma8z(z)
            elif 'Dz_unnorm_' in name:
                z = float(name.split('_')[-1])
                value = self.get_Dz_unnorm(z)
            elif 'Dz_' in name:
                z = float(name.split('_')[-1])
                value = self.get_Dz(z)
            else:
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived

    def get_S8z(self, z):
        Omega_m = self.get_Omegam()
        sigma8z = self.get_sigma8z(z)
        S8z = sigma8z * (Omega_m/0.3)**(0.5)
        return S8z

    def get_sigma8z(self, z):
        if not 'growth_param' in self.pars:
            a = 1 / (1 + z)
            D = ccl.growth_factor(self.cosmo_ccl, a)
            sigma8z = D *  ccl.sigma8(self.cosmo_ccl)
        else:
            D = self.get_Dz_new_unnorm_over_D0_Planck_unnorm(z)
            sigma8z = D * ccl.sigma8(self.cosmo_ccl_planck)
        return sigma8z

    def get_Dz_unnorm(self, z):
        if not 'growth_param' in self.pars['growth_param']:
            a = 1 / (1 + z)
            result = ccl.growth_factor_unnorm(self.cosmo_ccl, a)
        else:
            D = self.get_Dz_new_unnorm_over_D0_Planck_unnorm(z)
            result = D * ccl.growth_factor_unnorm(self.cosmo_ccl_planck, 1)

        return result

    def get_Dz(self, z):
        """
        Return normalized D(z)
        """
        result = self.get_sigma8z(z) / self.get_sigma8z(0)

        return result

    def get_Dz_new_unnorm_over_D0_Planck_unnorm(self, z):
        """
        Return the D(z)_new / D(0)_Planck that will modifiy growth
        """
        if self.pars['growth_param'] == 'taylor':
            a = 1 / (1 + z)
            # D(z) = (dpk0 + dpk1 * (1 - a) + ... ) * D_Planck(z)
            result = 0
            i = 0
            while True:
                pname = 'dpk' + str(i)
                if pname not in self.pars:
                    break
                dpki = self.pars[pname]
                result += dpki / np.math.factorial(i) * (1-a)**i
                i += 1
            result *= ccl.growth_factor(self.cosmo_ccl_planck, a)
        elif self.pars['growth_param'] == 'binning':
            # D(z) = D_binned(z)
            z_Dz = []
            for pname, pvalue in self.pars.items():
                if 'dpk' in pname:
                    z_pvalue = float(pname.split('_')[-1])
                    z_Dz.append((z_pvalue, pvalue))

            # Put a small value ~ 0 at really high z so that Dz is not
            # extrapolated to -infty
            a_anchor = 1e-4  # The lower value available in ccl for D(z)
            z_anchor = 1 / a_anchor - 1
            Dz_anchor = ccl.growth_factor(self.cosmo_ccl_planck, a_anchor)
            z_Dz.append((z_anchor, Dz_anchor))
            z_Dz = np.array(sorted(z_Dz)).T

            if 'spline' in self.pars:
                kind = self.pars['spline']
            else:
                kind = 'quadratic'
            # Interpolate in log-log space, as D ~ exp
            result = interp1d(np.log(z_Dz[0] + 1), np.log(z_Dz[1]),
                              kind=kind, fill_value='extrapolate',
                              assume_sorted=True)(np.log(z+1))
            result = np.exp(result)
        elif self.pars['growth_param'] == 'binning_softer':
            # D(z) = D_binned(z)
            z_Dz = []
            for pname, pvalue in self.pars.items():
                if 'dpk' in pname:
                    z_pvalue = float(pname.split('_')[-1])
                    z_Dz.append((z_pvalue, pvalue))

            # Put a small value ~ 0 at really high z so that Dz is not
            # extrapolated to -infty
            z_anchor = self.pars['z_anchor'] # z at which we go back to Planck's
            Dz_anchor = ccl.growth_factor(self.cosmo_ccl_planck, 1/(z_anchor + 1))
            z_Dz.append((z_anchor, Dz_anchor))
            z_Dz = np.array(sorted(z_Dz)).T
            if np.any(z < z_anchor):
                if 'spline' in self.pars:
                    kind = self.pars['spline']
                else:
                    kind = 'quadratic'

                logresult_intp1d = interp1d(np.log(z_Dz[0] + 1), np.log(z_Dz[1]),
                                  kind=kind, fill_value='extrapolate',
                                  assume_sorted=True)


            if (type(z) is int) or (type(z) is float):
                if z >= z_anchor:
                    result = ccl.growth_factor(self.cosmo_ccl_planck, 1/(z + 1))
                else:
                    result = np.exp(logresult_intp1d(np.log(z+1)))
            else:
                result = np.zeros_like(z)
                # Interpolate in log-log space, as D ~ exp
                result[z < z_anchor] = np.exp(logresult_intp1d(np.log(z[z < z_anchor]+1)))
                result[z >= z_anchor] = ccl.growth_factor(self.cosmo_ccl_planck, 1/(z[z >= z_anchor] + 1))
                # planck = ccl.growth_factor(self.cosmo_ccl_planck, 1/(z + 1))
                # np.savez('s8z_test.py', mod=result, planck=planck, z_Dz=z_Dz, z=z)
        else:
            raise ValueError(f'growth_param {self.pars["growth_param"]} not implemented.')

        return result

    def pk2D_arr(self, cosmo):
        z = 1/self.a_arr - 1
        D_new = self.get_Dz_new_unnorm_over_D0_Planck_unnorm(z)
        pk0 = ccl.linear_matter_power(cosmo, self.k_arr, 1)
        return D_new[:, None] ** 2 * pk0
