# last modified on 11/03/2024
from typing import Optional
from cobaya.likelihood import Likelihood
import sacc
import numpy as np
import pyccl as ccl
import rosatX
import xcell.mappers as xcm
import yaml
import matplotlib.pyplot as plt
import os
import pyatomdb
import pickle
import time


class xyshearLike(Likelihood):
    """
    """

    DES_bin_id: Optional[int]
    path_to_data: str
    tSZ_name: str
    mode: str
    lmin: int
    lmax: int
    rosat_data: str
    use_baryon_profile: bool

    def initialize(self):
        """
        """
        # Load the data
        self.sacc_file = sacc.Sacc.load_fits(f"{self.path_to_data}/cls_cov_{self.tSZ_name}.fits")
        #self.sacc_file.keep_selection(ell__gt=self.lmin, ell__lt=self.lmax)


        if self.DES_bin_id is not None:
            tracers_list = self.sacc_file.get_tracer_combinations()

            tr_tSZ_name = f"Planck__{self.tSZ_name}"
            tr_Xray_name = "ROSAT"
            tr_shear_name = f"DESY3wl__{self.DES_bin_id}"
            
            self.ells = []
            self.cls = []
            self.beam_array = []
            indices = []

            self.probes_order = []
            if "y" in self.mode.split("+"):
                if (tr_tSZ_name, tr_shear_name) in tracers_list:
                    t1, t2 = tr_tSZ_name, tr_shear_name
                else:
                    t1, t2 = tr_shear_name, tr_tSZ_name
                ells, cls_ys, ind_ys = self.sacc_file.get_ell_cl("cl_0e", t1, t2, return_ind=True)
                mask = (ells >= self.lmin) & (ells <= self.lmax)
                ells, cls_ys, ind_ys = ells[mask], cls_ys[mask], ind_ys[mask]
                self.ells.append(ells)
                self.cls.append(cls_ys)
                self.beam_array.append(self.beam(ells, mode="sy"))
                indices += list(ind_ys)

                self.probes_order.append(f"Planck__{self.DES_bin_id}")

            #if self.mode in ["x", "x+y"]:
            if "x" in self.mode.split("+"):
                if (tr_Xray_name, tr_shear_name) in tracers_list:
                    t1, t2 = tr_Xray_name, tr_shear_name
                else:
                    t1, t2 = tr_shear_name, tr_Xray_name
                ells, cls_xs, ind_xs = self.sacc_file.get_ell_cl("cl_0e", t1, t2, return_ind=True)
                mask = (ells >= 30) & (ells <= 2000)
                ells, cls_xs, ind_xs = ells[mask], cls_xs[mask], ind_xs[mask]
                self.ells.append(ells)
                self.cls.append(cls_xs)
                self.beam_array.append(self.beam(ells, mode="sx"))
                indices += list(ind_xs)

                self.probes_order.append(f"ROSAT__{self.DES_bin_id}")

            #if self.mode in ["x+xy", "y+xy", "x+y+xy"]:
            if "xy" in self.mode.split("+"):
                if (tr_Xray_name, tr_tSZ_name) in tracers_list:
                    t1, t2 = tr_Xray_name, tr_tSZ_name
                else:
                    t1, t2 = tr_tSZ_name, tr_Xray_name
                ells, cls_xy, ind_xy = self.sacc_file.get_ell_cl("cl_00", t1, t2, return_ind=True)
                mask = (ells >= self.lmin) & (ells <= self.lmax)
                ells, cls_xy, ind_xy = ells[mask], cls_xy[mask], ind_xy[mask]
                self.ells.append(ells)
                self.cls.append(cls_xy)
                self.beam_array.append(self.beam(ells, mode="xy"))
                indices += list(ind_xy)

                self.probes_order.append("ROSATxPlanck")

            self.cls = np.concatenate(self.cls) 
            self.cov = self.sacc_file.covariance
            self.cov = self.cov.keeping_indices(indices).covmat
            self.beam_array = np.concatenate(self.beam_array)

        else:
            tracers_list = self.sacc_file.get_tracer_combinations()
            tracers_select = []
            lrange = []
            self.probes_order = []
            for t1, t2 in tracers_list:
                #if "y" in self.mode:
                if "y" in self.mode.split("+"):
                    if ("Planck" in t1+t2) and ("DES" in t1+t2):
                        tracers_select.append((t1, t2))
                        lrange.append((self.lmin, self.lmax))
                        if not("Planck__0" in self.probes_order):
                            self.probes_order += ["Planck__0", "Planck__1", "Planck__2", "Planck__3"]
                        
                #if "x" in self.mode:
                if "x" in self.mode.split("+"):
                    if ("ROSAT" in t1+t2) and ("DES" in t1+t2):
                        tracers_select.append((t1, t2))
                        lrange.append((30, 2000))
                        if not("ROSAT__0" in self.probes_order):
                            self.probes_order += ["ROSAT__0", "ROSAT__1", "ROSAT__2", "ROSAT__3"]

                if "xy" in self.mode.split("+"):
                    if ("ROSAT" in t1+t2) and ("Planck" in t1+t2):
                        tracers_select.append((t1, t2))
                        lrange.append((self.lmin, self.lmax))
                        if not("ROSATxPlanck" in self.probes_order):
                            self.probes_order.append("ROSATxPlanck")

            nspec = len(tracers_select)
            indices = []
            self.ells = []
            self.cls = []
            self.beam_array = []
            for i, (tr1_name, tr2_name) in enumerate(tracers_select):
                if ("ROSAT" in tr1_name+tr2_name) and ("Planck" in tr1_name+tr2_name):
                    ells, cls_i, ind = self.sacc_file.get_ell_cl("cl_00", tr1_name, tr2_name, return_ind=True)
                else:
                    ells, cls_i, ind = self.sacc_file.get_ell_cl("cl_0e", tr1_name, tr2_name, return_ind=True)
                lmin, lmax = lrange[i]
                mask = (ells >= lmin) & (ells <= lmax)
                ells, cls_i, ind = ells[mask], cls_i[mask], ind[mask]

                indices += list(ind)

                self.ells.append(ells)
                self.cls.append(cls_i)

                if "ROSAT" in tr1_name+tr2_name:
                    beam = self.beam(ells)
                else:
                    beam = np.ones_like(ells)
                self.beam_array.append(beam)

            self.cov = self.sacc_file.covariance
            self.cov = self.cov.keeping_indices(indices).covmat

            self.cls = np.concatenate(self.cls)

            self.beam_array = np.concatenate(self.beam_array)
        self.mbias = [1-0.0063, 1-0.0198, 1-0.0241, 1-0.0369]
        self.cov_inv = np.linalg.inv(self.cov)

        self.dndz_dict = self.get_dndz()

        self.initialize_model()


    def beam(self, ell, mode="sx"):
        fwhm_hpx_amin_1024 = 60*41.7/1024
        fwhm_hpx_amin_2048 = 60*41.7/2048

        sigma_hpx_1024 = np.radians(fwhm_hpx_amin_1024/60)/2.355
        sigma_hpx_2048 = np.radians(fwhm_hpx_amin_2048/60)/2.355
        sigma_ROSAT = np.radians(1.8e0/60)/2.355

        if mode == "sx":
            sigma_tot_2 = sigma_ROSAT**2 + 2*sigma_hpx_1024**2
        if mode == "sy":
            sigma_tot_2 = sigma_hpx_1024**2 + sigma_hpx_2048**2
        if mode == "xy":
            sigma_tot_2 = sigma_ROSAT**2 + sigma_hpx_1024**2 + sigma_hpx_2048**2
        return np.exp(-0.5*sigma_tot_2*ell*(ell+1))

    # Necessary to estimate J(T) for Xray profile
    def get_spectrum(self):
            emin = 0.5
            emax = 2.0

            self.Zmetal = 0.3
            self.lines = True
            self.spec_pyatomdb = False
            self.wclump = True
            self.A_IA = None

            fname = f'{self.rosat_data}/J'
            if self.lines:
                fname += 'tot'
            else:
                fname += 'cont'
            fname += '_Z%.2lf' % self.Zmetal
            fname += '_Emin%.2lf' % emin
            fname += '_Emax%.2lf' % emax

            if self.spec_pyatomdb:
                fname += '_padb'
            fname += '.pck'

            print(fname)
            if os.path.isfile(fname):
                with open(fname, "rb") as f:
                    J = pickle.load(f)
            else:
                raise ValueError("J factor should be precomputed")

            self.J = J


    def initialize_model(self):
        """
        """
        self.get_spectrum()

        self.k_arr = np.geomspace(1e-4, 1e2, 256)
        self.a_arr = np.linspace(0.1, 1, 16)
        self.cosmology = ccl.Cosmology(
            Omega_c=0.26066676,
            Omega_b=0.048974682,
            h=0.6766,
            sigma8=0.8102,
            n_s=0.9665,
            matter_power_spectrum="linear"
        )
        halo_mass_definition = ccl.halos.MassDef200c
        halo_bias = ccl.halos.HaloBiasTinker10(mass_def=halo_mass_definition)
        halo_mass = ccl.halos.MassFuncTinker08(mass_def=halo_mass_definition)

        concentration = ccl.halos.ConcentrationDuffy08(mass_def=halo_mass_definition)

        #if "y" in self.mode:
        if ("y" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            # TSZ profile
            self.profile_tSZ = rosatX.profiles.HaloProfilePressureHE(
                mass_def=halo_mass_definition, 
                concentration=concentration,
                kind="n_electron"
            )

        #if "x" in self.mode:
        if ("x" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            # XRAY profile
            self.prof_dens = rosatX.profiles.HaloProfileDensityHE(
                mass_def=halo_mass_definition,
                concentration=concentration,
                kind="n_total"
            )
            self.prof_pres = rosatX.profiles.HaloProfilePressureHE(
                mass_def=halo_mass_definition,
                concentration=concentration,
                kind="n_total"
            )
            self.profile_Xray = rosatX.profiles.HaloProfileXray(
                mass_def=halo_mass_definition,
                Jinterp=self.J,
                dens=self.prof_dens,
                pres=self.prof_pres,
                fourier_approx=False,
                with_clumping=self.wclump
            )
            self.profile_Xray.update_precision_fftlog(n_per_decade=100)


        # SHEAR profile
        if self.use_baryon_profile:
            self.profile_nfw = rosatX.profiles.HaloProfileNFWBaryon(
                mass_def=halo_mass_definition,
                concentration=concentration
            )
        else:
            self.profile_nfw = ccl.halos.HaloProfileNFW(
                mass_def=halo_mass_definition, 
                concentration=concentration
            )

        self.hm_calc = ccl.halos.HMCalculator(
            mass_function=halo_mass, 
            halo_bias=halo_bias,
            mass_def=halo_mass_definition,
            log10M_max=15.,
            log10M_min=10.,
            nM=32
        )

        #if "y" in self.mode:
        if ("y" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            # TSZ tracer
            self.tracer_tSZ = ccl.tSZTracer(self.cosmology)

        #if "x" in self.mode:
        if ("x" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            # XRAY tracers
            self.tracer_Xray = rosatX.XrayTracer(self.cosmology)
            if self.A_IA is not None:
                ia = (np.linspace(0, 3, 1024),
                    self.A_IA * np.ones(1024))
            else:
                ia = None

        # SHEAR tracers 
        self.tracer_shear_dict = {}
        for bin_id, dndz in self.dndz_dict.items():
            self.tracer_shear_dict[bin_id] = ccl.WeakLensingTracer(
                self.cosmology,
                dndz=dndz,
                has_shear=True
            )
        
        self.prof_2pt = ccl.halos.Profile2pt()

    def get_dndz(self):
        """
        """
        path_to_inputs = xcm.__file__.replace("xcell/mappers/__init__.py", "input")

        dndz_dict = {}

        if self.DES_bin_id is not None:
            bin_ids = [self.DES_bin_id]
        else:
            bin_ids = [0, 1, 2, 3]

        for bin_id in bin_ids:
            config_path = f"{path_to_inputs}/DESY3wl__{bin_id}.yml"

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config["nside"] = 1024
            config["coords"] = "C"
            mapper = xcm.MapperDESY3wl(config)
            dndz = mapper.get_nz()

            dndz_dict[bin_id] = dndz

        return dndz_dict

    def update_parameters(self, **kwargs):
        """
        """
        pars_to_update = {k: v for k, v in kwargs.items() if k in self.get_requirements()}
        if ("y" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            self.profile_tSZ.update_parameters(**pars_to_update)
        if ("x" in self.mode.split("+")) or ("xy" in self.mode.split("+")):
            self.prof_dens.update_parameters(**pars_to_update)
            self.prof_pres.update_parameters(**pars_to_update)

        if self.use_baryon_profile:
            pars_baryons = {k:v for k,v in pars_to_update.items() if k != "alpha_T"}
            self.profile_nfw.update_parameters(**pars_baryons)


    def get_model(self, **kwargs):
        """
        """
        self.update_parameters(**kwargs)

        #if "y" in self.mode:
        if "y" in self.mode.split("+"):
            p_of_k_tSZ = ccl.halos.halomod_Pk2D(
                        self.cosmology,
                        self.hm_calc,
                        self.profile_nfw,
                        prof2=self.profile_tSZ,
                        prof_2pt=self.prof_2pt,
                        lk_arr=np.log(self.k_arr),
                        a_arr=self.a_arr
            )
        #if "x" in self.mode:
        if "x" in self.mode.split("+"):
            p_of_k_Xray = ccl.halos.halomod_Pk2D(
                        self.cosmology,
                        self.hm_calc,
                        self.profile_nfw,
                        prof2=self.profile_Xray,
                        prof_2pt=self.prof_2pt,
                        lk_arr=np.log(self.k_arr),
                        a_arr=self.a_arr
            )
        
        if "xy" in self.mode.split("+"):
            p_of_k_XtSZ = ccl.halos.halomod_Pk2D(
                self.cosmology,
                self.hm_calc,
                self.profile_Xray,
                prof2=self.profile_tSZ,
                prof_2pt=self.prof_2pt,
                lk_arr=np.log(self.k_arr),
                a_arr=self.a_arr
            )

        cl_vec = []
        for id_probe, probe in enumerate(self.probes_order):
            ell = self.ells[id_probe]
            if probe == "ROSATxPlanck":
                cl = ccl.angular_cl(
                    self.cosmology,
                    self.tracer_Xray,
                    self.tracer_tSZ,
                    ell,
                    p_of_k_a=p_of_k_XtSZ
                )
            else:
                exp, bin_id = probe.split("__")
                bin_id = int(bin_id)

                if exp == "Planck":
                    cl = ccl.angular_cl(
                        self.cosmology,
                        self.tracer_shear_dict[bin_id],
                        self.tracer_tSZ,
                        ell,
                        p_of_k_a=p_of_k_tSZ
                    )
                elif exp == "ROSAT":
                    cl = ccl.angular_cl(
                        self.cosmology,
                        self.tracer_shear_dict[bin_id],
                        self.tracer_Xray,
                        ell,
                        p_of_k_a=p_of_k_Xray
                    )

                cl *= self.mbias[bin_id]
            cl_vec.append(cl)

        return np.concatenate(cl_vec) * self.beam_array


    def get_requirements(self):
        return {
            par: None 
            for par in ["lMc", "gamma", "alpha_T", "eta_b",
                        "logTw0", "Tw1", "beta", "epsilon",
                        "alpha_Tz", "alpha_Tm", "gamma_T"]
        }

    def logp(self, **kwargs):
        
        model = self.get_model(**kwargs)
        res = self.cls - model
        
        chi2 = res @ self.cov_inv @ res
        return -0.5 * chi2
