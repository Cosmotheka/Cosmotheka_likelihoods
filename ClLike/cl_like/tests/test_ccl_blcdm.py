import cl_like as cll
from cl_like.ccl_blcdm import CCL_BLCDM
from cl_like.limber import Limber
from cl_like.power_spectrum import Pk
from cl_like.cl_final import ClFinal
import numpy as np
from cobaya.model import get_model
import pytest
import os
import shutil
import sacc
from classy import Class
from scipy.interpolate import interp1d


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")


def get_info(non_linear="halofit", nonlinear_model="muSigma", pars_smg=(0, 0)):
    # FIXME: kp x kp has been commented out because the scale factor up to
    # which we compute the growth factor and Pks is not too high and fails
    data = "cl_like/tests/data/gc_kp_sh_linear_nuisances.fits.gz"
    info = {"params": {"A_s": 2.23e-9,
                       "Omega_cdm": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "parameters_smg__1": pars_smg[0],  # dmu
                       "parameters_smg__2": pars_smg[1],  # dSigma
                       "expansion_smg": 0.7,    # DE, tuned
                       "bias_gc1_b1": 1.0,
                       "bias_gc1_b1p": 0.0,
                       "bias_gc1_b2": 0.0,
                       "bias_gc1_bs": 0.0,
                       "bias_sh1_m": 0.3,
                       "limber_gc1_dz": -0.1,
                       "limber_sh1_dz": -0.2,
                       "limber_sh1_eta_IA": 1,
                       "bias_sh1_A_IA": 0.1,
                       "sigma8": None},
            "theory": {"ccl_blcdm": {"external": CCL_BLCDM,
                                     "nonlinear_model": nonlinear_model,
                                     "classy_arguments": {
                                     "Omega_Lambda": 0,
                                     "Omega_fld": 0,
                                     "Omega_smg": -1,
                                     "non linear": non_linear,
                                     "gravity_model": "mgclass_fs",
                                     "expansion_model": "lcdm",
                                     "use_Sigma": "yes"}
                               },
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1_PerSurvey"},
                       "Pk": {"external": Pk,
                             "bias_model": "Linear"},
                       "clfinal": {"external": ClFinal,
                                   "input_params_prefix": "bias",
                                   "shape_model": "ShapeMultiplicative"}
                       },
            "likelihood": {"ClLike": {"external": cll.ClLike,
                                      "input_file": data,
                                      "bins": [{"name": "gc1"},
                                               {"name": "kp"},
                                               {"name": "sh1"}
                                               ],
                                      "twopoints": [{"bins": ["gc1", "gc1"]},
                                                    {"bins": ["gc1", "kp"]},
                                                    {"bins": ["gc1", "sh1"]},
                                                    #{"bins": ["kp", "kp"]},
                                                    {"bins": ["kp", "sh1"]},
                                                    {"bins": ["sh1", "sh1"]}
                                                    ],
                                      "defaults": {"kmax": 0.5,
                                                   "lmin": 0,
                                                   "lmax": 2000,
                                                   "gc1": {"lmin": 20}},
                                      }
                           },
            "output": "dum",
            "debug": False}

    if non_linear == "hmcode":
        # HMCode will not be able to match the Pk as much as halofit because
        # the data was generated with halofit and not hmcode
        info["likelihood"]["ClLike"]["defaults"]["kmax"] = 0.15

    return info


def get_pk_lin(cosmo, pair, z):
    Tks = cosmo.get_transfer(z)
    kTk = Tks['k (h/Mpc)'] * cosmo.h()

    Pk = cosmo.get_primordial()
    lpPk = np.log(Pk['P_scalar(k)'])
    lkPk = np.log(Pk['k [1/Mpc]'])

    kPk = kTk
    pPk = np.exp(interp1d(lkPk, lpPk, kind="cubic",
                          fill_value="extrapolate")(np.log(kTk)))
    pPk = (2 * np.pi ** 2) * pPk / kTk ** 3

    H0 = cosmo.Hubble(0)
    Omega_m = cosmo.Omega_m()

    Tk = []
    for ti in pair:
        if (ti == 'delta_matter'):
            iTk = Tks['d_m']
        elif (ti.lower() == 'weyl'):
            iTk = (Tks['phi'] + Tks['psi']) / 2.
            # Correct by the GR W->d factor. CCL devides by it
            iTk *= (- kTk**2 * 2 / 3 / (H0)**2 / Omega_m / (1 + z))
        else:
            raise ValueError(f'Tracer {ti} not implemented')

        Tk.append(iTk)

    return kTk, Tk[0] * Tk[1] * pPk


@pytest.mark.parametrize('non_linear', ['halofit', 'hmcode'])
def test_dum(non_linear):
    if non_linear == "halofit":
        # halofit: this should match current data at high precission.
        chi2_allowed = 2E-3
    else:
        # hmcode: this will not be able to match the Pk as much as halofit
        # because the data was generated with halofit and not hmcode
        chi2_allowed = 3

    info = get_info(non_linear)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.fabs(loglikes[0]) < chi2_allowed


@pytest.mark.parametrize('non_linear', ['halofit', 'hmcode'])
def test_timing(non_linear):
    info = get_info(non_linear)
    info["timing"] = True
    model = get_model(info)
    model.measure_and_set_speeds(5)
    model.dump_timing()
    time = np.sum([c.timer.get_time_avg() for c in model.components])
    # Before restructuring, the average evaluation time was ~0.54s in my laptop
    # After the restructuration, it went to 0.56s.
    assert time < 0.6

@pytest.mark.parametrize('pars_smg', [(0, 1), (1, 0), (1, 1)])
def test_blcdm_pars(pars_smg):
    info = get_info(non_linear="hmcode", pars_smg=pars_smg)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert not np.fabs(loglikes[0]) < 3

    if pars_smg == (0, 1):
        # Remove weak lensing if Sigma != 1
        info['likelihood']["ClLike"]['bins'] = [{"name": 'gc1'}]
        info['likelihood']["ClLike"]['twopoints'] = [{"bins": ["gc1", "gc1"]}]
        model = get_model(info)
        loglikes, derived = model.loglikes()
        assert np.fabs(loglikes[0]) < 3


@pytest.mark.parametrize('non_linear', ['Linear', 'muSigma'])
@pytest.mark.parametrize('pars_smg', [(0, 0), (1, 1)])
def test_get_pks_muSigma(non_linear, pars_smg):
    # We compare with the Pk obtained through the tranfers functions
    info = get_info(nonlinear_model=non_linear, pars_smg=pars_smg)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    pk_data = model.provider.get_Pk()['pk_data']

    pars = info["params"]

    cosmo = Class()
    cosmo.set({'output': 'mPk, mTk', 'z_max_pk': 4, 'P_k_max_1/Mpc': 2,
               'hmcode_min_k_max': 35, 'output_background_smg': 10})
    cosmo.set(info['theory']['ccl_blcdm']['classy_arguments'])
    cosmo.set({"A_s": pars["A_s"],
               "Omega_cdm": pars["Omega_cdm"],
               "Omega_b": pars["Omega_b"],
               "h": pars["h"],
               "n_s": pars["n_s"],
               "parameters_smg": ",".join([str(pars["parameters_smg__1"]),
                                           str(pars["parameters_smg__2"])]),
               "expansion_smg": pars["expansion_smg"]})
    cosmo.compute()

    if non_linear == 'muSigma':
        # In this case we will be comparing a non_linear pk vs a linear pk.
        # These only match at large scales
        kmax = 0.01
        rdev_mm = 1e-2  # Relaxing the condition for matter
    else:
        kmax = 2
        rdev_mm = 1e-4

    for z in np.linspace(0, 4, 10):
        k, pk_mm = get_pk_lin(cosmo, ["delta_matter", "delta_matter"], z)
        pk_mm = pk_mm[k < kmax]
        k = k[k < kmax]
        pk_mm2 = pk_data["pk_mm"].eval(k, 1/(1+z))

        # from matplotlib import pyplot as plt
        # plt.loglog(k, pk_mm/pk_mm2)
        # plt.loglog(k, pk_mm2, ls='--')
        # plt.title(f"z = {z}")
        # plt.show()
        # plt.close()
        assert pk_mm == pytest.approx(pk_mm2, rdev_mm)

        # Not sure why the precission for Pk_mw is only 1%
        k, pk_mw = get_pk_lin(cosmo, ["delta_matter", "weyl"], z)
        pk_mw = pk_mw[k < kmax]
        k = k[k < kmax]
        pk_mw2 = pk_data["pk_mw"].eval(k, 1/(1+z))
        assert pk_mw == pytest.approx(pk_mw2, 1e-2)

        # Not sure why the precission for Pk_ww is only 1%
        k, pk_ww = get_pk_lin(cosmo, ["weyl", "weyl"], z)
        pk_ww = pk_ww[k < kmax]
        k = k[k < kmax]
        pk_ww2 = pk_data["pk_ww"].eval(k, 1/(1+z))
        assert pk_ww == pytest.approx(pk_ww2, 1e-2)

    cosmo.struct_cleanup()
