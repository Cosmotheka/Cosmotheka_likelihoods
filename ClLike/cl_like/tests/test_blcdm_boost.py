import cl_like as cll
from cl_like.limber import Limber
from cl_like.power_spectrum import Pk
from cl_like.cl_final import ClFinal
from cl_like.blcdm_boost import BLCDMCalculator
import numpy as np
from cobaya.model import get_model
import pytest
import os
import shutil
import sacc
import pyccl as ccl


EMU_FOLDER =  "../../../codes/mg_boost_emu_neurons_400_400_dropout_0.0_bn_False/"

if os.getenv("GITHUB_ACTIONS") == "true":
    pytest.skip("Skipping all tests in CI environment until the release of "
                "the emulator", allow_module_level=True)

# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")

@pytest.fixture()
def ptc():
    ptc = BLCDMCalculator(emu_folder=EMU_FOLDER,
                          parametrizaton='1_minus_mu0OmegaDE')
    return ptc

def get_Omega_de(model):
    cosmo = model.likelihood['ClLike'].provider.get_CCL()['cosmo']
    return cosmo.omega_x(1, 'dark_energy')


def get_info(baryons=False, A_sE9=True):
    data = "" if "ClLike" in os.getcwd() else "ClLike/"
    case_pars = {}
    pk_pars = {}

    if not baryons:
        data += "cl_like/tests/data/linear_baccopkmm_5x2pt.fits.gz"
        case_pars = {# gc0
                   "bias_gc0_b1": 1.2,
                   "bias_gc0_b1p": 0.0,
                   "bias_gc0_b2": 0.0,
                   "bias_gc0_bs": 0.0,
                   "bias_gc0_bk2": 0.0,
                   "limber_gc0_dz": 0.1,
                   # gc1
                   "bias_gc1_b1": 1.4,
                   "bias_gc1_b1p": 0.0,
                   "bias_gc1_b2": 0.0,
                   "bias_gc1_bs": 0.0,
                   "bias_gc1_bk2": 0.0,
                   "limber_gc1_dz": 0.15,
                   "bias_gc1_s": 2/5,
        }

        bins = [{"name": "gc0"},
                {"name": "gc1"},
                {"name": "sh0"},
                {"name": "sh1"},
                {"name": "sh2"},
                {"name": "kp"},
               ]

        twopoints = [{"bins": ["gc0", "gc0"]},
                     {"bins": ["gc1", "gc1"]},

                     {"bins": ["gc0", "sh0"]},
                     {"bins": ["gc0", "sh1"]},
                     {"bins": ["gc0", "sh2"]},
                     {"bins": ["gc1", "sh0"]},
                     {"bins": ["gc1", "sh1"]},
                     {"bins": ["gc1", "sh2"]},

                     {"bins": ["gc0", "kp"]},
                     {"bins": ["gc1", "kp"]},

                     {"bins": ["sh0", "sh0"]},
                     {"bins": ["sh0", "sh1"]},
                     {"bins": ["sh0", "sh2"]},
                     {"bins": ["sh1", "sh1"]},
                     {"bins": ["sh1", "sh2"]},
                     {"bins": ["sh2", "sh2"]},

                     {"bins": ["sh0", "kp"]},
                     {"bins": ["sh1", "kp"]},
                     {"bins": ["sh2", "kp"]},

                     {"bins": ["kp", "kp"]},
                     ]
    else:
        data += "cl_like/tests/data/sh_baccoemu_baryons.fits.gz"
        case_pars = {# Baryons
                    "M_c" :  14,
                    "eta" : -0.3,
                    "beta" : -0.22,
                    "M1_z0_cen" : 10.5,
                    "theta_out" : 0.25,
                    "theta_inn" : -0.86,
                    "M_inn" : 13.4,
                    }
        pk_pars = {
               "use_baryon_boost" : True,
               "baryon_model" : 'Bacco',
        }

        bins = [{"name": "sh0"},
                {"name": "sh1"},
                {"name": "sh2"}]

        twopoints = [{"bins": ["sh0", "sh0"]},
                     {"bins": ["sh0", "sh1"]},
                     {"bins": ["sh0", "sh2"]},
                     {"bins": ["sh1", "sh1"]},
                     {"bins": ["sh1", "sh2"]},
                     {"bins": ["sh2", "sh2"]},
                     ]

    params = {"A_sE9": 2.1265,
                       "Omega_c": 0.26,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.15,
                       # m
                       "bias_sh0_m": 0.1,
                       "bias_sh1_m": 0.3,
                       "bias_sh2_m": 0.5,
                       # dz
                       "limber_sh0_dz": 0.2,
                       "limber_sh1_dz": 0.4,
                       "limber_sh2_dz": 0.6,
                       # IA
                       "limber_sh0_eta_IA": 1,
                       "bias_sh0_A_IA": 0.1,
                       "limber_sh1_eta_IA": 1,
                       "bias_sh1_A_IA": 0.1,
                       "limber_sh2_eta_IA": 1,
                       "bias_sh2_A_IA": 0.1,
                       # MG
                       "mu0": 0,
                       "Sigma0": 0,
                       # Derived
                       "sigma8": None,
                       }
    params.update(case_pars)


    pk_dict = {"external": Pk,
               "nonlinear_pk": 'Bacco',
               "bias_model": "Linear",
               "zmax_pks": 1.5,  # For baccoemu with baryons
               "use_mg_boost": True,
               "mg_model": 'BLCDM',
               "mg_parametrization": "1_minus_mu0OmegaDE",
               "mg_emulator_folder": EMU_FOLDER,
               }
    pk_dict.update(pk_pars)


    info = {"params": params,
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               # "matter_pk": "linear",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1_PerSurvey"},
                       "Pk": pk_dict,
                       "clfinal": {"external": ClFinal,
                                   "input_params_prefix": "bias",
                                   "shape_model": "ShapeMultiplicative"}
                       },
            "likelihood": {"ClLike": {"external": cll.ClLike,
                                      "input_file": data,
                                      "bins": bins,
                                      "twopoints": twopoints,
                                      "defaults": {"kmax": 0.15,
                                                   "lmin": 0,
                                                   "lmax": 1000},
                                      }
                           },
            "debug": False}

    if not A_sE9:
        info["params"]["sigma8"] = 0.78220521
        del info["params"]["A_sE9"]

    return info

@pytest.mark.parametrize('baryons', [False, True])
def test_dum(baryons):
    info = get_info(baryons)

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)

    # lmax = 1000 and Delta chi2 < 1 because the kmax = 3 h/Mpc of the emulator
    assert np.fabs(loglikes[0]) < 1

    info['params']['mu0'] = 0.3
    info['params']['Sigma0'] = 10


    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) > 20

    # Check use_mg_boost = False means no MG i.e. default pk is used
    info = get_info(baryons)
    info['theory']['Pk']['use_mg_boost'] = False
    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.fabs(loglikes[0]) < 0.2


def test_sigma8_error(ptc):
    # Test that one can only sample on A_s
    info = get_info(A_sE9=False)
    model = get_model(info)
    lkl = model.likelihood['ClLike']
    loglikes, derived = model.loglikes()
    cosmo = lkl.provider.get_CCL()['cosmo']

    with pytest.raises(ValueError):
        ptc.update_pk(cosmo, 0, 0)


def test_pk_data():
    info = get_info()
    model_lcdm = get_model(info)
    lkl_lcdm = model_lcdm.likelihood['ClLike']
    loglikes, derived = model_lcdm.loglikes()

    info['params']['mu0'] = 0.3
    info['params']['Sigma0'] = 2
    model_mg = get_model(info)
    lkl_mg = model_mg.likelihood['ClLike']
    loglikes, derived = model_mg.loglikes()

    Omega_de = get_Omega_de(model_lcdm)

    pkd_lcdm = lkl_lcdm.provider.get_Pk()['pk_data']
    pkd_mg = lkl_mg.provider.get_Pk()['pk_data']

    Qk = pkd_mg['pk_mm'](k=1, a=1) / pkd_lcdm['pk_mw'](k=1, a=1)
    Sigma = pkd_mg['pk_wm'](k=1, a=1) / pkd_lcdm['pk_mw'](k=1, a=1)
    Sigma2 = pkd_mg['pk_ww'](k=1, a=1) / pkd_lcdm['pk_ww'](k=1, a=1)

    assert np.fabs(Qk - 1) > 1e-2
    assert Sigma2/Qk == pytest.approx((Sigma/Qk)**2, rel=1e-5)
    assert Sigma/Qk - 1 == pytest.approx(2 * Omega_de, rel=1e-5)

    info['params']['mu0'] = 0.
    model_mg = get_model(info)
    lkl_mg = model_mg.likelihood['ClLike']
    loglikes, derived = model_mg.loglikes()
    pkd_mg = lkl_mg.provider.get_Pk()['pk_data']

    Qk = pkd_mg['pk_mm'](k=1, a=1) / pkd_lcdm['pk_mm'](k=1, a=1)
    assert Qk - 1 == pytest.approx(0., rel=1e-2)


def test_Wk():
    info = get_info()
    info['params']['mu0'] = 0.3
    info['params']['Sigma0'] = 2
    model = get_model(info)
    lkl = model.likelihood['ClLike']
    loglikes, derived = model.loglikes()

    pkd = lkl.provider.get_Pk()['pk_data']
    Qk_mm = pkd['Qk_mm']
    Qk_mw = pkd['Qk_mw']
    Qk_wm = pkd['Qk_wm']
    Qk_ww = pkd['Qk_ww']

    a_arr, lk_arr, qkmm_arr = Qk_mm.get_spline_arrays()
    _, _, qkmw_arr = Qk_mw.get_spline_arrays()
    _, _, qkwm_arr = Qk_wm.get_spline_arrays()
    _, _, qkww_arr = Qk_ww.get_spline_arrays()

    # qk_arr shape [a_arr.size, lk_arr.size]. Note that a=1 is a[-1]

    # For BLCDM: m -> w implies multiplying by Sigma
    # 1. Check that Sigma only depends on time, not k
    assert qkmm_arr[:, 0] / qkmw_arr[:, 0] == pytest.approx(qkmm_arr[:, -1] /
                                                            qkmw_arr[:, -1],
                                                            rel=1e-5)

    # 2. Check that mm/ww = Sigma^2
    Sigma_a = qkwm_arr[:, 0] / qkmm_arr[:, 0]

    assert qkww_arr[:, 0] / qkmm_arr[:, 0] == pytest.approx(Sigma_a**2, rel=1e-5)

    # 3. Check that wm == mw
    assert np.all(qkwm_arr == qkmw_arr)

    # 4. Check that Sigma0 -1 = 2 (input). Note that a=1 is the last array
    # entry.
    # This tests will need to be generalized if parametrization changes
    Omega_de = get_Omega_de(model)
    assert Sigma_a[-1] - 1 == pytest.approx(2 * Omega_de, rel=1e-5)


def test_mu_Sigma_a():
    info = get_info()
    model = get_model(info)
    lkl = model.likelihood['ClLike']
    loglikes, derived = model.loglikes()
    cosmo = lkl.provider.get_CCL()['cosmo']

    # This tests will need to be generalized if parametrization changes
    ptc = BLCDMCalculator(emu_folder=EMU_FOLDER,
                          parametrizaton='1_minus_mu0OmegaDE')

    ptc.update_pk(cosmo, 0, 0)
    mu_a = ptc.get_mu_a()
    Sigma_a = ptc.get_Sigma_a()

    assert np.all(Sigma_a == 1)
    assert np.all(Sigma_a == mu_a)

    # Check that the internal values of Sigma0 and mu0 are correct
    ptc.update_pk(cosmo, 0.3, 2)
    mu_a = ptc.get_mu_a(a_arr=1)
    Sigma_a = ptc.get_Sigma_a(a_arr=1)

    Omega_de = get_Omega_de(model)

    assert Sigma_a - 1 == pytest.approx(2 * Omega_de, rel=1e-5)
    assert mu_a - 1 == pytest.approx(0.3 * Omega_de, rel=1e-5)

    # Check that you can pass other values of mu0 and Sigma0
    mu_a = ptc.get_mu_a(mu0=1)
    Sigma_a = ptc.get_Sigma_a(Sigma0=2)

    DE_factor = cosmo.omega_x(ptc.a_arr, 'dark_energy')
    assert Sigma_a-1 == pytest.approx(2 * DE_factor, rel=1e-5)
    assert (Sigma_a-1) / (mu_a-1) == pytest.approx(2, rel=1e-5)

    with pytest.raises(NotImplementedError):
        BLCDMCalculator(emu_folder=EMU_FOLDER, parametrizaton='tofail')

