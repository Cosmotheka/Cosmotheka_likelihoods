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
                                      "defaults": {"lmin": 0,
                                                   "lmax": 8192},
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

    assert np.fabs(loglikes[0]) < 0.2

def test_sigma8_error(ptc):
    # Test that one can only sample on A_s
    info = get_info(A_sE9=False)
    model = get_model(info)
    lkl = model.likelihood['ClLike']
    loglikes, derived = model.loglikes()
    cosmo = lkl.provider.get_CCL()['cosmo']

    print(model.log.log)

    with pytest.raises(ValueError):
        ptc.update_pk(cosmo, 0, 0)
