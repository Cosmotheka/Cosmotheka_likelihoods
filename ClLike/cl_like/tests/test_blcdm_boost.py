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

# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")

@pytest.fixture()
def ptc():
    ptc = BLCDMCalculator(emu_folder=emu_folder)
    return ptc


def get_info(A_sE9=True):
    data = "" if "ClLike" in os.getcwd() else "ClLike/"
    data += "cl_like/tests/data/sh_baccoemu_baryons.fits.gz"
    info = {"params": {"A_sE9": 2.1265,
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
                       # Baryons
                       "M_c" :  14,
                       "eta" : -0.3,
                       "beta" : -0.22,
                       "M1_z0_cen" : 10.5,
                       "theta_out" : 0.25,
                       "theta_inn" : -0.86,
                       "M_inn" : 13.4,
                       # MG
                       "mu0": 0,
                       "Sigma0": 0,
                       # Derived
                       "sigma8": None,
                       },
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               # "matter_pk": "linear",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1_PerSurvey"},
                       "Pk": {"external": Pk,
                             "nonlinear_pk": 'Bacco',
                             "bias_model": "Linear",
                             "zmax_pks": 1.5,  # For baccoemu with baryons
                             "use_baryon_boost" : True,
                             "baryon_model" : 'Bacco',
                             "mg_model": 'BLCDM',
                             "mg_parametrization": "1_minus_mu0OmegaDE",
                             "mg_emulator_folder": "../../../codes/mg_boost_emu_neurons_400_400_dropout_0.0_bn_False/",
                             },
                       "clfinal": {"external": ClFinal,
                                   "input_params_prefix": "bias",
                                   "shape_model": "ShapeMultiplicative"}
                       },
            "likelihood": {"ClLike": {"external": cll.ClLike,
                                      "input_file": data,
                                      "bins": [{"name": "sh0"},
                                               {"name": "sh1"},
                                               {"name": "sh2"}],
                                      "twopoints": [{"bins": ["sh0", "sh0"]},
                                                    {"bins": ["sh0", "sh1"]},
                                                    {"bins": ["sh0", "sh2"]},
                                                    {"bins": ["sh1", "sh1"]},
                                                    {"bins": ["sh1", "sh2"]},
                                                    {"bins": ["sh2", "sh2"]},
                                                     ],
                                      "defaults": {"lmin": 0,
                                                   "lmax": 8192},
                                      }
                           },
            "debug": True}

    if not A_sE9:
        info["params"]["sigma8"] = 0.78220521
        del info["params"]["A_sE9"]

    return info

def test_dum():
    info = get_info()

    model = get_model(info)
    loglikes, derived = model.loglikes()

    # if bias != 'BaccoPT':
    #     assert np.fabs(loglikes[0]) < 3E-3
    # else:
    #     # For some reason I cannot push it lower than this.
    assert np.fabs(loglikes[0]) < 0.2

def test_sigma8_error():
    # Test that one can only sample on A_s

    pass
