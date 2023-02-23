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


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")


def get_info():
    data = "cl_like/tests/data/gc_kp_sh_linear_nuisances.fits.gz"
    info = {"params": {"A_s": 2.23e-9,
                       "Omega_cdm": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "parameters_smg__1": 0,  # dmu
                       "parameters_smg__2": 0,  # dSigma
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
                                     "nonlinear_model": "muSigma",
                                     "classy_arguments": {
                                     "Omega_Lambda": 0,
                                     "Omega_fld": 0,
                                     "Omega_smg": -1,
                                     "non linear": "hmcode",
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
                                               {"name": "sh1"}],
                                      "twopoints": [{"bins": ["gc1", "gc1"]},
                                                    {"bins": ["gc1", "kp"]},
                                                    {"bins": ["gc1", "sh1"]},
                                                    {"bins": ["kp", "kp"]},
                                                    {"bins": ["kp", "sh1"]},
                                                    {"bins": ["sh1", "sh1"]}],
                                      "defaults": {"kmax": 0.5,
                                                   "lmin": 0,
                                                   "lmax": 2000,
                                                   "gc1": {"lmin": 20}},
                                      }
                           },
            "output": "dum",
            "debug": True}

    return info


def test_dum():
    info = get_info()

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 2E-3


def test_timing():
    info = get_info()
    info["timing"] = True
    model = get_model(info)
    model.measure_and_set_speeds(3)
    model.dump_timing()
    time = np.sum([c.timer.get_time_avg() for c in model.components])
    # Before restructuring, the average evaluation time was ~0.54s in my laptop
    # After the restructuration, it went to 0.56s.
    assert time < 0.6
