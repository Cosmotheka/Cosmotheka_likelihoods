import cl_like as cll
from cl_like.limber import Limber
from cl_like.power_spectrum import Pk
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


def get_info(bias, A_sE9=True):
    data = "cl_like/tests/data/gc_kp_sh_linear_nuisances.fits.gz"
    info = {"params": {"A_sE9": 2.23,
                       "Omega_c": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.0,
                       "cll_gc1_b1": 1.0,
                       "cll_gc1_b1p": 0.0,
                       "cll_gc1_b2": 0.0,
                       "cll_gc1_bs": 0.0,
                       "limber_gc1_dz": -0.1,
                       "limber_sh1_dz": -0.2,
                       "cll_sh1_m": 0.3,
                       "sigma8": None},
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber"},
                       "Pk": {"external": Pk,
                              "bias_model": bias}
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
                                      "input_params_prefix": "cll",
                                      }
                           },
            "output": "dum",
            "debug": False}

    if not A_sE9:
        info["params"]["sigma8"] = 0.8098
        del info["params"]["A_sE9"]

    return info


@pytest.mark.parametrize('bias', ['Linear', 'EulerianPT', 'LagrangianPT'])
def test_dum(bias):
    info = get_info(bias)

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 2E-3


# TODO: Move this test to another file or rename this one
def test_sigma8():
    info = get_info(bias="Linear", A_sE9=False)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 2E-3

    del info["params"]["sigma8"]
    with pytest.raises(ValueError):
        model = get_model(info)


def test_timing():
    info = get_info(bias="Linear", A_sE9=False)
    info["timing"] = True
    model = get_model(info)
    model.measure_and_set_speeds(10)
    model.dump_timing()
    time = np.sum([c.timer.get_time_avg() for c in model.components])
    # Before restructuring, the average evaluation time was ~0.54s in my laptop
    # After the restructuration, it went to 0.56s.
    assert time < 0.6
