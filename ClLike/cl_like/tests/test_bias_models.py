import cl_like as cll
import numpy as np
from cobaya.model import get_model
import pytest


@pytest.mark.parametrize('bias', ['Linear', 'EulerianPT', 'LagrangianPT'])
def test_dum(bias):
    data = "cl_like/tests/data/gc_kp_sh_linear.fits.gz"
    info = {"params": {"A_sE9": 2.23,
                       "Omega_c": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.0,
                       "cll_gc1_b1": 1.0,
                       "cll_gc1_b1p": 0.0,
                       "cll_gc1_b2": 0.0,
                       "cll_gc1_bs": 0.0},
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"}},
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
                                      "bias_model": bias}},
            "output": "dum",
            "debug": False}

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 2E-3
