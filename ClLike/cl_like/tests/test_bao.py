import cl_like as cll
from cl_like.bao_like import BAOLike
import numpy as np
from cobaya.model import get_model
import pytest
import os
import shutil


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")


def get_info():
    info = {"params": {"A_sE9": 2.1265,
                       "Omega_c": 0.26,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.15,
                       "T_CMB": 2.7255,
                       "sigma8": None},
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       },
            "likelihood": {"BAOLike": {"external": BAOLike,
                                       "bins": [0, 1, 2],
                                      }
                           },
            "debug": False}

    return info

def test_dum():
    info = get_info()

    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.isfinite(loglikes)
    chi2 = -2 * loglikes[0]
    assert (chi2 >= 0) and (chi2 < 50)
