import cl_like as cll
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


def get_info(bias, A_sE9=True):
    data = "cl_like/tests/data/gc_kp_sh_linear_nuisances.fits.gz"
    bias_gc1_b1 = 1.0 if bias not in ['LagrangianPT', 'BaccoPT'] else 0.0
    info = {"params": {"A_sE9": 2.23,
                       "Omega_c": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.0,
                       "bias_gc1_b1": bias_gc1_b1,
                       "bias_gc1_b1p": 0.0,
                       "bias_gc1_b2": 0.0,
                       "bias_gc1_bs": 0.0,
                       "bias_gc1_bk2": 0.0,
                       "bias_sh1_m": 0.3,
                       "limber_gc1_dz": -0.1,
                       "limber_sh1_dz": -0.2,
                       "limber_sh1_eta_IA": 1,
                       "bias_sh1_A_IA": 0.1,
                       "bias_gc1_s": 2/5,
                       "sigma8": None},
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1_PerSurvey"},
                       "Pk": {"external": Pk,
                             "bias_model": bias,
                              "zmax_pks": 2.6},  # For baccoemu
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
                                                   "gc1": {"lmin": 20,
                                                           "mag_bias": True}},
                                      }
                           },
            "output": "dum",
            "debug": False}

    if not A_sE9:
        info["params"]["sigma8"] = 0.8098
        del info["params"]["A_sE9"]

    return info


@pytest.mark.parametrize('bias', ['Linear', 'EulerianPT', 'LagrangianPT',
                                  'BaccoPT'])
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

@pytest.mark.parametrize('case', ['IAPerBin', 'IADESY1', 'IADESY1_PerSurvey'])
def test_ia_models(case):
    info = get_info(bias="Linear")
    if case not in ['IAPerBin', 'IADESY1_PerSurvey']:
        # In this case, the params are already as they have to
        info["params"]["bias_A_IA"] = info["params"].pop("bias_sh1_A_IA")
        info["params"]["limber_eta_IA"] = info["params"].pop("limber_sh1_eta_IA")
    info["theory"]["limber"]["ia_model"] = case
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    cond = np.fabs(loglikes[0]) < 2E-3
    if case == 'IAPerBin':
        # IAPerBin has a constant ia_bias, which does not fit the generated
        # data.
        assert not cond
    else:
        assert cond

def test_shape_model():
    info = get_info(bias="Linear", A_sE9=False)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 2E-3

    info["theory"]["clfinal"]["shape_model"] = "ShapeNone"
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) > 2E-3


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
