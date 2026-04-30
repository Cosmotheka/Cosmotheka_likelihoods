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
import time


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")


def get_info(T_AGN, use_S8=False, sigma8_to_As='ccl'):
    # Generated with T_AGN = 7.8
    data = "" if "ClLike" in os.getcwd() else "ClLike/"
    data += "cl_like/tests/data/sh_baryons_hmcode2020_TAGN.fits.gz"

    extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
                                 }}


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
                       "HMCode_logT_AGN": T_AGN,
                       # Derived
                       "sigma8": None,
                       "S8": None,
                       },
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "camb",
                               "baryons_pk": "nobaryons",
                               "sigma8_to_As": sigma8_to_As,
                               "ccl_arguments": {"extra_parameters":
                                                 extra_parameters}
                               },
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1_PerSurvey"},
                       "Pk": {"external": Pk, "bias_model": "Linear"},
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
            "debug": False}

    if use_S8:
        # From mk file: 0.80010503150849170036
        # From derived: 0.8001052150162911
        info['params']['S8'] = 0.80010503150849170036
        del info['params']['A_sE9']
        info['params']['A_s'] = None

    return info


def test_dum():
    info = get_info(T_AGN=9)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.fabs(loglikes[0]) > 10


    # Mock data generated with T_AGN = 7.8
    info = get_info(T_AGN=7.8)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    # from matplotlib import pyplot as plt
    # sd = model.likelihood['ClLike'].get_cl_data_sacc()
    # st = model.likelihood['ClLike'].get_cl_theory_sacc()
    # f, ax = plt.subplots(3, 3)

    # for trs in st.get_tracer_combinations():
    #     i = int(trs[0][-1])
    #     j = int(trs[1][-1])
    #     ell, cld, cov = sd.get_ell_cl('cl_ee', *trs, return_cov=True)
    #     # ax[i, j].errorbar(ell, cld, label=f'{trs}', fmt='.k')
    #     ell, clt = st.get_ell_cl('cl_ee', *trs)
    #     # ax[i, j].errorbar(ell, clt, fmt='.')
    #     #ax[i, j].loglog()
    #     ax[i, j].semilogx(ell, cld/clt-1, label=f'{trs}')
    #     #err = np.sqrt(np.diag(cov))
    #     #ax[i, j].semilogx(ell, (cld-clt)/err, label=f'{trs}')
    #     ax[i, j].legend()
    # plt.show()

    # For some reason I cannot push it lower than this.
    assert np.fabs(loglikes[0]) < 0.25


@pytest.mark.parametrize('sigma8_to_As', ['ccl', 'baccoemu'])
def test_S8_input(sigma8_to_As):
    # Mock data generated with T_AGN = 7.8
    info = get_info(T_AGN=7.8, use_S8=True, sigma8_to_As=sigma8_to_As)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert derived[1] == pytest.approx(2.1265e-9, rel=0.01)

    # For some reason I cannot push it lower than this.
    if sigma8_to_As == 'ccl':
        assert np.fabs(loglikes[0]) < 0.25
    else:
        # With baccoemu there is a small difference
        assert np.fabs(loglikes[0]) < 2

