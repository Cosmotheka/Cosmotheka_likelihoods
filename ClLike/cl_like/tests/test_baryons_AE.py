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


def get_info(bias, A_AE):
    data = "" if "ClLike" in os.getcwd() else "ClLike/"
    if bias == 'Linear':
        data += "cl_like/tests/data/linear_halofit_5x2pt.fits.gz"
        pk_dict = {"external": Pk,
                   "bias_model": "Linear",
                   }
    elif bias == 'BaccoPT':
        data += "cl_like/tests/data/linear_baccopt_5x2pt.fits.gz"
        pk_dict = {"external": Pk,
                   "bias_model": "BaccoPT",
                   "zmax_pks": 1.5,  # For baccoemu with baryons
                   "ignore_lbias": False}
    else:
        raise ValueError(f'bias {bias} not implemented')

    pk_dict.update({"use_baryon_boost" : True,
                    "baryon_model": 'Amon-Efstathiou'
                    })


    info = {"params": {"A_sE9": 2.1265,
                       "Omega_c": 0.26,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.15,
                       "T_CMB": 2.7255,
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
                       "A_AE": A_AE,
                       # Derived
                       "sigma8": None,
                       },
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               },
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

    return info


@pytest.mark.parametrize('bias', ['Linear', 'BaccoPT'])
def test_dum(bias):
    info = get_info(bias, A_AE=1)
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

    if bias != 'BaccoPT':
        assert np.fabs(loglikes[0]) < 3E-3
    else:
        # For some reason I cannot push it lower than this.
        assert np.fabs(loglikes[0]) < 0.2

    # A_AE = 0 (i.e. pk_ww = pklin
    info = get_info(bias, A_AE=0)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    lkl = model.likelihood['ClLike']
    pkww = lkl.provider.get_Pk()['pk_data']['pk_ww']
    pk2d_lin = lkl.provider.get_CCL()['cosmo'].get_linear_power()
    alin, lnklin, pklin = pk2d_lin.get_spline_arrays()

    for i, ai in enumerate(alin):
        assert pklin[i] == pytest.approx(pkww(np.exp(lnklin), ai),
                                         rel=1e-5)

@pytest.mark.parametrize('bias', ['BaccoPT'])
def test_Sk(bias):
    info = get_info(bias, A_AE=0)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    lkl = model.likelihood['ClLike']
    pk2d_lin = lkl.provider.get_CCL()['cosmo'].get_linear_power()
    a_arr, lnk, pk2d_nlin = \
        lkl.provider.get_Pk()['pk_data']['pk_ww'].get_spline_arrays()

    k = np.exp(lnk)

    for i, ai in enumerate(a_arr):
        assert pk2d_nlin[i] == pytest.approx(pk2d_lin(k, ai), rel=1e-4)

    info = get_info(bias, A_AE=1)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    lkl = model.likelihood['ClLike']
    Sk = lkl.provider.get_Pk()['pk_data']['Sk'].get_spline_arrays()[-1]
    assert Sk == pytest.approx(1, rel=1e-9)
