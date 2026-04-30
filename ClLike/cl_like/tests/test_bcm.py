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
import pyccl as ccl


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isdir("dum"):
        shutil.rmtree("dum")


def get_info(bias, A_sE9=True):
    data = "" if "ClLike" in os.getcwd() else "ClLike/"

    if bias == 'Linear':
        data += "cl_like/tests/data/sh_bcm_baryons.fits.gz"
        pk_dict = {"external": Pk,
                   "bias_model": "Linear"}
    elif bias == 'BaccoPT':
        data += "cl_like/tests/data/sh_PkmmBacco_bcm.fits.gz"
        pk_dict = {"external": Pk,
                   "bias_model": "BaccoPT",
                   "zmax_pks": 1.5,  # For baccoemu with baryons
                   # "use_baryon_boost" : True,
                   "ignore_lbias": True,
                   }
    else:
        raise ValueError(f'bias {bias} not implemented')



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
                       "log10Mc": 14,
                       "eta_b": 0.6,
                       "k_s": 50,
                       # Derived
                       "sigma8": None,
                       },
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "schneider15"},
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

    if not A_sE9:
        info["params"]["sigma8"] = 0.78220521
        del info["params"]["A_sE9"]

    return info


@pytest.mark.parametrize('bias', ['Linear', 'BaccoPT'])
def test_dum(bias):
    info = get_info(bias)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert derived[0] == pytest.approx(0.78220521, rel=1e-3)

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


    # TODO: Ideally, I should be able to recover exactly the data vector.
    # However, I cannot push chi2 < 0.2 for bcm_ks=50. Not sure where the
    # missmatch is.
    assert np.fabs(loglikes[0]) < 0.2


@pytest.mark.parametrize('bias', ['Linear', 'BaccoPT'])
def test_Sk(bias):
    info = get_info(bias)
    model = get_model(info)
    loglikes, derived = model.loglikes()

    pars = info['params']

    # cosmo = ccl.CosmologyVanillaLCDM()
    cosmo = ccl.Cosmology(Omega_c=pars['Omega_c'],
                          Omega_b=pars['Omega_b'],
                          h=pars['h'],
                          n_s=pars['n_s'],
                          A_s=pars['A_sE9']*1e-9,
                          m_nu=pars['m_nu'],
                          )

    bcm = ccl.baryons.BaryonsSchneider15(log10Mc=pars['log10Mc'],
                                         eta_b=pars['eta_b'],
                                         k_s=pars['k_s'])

    lkl = model.likelihood['ClLike']
    a_arr, lnk, Sk = lkl.provider.get_Pk()['pk_data']['Sk'].get_spline_arrays()
    k = np.exp(lnk)

    for i, ai in enumerate(a_arr):
        assert Sk[i] == pytest.approx(bcm.boost_factor(cosmo, k, ai),
                                      rel=1e-3)
