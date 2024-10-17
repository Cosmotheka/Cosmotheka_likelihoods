import cl_like as cll
from cl_like.limber import Limber
from cl_like.power_spectrum import Pk
from cl_like.cl_final import ClFinal
from cl_like.bacco import BaccoCalculator
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
    # min k 3D power spectra
    l10k_min_pks: float = -4.0
    # max k 3D power spectra
    l10k_max_pks: float = 2.0
    # #k for 3D power spectra
    nk_per_dex_pks: int = 25

    nk_pks = int((l10k_max_pks - l10k_min_pks) * nk_per_dex_pks)
    ptc = BaccoCalculator(use_baryon_boost=True,
                          ignore_lbias=True,
                          allow_bcm_emu_extrapolation_for_shear=True,
                          allow_halofit_extrapolation_for_shear=True)
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
                             "bias_model": "BaccoPT",
                             "zmax_pks": 1.5,  # For baccoemu with baryons
                             "use_baryon_boost" : True,
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
            "debug": False}

    if not A_sE9:
        info["params"]["sigma8"] = 0.78220521
        del info["params"]["A_sE9"]

    return info


def test_dum():
    info = get_info()
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
    # However, I cannot push chi2 < 0.19. Not sure where the missmatch is.
    assert np.fabs(loglikes[0]) < 0.2

    info['params']['M_c'] = 10
    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.fabs(loglikes[0]) > 0.2


def test_A_s_sigma8():
    info = get_info(A_sE9=False)
    model = get_model(info)
    loglikes_sigma8, derived = model.loglikes()
    assert np.fabs(loglikes_sigma8[0]) < 0.2

    info = get_info(A_sE9=True)
    model = get_model(info)
    loglikes_A_s, derived = model.loglikes()
    assert np.fabs(loglikes_A_s[0]) < 0.2

    assert np.fabs(loglikes_sigma8[0] / loglikes_A_s[0] - 1) < 2e-3


def test_baryons_Sk(ptc):
    info = get_info()
    pars = info['params']

    cosmo = ccl.Cosmology(Omega_c=pars['Omega_c'],
                          Omega_b=pars['Omega_b'],
                          h=pars['h'],
                          n_s=pars['n_s'],
                          A_s=pars['A_sE9']*1e-9,
                          m_nu=pars['m_nu'])

    bcmpar = {
               "M_c" :  14,
               "eta" : -0.3,
               "beta" : -0.22,
               "M1_z0_cen" : 10.5,
               "theta_out" : 0.25,
               "theta_inn" : -0.86,
               "M_inn" : 13.4,
    }

    ptc.update_pk(cosmo, bcmpar=None)
    pk = ptc.get_pk('mm_sh_sh').get_spline_arrays()[-1]
    ptc.update_pk(cosmo, bcmpar=bcmpar)
    pkb = ptc.get_pk('mm_sh_sh').get_spline_arrays()[-1]
    Sk = ptc.get_pk('Sk').get_spline_arrays()[-1]
    assert Sk == pytest.approx(pkb/pk, rel=1e-5)

    model = get_model(info)
    loglikes, derived = model.loglikes()
    lkl = model.likelihood['ClLike']
    a_arr, lnk, Sk2 = lkl.provider.get_Pk()['pk_data']['Sk'].get_spline_arrays()
    k = np.exp(lnk)
    for i, ai in enumerate(a_arr):
        assert ptc.get_pk('Sk').eval(k, ai) == pytest.approx(Sk2[i], rel=1e-3)


def test_get_pars_and_a_for_bacco(ptc):
    bcmpar = {
               "M_c" :  14,
               "eta" : -0.3,
               "beta" : -0.22,
               "M1_z0_cen" : 10.5,
               "theta_out" : 0.25,
               "theta_inn" : -0.86,
               "M_inn" : 13.4,
    }

    a_arr = [0.8, 0.9, 1]

    combined_pars = ptc._get_pars_and_a_for_bacco(bcmpar, a_arr)

    assert combined_pars['expfactor'] == a_arr

    for pn, pv in bcmpar.items():
        assert np.all(combined_pars[pn] == np.array([pv] * len(a_arr)))

def test_hfit_extrapolation(ptc):
    info = get_info()
    info['theory']['Pk']['allow_halofit_extrapolation_for_shear_on_k'] = True
    pars = info['params']

    cosmo = ccl.Cosmology(Omega_c=pars['Omega_c'],
                          Omega_b=pars['Omega_b'],
                          h=pars['h'],
                          n_s=pars['n_s'],
                          A_s=pars['A_sE9']*1e-9,
                          m_nu=pars['m_nu'])

    bcmpar = {
               "M_c" :  14,
               "eta" : -0.3,
               "beta" : -0.22,
               "M1_z0_cen" : 10.5,
               "theta_out" : 0.25,
               "theta_inn" : -0.86,
               "M_inn" : 13.4,
    }

    ptc.update_pk(cosmo, bcmpar=bcmpar)
    pkb = ptc.get_pk('mm_sh_sh')
    lnk = pkb.get_spline_arrays()[1]
    Sk = ptc.get_pk('Sk')

    cosmo.compute_nonlin_power()
    pkh = cosmo.get_nonlin_power()

    model = get_model(info)
    loglikes, derived = model.loglikes()
    lkl = model.likelihood['ClLike']
    a_arr, lnk2, pkb2 = lkl.provider.get_Pk()['pk_data']['pk_ww'].get_spline_arrays()
    a_arr, lnk2, Sk2 = lkl.provider.get_Pk()['pk_data']['Sk'].get_spline_arrays()

    k = np.exp(lnk)
    k2 = np.exp(lnk2)
    for i, ai in enumerate(a_arr):
        sel = lnk2 > lnk[-1]
        # k < baccoemu kmax
        assert Sk(k, ai) == pytest.approx(Sk2[i, ~sel], rel=5e-3)
        assert pkb(k, ai) == pytest.approx(pkb2[i, ~sel], rel=5e-3)
        # k > baccoemu kmax
        assert pkh(k2[sel], ai) * Sk(k2[sel], ai)  == pytest.approx(pkb2[i, sel], rel=1e-2)
