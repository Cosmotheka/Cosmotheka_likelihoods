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
    data = "" if "ClLike" in os.getcwd() else "ClLike/"
    data += "cl_like/tests/data/gc_kp_sh_linear_nuisances.fits.gz"
    info = {"params": {"A_sE9": 2.23,
                       "Omega_c": 0.25,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.0,
                       "bias_sh1_m": 0.3,
                       "limber_sh1_dz": -0.2,
                       "limber_sh1_eta_IA": 1,
                       "bias_sh1_A_IA": 0.1,
                       "sigma8": None,
                       "M_c" :  14,
                       "eta" : -0.3,
                       "beta" : -0.22,
                       "M1_z0_cen" : 10.5,
                       "theta_out" : 0.25,
                       "theta_inn" : -0.86,
                       "M_inn" : 13.4},
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
                             "zmax_pks": 1.5,  # For baccoemu with baryons
                             "use_baryon_boost" : True,
                             },
                       "clfinal": {"external": ClFinal,
                                   "input_params_prefix": "bias",
                                   "shape_model": "ShapeMultiplicative"}
                       },
            "likelihood": {"ClLike": {"external": cll.ClLike,
                                      "input_file": data,
                                      "bins": [{"name": "sh1"}],
                                      "twopoints": [{"bins": ["sh1", "sh1"]}],
                                      "defaults": {"kmax": 50,
                                                   "lmin": 0,
                                                   "lmax": 2000},
                                      }
                           },
            "output": "dum",
            "debug": False}

    if not A_sE9:
        info["params"]["sigma8"] = 0.8098
        del info["params"]["A_sE9"]

    return info


@pytest.mark.parametrize('bias', ['BaccoPT'])
def test_dum(bias):
    fiducial_cls = np.array([1.66251413e-08, 9.31467887e-09, 5.57499362e-09, 3.78614838e-09,
                            2.75582068e-09, 2.11894024e-09, 1.70646324e-09, 1.41966536e-09,
                            1.20893262e-09, 1.04966128e-09, 9.25580250e-10, 8.26779309e-10,
                            7.46644021e-10, 6.80093772e-10, 6.24297208e-10, 5.76490038e-10,
                            5.35381203e-10, 4.99664009e-10, 4.68296780e-10, 4.40227361e-10,
                            4.15381851e-10, 3.93140905e-10, 3.72803472e-10, 3.54650242e-10,
                            3.37724093e-10, 3.22544230e-10, 3.08260792e-10, 2.95364949e-10,
                            2.83136891e-10, 2.72040107e-10, 2.61460408e-10, 2.51725588e-10,
                            2.42579380e-10, 2.33851084e-10, 2.25867064e-10, 2.18150036e-10,
                            2.10904923e-10, 2.04167903e-10, 1.97607517e-10, 1.91431363e-10,
                            1.85671667e-10, 1.80051968e-10, 1.74713642e-10, 1.69756121e-10,
                            1.64914957e-10, 1.60202458e-10, 1.55850873e-10, 1.51665521e-10,
                            1.47567105e-10, 1.43615178e-10, 1.39959943e-10, 1.36393970e-10,
                            1.32896385e-10, 1.29538368e-10, 1.26423017e-10, 1.23371969e-10,
                            1.20375263e-10, 1.17444826e-10, 1.14732058e-10, 1.12113008e-10,
                            1.09537504e-10, 1.07004122e-10, 1.04567502e-10, 1.02306739e-10,
                            1.00087158e-10, 9.79016495e-11, 9.57491816e-11, 9.36655568e-11,
                            9.17397630e-11, 8.98522302e-11, 8.79916062e-11, 8.61571347e-11,
                            8.43526808e-11, 8.26779945e-11, 8.10683536e-11, 7.94801329e-11,
                            7.79127699e-11, 7.63657237e-11, 7.48465585e-11, 7.34445309e-11,
                            7.20880562e-11, 7.07482980e-11, 6.94248493e-11, 6.81173177e-11,
                            6.68253249e-11, 6.56013348e-11, 6.44558735e-11, 6.33235564e-11,
                            6.22040851e-11, 6.10971716e-11, 6.00025371e-11, 5.89199122e-11,
                            5.79050022e-11, 5.69466157e-11, 5.59984090e-11, 5.50601680e-11,
                            5.41316855e-11, 5.32127604e-11, 5.23031982e-11, 5.14213388e-11,
                            5.06124821e-11, 4.98168814e-11, 4.90290696e-11, 4.82488955e-11,
                            4.74762125e-11, 4.67108781e-11, 4.59527538e-11, 4.52177944e-11,
                            4.45453886e-11, 4.38837123e-11, 4.32280432e-11, 4.25782732e-11,
                            4.19342971e-11, 4.12960126e-11, 4.06633198e-11, 4.00361216e-11,
                            3.94440606e-11, 3.88932960e-11, 3.83471931e-11, 3.78056737e-11,
                            3.72686614e-11, 3.67360818e-11, 3.62078624e-11, 3.56839322e-11,
                            3.51642223e-11, 3.46571733e-11, 3.41965724e-11, 3.37452364e-11,
                            3.32974225e-11, 3.28530760e-11, 3.24121437e-11, 3.19745735e-11,
                            3.15403145e-11, 3.11093170e-11, 3.06815324e-11, 3.02597977e-11,
                            2.98761721e-11, 2.95053911e-11, 2.91372941e-11, 2.87718424e-11,
                            2.84089983e-11, 2.80487249e-11, 2.76909860e-11, 2.73357462e-11,
                            2.69829708e-11, 2.66326258e-11, 2.62887499e-11, 2.59776802e-11,
                            2.56747905e-11, 2.53739317e-11, 2.50750765e-11, 2.47781986e-11,
                            2.44832720e-11, 2.41902711e-11, 2.38991710e-11, 2.36099472e-11,
                            2.33225756e-11, 2.30370327e-11, 2.27584322e-11, 2.25072133e-11,
                            2.22609735e-11, 2.20162617e-11, 2.17730591e-11, 2.15313472e-11,
                            2.12911079e-11, 2.10523232e-11, 2.08149756e-11, 2.05790481e-11,
                            2.03445237e-11, 2.01113857e-11, 1.98796180e-11, 1.96571902e-11,
                            1.94556956e-11, 1.92562170e-11, 1.90578839e-11, 1.88606832e-11,
                            1.86646019e-11, 1.84696276e-11, 1.82757477e-11, 1.80829500e-11,
                            1.78912225e-11, 1.77005535e-11, 1.75109312e-11, 1.73223443e-11,
                            1.71347815e-11, 1.69589046e-11, 1.67969938e-11, 1.66359482e-11,
                            1.64757585e-11, 1.63164157e-11, 1.61579110e-11, 1.60002354e-11,
                            1.58433804e-11, 1.56873376e-11, 1.55320984e-11, 1.53776548e-11,
                            1.52239985e-11, 1.50711217e-11, 1.49190163e-11, 1.47676748e-11])

    info = get_info(bias)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    ll = model.likelihood['ClLike']
    cls_th = ll.provider.get_cl_theory()
    cls_bin = []
    for i, bins in enumerate(ll.twopoints):
        t1, t2 = bins['bins']
        inds = ll.cl_meta[i]['inds']
        cls_bin.append(cls_th[inds])
    assert cls_bin[0] == pytest.approx(fiducial_cls, rel=2E-3)


def test_A_s_sigma8():
    info = get_info('BaccoPT', A_sE9=False)
    model = get_model(info)
    loglikes_sigma8, derived = model.loglikes()

    info = get_info('BaccoPT', A_sE9=True)
    model = get_model(info)
    loglikes_A_s, derived = model.loglikes()

    assert np.fabs(loglikes_sigma8[0] / loglikes_A_s[0] - 1) < 2e-2