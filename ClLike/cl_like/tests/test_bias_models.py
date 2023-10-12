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
    if bias == 'BaccoPT':
        data += "cl_like/tests/data/linear_baccopt_5x2pt.fits.gz"
    else:
        data += "cl_like/tests/data/linear_halofit_5x2pt.fits.gz"
    # TODO: The biases are the culprit of the disagreement for baccopt
    bias_gc0_b1 = 1.2 if bias not in ['LagrangianPT', 'BaccoPT'] else 0.2
    bias_gc1_b1 = 1.4 if bias not in ['LagrangianPT', 'BaccoPT'] else 0.4
    info = {"params": {"A_sE9": 2.1265,
                       "Omega_c": 0.26,
                       "Omega_b": 0.05,
                       "h": 0.67,
                       "n_s": 0.96,
                       "m_nu": 0.15,
                       "T_CMB": 2.7255,
                       # gc0
                       "bias_gc0_b1": bias_gc0_b1,
                       "bias_gc0_b1p": 0.0,
                       "bias_gc0_b2": 0.0,
                       "bias_gc0_bs": 0.0,
                       "bias_gc0_bk2": 0.0,
                       "limber_gc0_dz": 0.1,
                       # gc1
                       "bias_gc1_b1": bias_gc1_b1,
                       "bias_gc1_b1p": 0.0,
                       "bias_gc1_b2": 0.0,
                       "bias_gc1_bs": 0.0,
                       "bias_gc1_bk2": 0.0,
                       "limber_gc1_dz": 0.15,
                       "bias_gc1_s": 2/5,
                       # sh0
                       "bias_sh0_m": 0.1,
                       "limber_sh0_dz": 0.2,
                       # sh1
                       "bias_sh1_m": 0.3,
                       "limber_sh1_dz": 0.4,
                       # sh2
                       "bias_sh2_m": 0.5,
                       "limber_sh2_dz": 0.6,
                       # IA
                       "limber_eta_IA": 1,
                       "bias_A_IA": 0.1,
                       # Derived
                       "sigma8": None},
            "theory": {"ccl": {"external": cll.CCL,
                               "transfer_function": "boltzmann_camb",
                               "matter_pk": "halofit",
                               "baryons_pk": "nobaryons"},
                       "limber": {"external": Limber,
                                  "nz_model": "NzShift",
                                  "input_params_prefix": "limber",
                                  "ia_model": "IADESY1"},
                       "Pk": {"external": Pk,
                             "bias_model": bias,
                              "zmax_pks": 1.5},  # For baccoemu
                       "clfinal": {"external": ClFinal,
                                   "input_params_prefix": "bias",
                                   "shape_model": "ShapeMultiplicative"}
                       },
            "likelihood": {"ClLike": {"external": cll.ClLike,
                                      "input_file": data,
                                      "bins": [
                                               {"name": "gc0"},
                                               {"name": "gc1"},
                                               {"name": "sh0"},
                                               {"name": "sh1"},
                                               {"name": "sh2"},
                                               {"name": "kp"},
                                               ],
                                      "twopoints": [{"bins": ["gc0", "gc0"]},
                                                    {"bins": ["gc1", "gc1"]},

                                                    {"bins": ["gc0", "sh0"]},
                                                    {"bins": ["gc0", "sh1"]},
                                                    {"bins": ["gc0", "sh2"]},
                                                    {"bins": ["gc1", "sh0"]},
                                                    {"bins": ["gc1", "sh1"]},
                                                    {"bins": ["gc1", "sh2"]},

                                                    {"bins": ["gc0", "kp"]},
                                                    {"bins": ["gc1", "kp"]},

                                                    {"bins": ["sh0", "sh0"]},
                                                    {"bins": ["sh0", "sh1"]},
                                                    {"bins": ["sh0", "sh2"]},
                                                    {"bins": ["sh1", "sh1"]},
                                                    {"bins": ["sh1", "sh2"]},
                                                    {"bins": ["sh2", "sh2"]},

                                                    {"bins": ["sh0", "kp"]},
                                                    {"bins": ["sh1", "kp"]},
                                                    {"bins": ["sh2", "kp"]},

                                                    {"bins": ["kp", "kp"]},
                                                    ],
                                      "defaults": {"kmax": 0.5,
                                                   "lmin": 0,
                                                   "lmax": 2000,
                                                   # Removing the large scales
                                                   # due to extrapolation
                                                   # differences in pkmd1 &
                                                   # pkd1d1
                                                   "gc1": {"lmin": 0,
                                                           "mag_bias": True}
                                                   },
                                      }
                           },
            "debug": False}

    if not A_sE9:
        # info["params"]["sigma8"] = 0.78220521  # From Bacco
        info["params"]["sigma8"] = 0.7824601264149301  # From CCL
        del info["params"]["A_sE9"]

    return info

@pytest.mark.parametrize('bias', ['Linear', 'EulerianPT', 'LagrangianPT',
                                  'BaccoPT'])
def test_dum(bias):
    info = get_info(bias)

    model = get_model(info)
    loglikes, derived = model.loglikes()

    if bias != 'BaccoPT':
        assert np.fabs(loglikes[0]) < 3E-3
    else:
        # For some reason I cannot push it lower than this.
        assert np.fabs(loglikes[0]) < 0.2


# TODO: Move this test to another file or rename this one
def test_sigma8():
    info = get_info(bias="Linear", A_sE9=False)
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 3E-3

    del info["params"]["sigma8"]
    with pytest.raises(ValueError):
        model = get_model(info)

@pytest.mark.parametrize('case', ['IAPerBin', 'IADESY1', 'IADESY1_PerSurvey'])
def test_ia_models(case):
    info = get_info(bias="Linear")
    if case in ['IAPerBin', 'IADESY1_PerSurvey']:
        # In this case, the params are already as they have to
        A = info["params"].pop("bias_A_IA")
        eta = info["params"].pop("limber_eta_IA")
        for i in range(3):
            info["params"][f"bias_sh{i}_A_IA"] = A
            info["params"][f"limber_sh{i}_eta_IA"] = eta
    info["theory"]["limber"]["ia_model"] = case
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    cond = np.fabs(loglikes[0]) < 3E-3
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
    assert np.fabs(loglikes[0]) < 3E-3

    info["theory"]["clfinal"]["shape_model"] = "ShapeNone"
    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) > 3E-3


def test_timing():
    info = get_info(bias="Linear", A_sE9=False)
    info["timing"] = True
    model = get_model(info)
    model.measure_and_set_speeds(10)
    model.dump_timing()
    time = np.sum([c.timer.get_time_avg() for c in model.components])
    if os.getenv("GITHUB_ACTIONS") == "true":
        # First time it was 1.1, next 1.7. Choosing 2s to have enough margin.
        # We might need to skip this test in GitHub it if it's not very stable
        # Multiplying by 4 since we have many more data now
        assert time < 8
    else:
        # Before restructuring, the average evaluation time was ~0.54s in my laptop
        # After the restructuration, it went to 0.56s.
        # With the new data, it takes longer because there are more data
        assert time < 2.4


def test_null_negative_eigvals_in_icov():
    info = get_info(bias="Linear", A_sE9=False)

    # The original cov does not have negative eigval. So the chi2 should be ok
    info['likelihood']['ClLike']['null_negative_cov_eigvals_in_icov'] = True
    model = get_model(info)
    loglikes, derived = model.loglikes()
    loglike0 = loglikes[0]
    assert np.fabs(loglike0) < 3E-3

    # Let's change the covariance
    s = sacc.Sacc.load_fits(info['likelihood']['ClLike']['input_file'])
    cov = s.covariance.covmat
    evals, evecs = np.linalg.eigh(cov)
    evals[np.argmin(evals)] *= -1
    cov = evecs.dot(np.diag(evals).dot(evecs.T))
    s.add_covariance(cov, overwrite=True)
    os.makedirs('dum', exist_ok=True)
    s.save_fits('dum/cls.fits')

    info['likelihood']['ClLike']['input_file'] = 'dum/cls.fits'
    model = get_model(info)
    loglikes, derived = model.loglikes()

    # The chi2 cannot be the same as before because the inv cov is (slightly)
    # different
    assert np.fabs(loglikes[0]) != pytest.approx(loglike0, rel=1e-3)


def test_get_theory_cl_sacc():
    info = get_info('Linear')

    model = get_model(info)
    loglikes, derived = model.loglikes()
    lkl = model.likelihood['ClLike']

    s = lkl.get_cl_theory_sacc()
    assert s.mean == pytest.approx(model.provider.get_cl_theory(), rel=1e-5)
    for trn, tr in s.tracers.items():
        assert lkl.tracer_qs[trn] == tr.quantity
        if tr.quantity in ['galaxy_density', 'galaxy_shear']:
            assert tr.z == pytest.approx(lkl.bin_properties[trn]['z_fid'], rel=1e-5)
            assert tr.nz == pytest.approx(lkl.bin_properties[trn]['nz_fid'], rel=1e-5)
        # Not testing the gc and sh tracers because spin is not an attribute of
        # the NzTracer
        if tr.quantity == 'cmb_lensing':
            assert tr.spin == 0


def test_get_cl_data_sacc():
    info = get_info('Linear')

    model = get_model(info)
    loglikes, derived = model.loglikes()
    lkl = model.likelihood['ClLike']

    s = lkl.get_cl_data_sacc()
    assert s.mean.size == lkl.ndata
    assert s.mean == pytest.approx(lkl.data_vec, rel=1e-5)
    assert np.all(s.covariance.covmat == lkl.cov)


def test_Omega_m():
    info = get_info('Linear')
    pars = info['params']
    Omega_nu = info['params']['m_nu']/(93.4 * info['params']['h']**2)
    pars['Omega_m'] = pars['Omega_c'] + pars['Omega_b'] + Omega_nu
    del pars['Omega_c']

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    # The loglike is slightly higher probably because of the way Omega_nu is
    # computed here
    assert np.fabs(loglikes[0]) < 5E-3

def test_neutrinos():
    info = get_info('Linear')
    pars = info['params']
    pars['m_nu'] = 0.4
    model = get_model(info)
    loglikes, derived = model.loglikes()


    Onu = pars['m_nu'] / (93.14 * pars['h']**2)
    pars['Omega_m'] = pars['Omega_c'] + pars['Omega_b'] + Onu
    del pars['Omega_c']
    model = get_model(info)
    loglikes2, derived = model.loglikes()

    assert np.fabs(loglikes[0]/loglikes2[0] -1) < 1E-4

def test_S8():
    info = get_info('Linear', False)
    sigma8 = info['params']['sigma8']
    Omega_nu = info['params']['m_nu']/(93.4 * info['params']['h']**2)
    Omega_m = info['params']['Omega_c'] + info['params']['Omega_b'] + Omega_nu
    del info['params']['sigma8']
    info['params']['S8'] = sigma8 * np.sqrt(Omega_m/0.3)

    model = get_model(info)
    loglikes, derived = model.loglikes()
    print(loglikes)
    assert np.fabs(loglikes[0]) < 3E-3


def test_sigma8_to_As():
    info = get_info('Linear', False)
    info['theory']['ccl']['sigma8_to_As'] = True
    info['params']['A_s'] = None
    cosmopars = {
       "Omega_c": 0.26,
       "Omega_b": 0.05,
       "h": 0.67,
       "n_s": 0.96,
       "m_nu": 0.15,
       "sigma8": 0.7824601264149301,
    }

    model = get_model(info)

    As = model.theory['ccl']._get_As_from_sigma8(cosmopars)
    assert np.abs(As*1E9 / 2.1265 -1) < 1e-3

    loglikes, derived = model.loglikes()
    assert np.abs(As / derived[0] - 1) < 1e-9

    print(loglikes)
    # The O(1e-3) difference between As makes chi2~0.37
    assert np.fabs(loglikes[0]) < 0.4


def test_camb_hmcode_dum():
    info = get_info('Linear', False)
    info['theory']['ccl']['sigma8_to_As'] = True

    ccl_arguments = {'extra_parameters': {"camb": {"halofit_version":
                                                   "mead2020_feedback",
                                                   "HMCode_logT_AGN": 7.8}}}
    info['theory']['ccl']['ccl_arguments'] = ccl_arguments

    model = get_model(info)
    loglikes, derived = model.loglikes()


    info['params']['HMCode_logT_AGN'] = 7.8
    ccl_arguments = {'extra_parameters': {"camb": {"halofit_version":
                                                   "mead2020_feedback"}}}
    info['theory']['ccl']['ccl_arguments'] = ccl_arguments

    model = get_model(info)
    loglikes2, derived = model.loglikes()


    assert np.abs(loglikes[0] / loglikes2[0] - 1) < 1e-5
