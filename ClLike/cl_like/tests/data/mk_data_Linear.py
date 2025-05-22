import pyccl as ccl
import numpy as np
import sacc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("non_linear", type=str, help='halofit, baccopkmm or baccopt')
args = parser.parse_args()
args.non_linear

cosmopars = {'omega_cold' : 0.31, # this is cold = cdm + baryons
             'omega_baryon' : 0.05,
             'A_s' : 2.1265e-9,
             'ns' : 0.96,
             'hubble' : 0.67,
             'neutrino_mass' : 0.15,
             'w0' : -1,
             'wa' : 0 }


if args.non_linear == 'baccopt':
    # pip install git+https://bitbucket.org/rangulo/baccoemu.git@master
    import baccoemu

    nonlinear_emu_path =  None
    nonlinear_emu_details = None
    mpk = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
                                        nonlinear_emu_details=nonlinear_emu_details) # this generates many warnings from tensorflow, it's normal
    lbias = baccoemu.Lbias_expansion()

    # Scale factor
    a = np.linspace(lbias.emulator['nonlinear']['bounds'][-1][0], 1, 100)

    # Hubble
    h = cosmopars['hubble']

    # Array of k's as in bacco.py to avoid interpolation issues that prevent us
    # from getting chi2 < 1e-3
    log10k_min = np.log10(0.008)
    log10k_max = np.log10(0.5)
    nk_per_decade=20
    nk_total = int((log10k_max - log10k_min) * nk_per_decade)
    k_emu = ks = np.logspace(log10k_min, log10k_max, nk_total)
    k_for_bacco = ks/h

    mask_ks_for_bacco = np.squeeze(np.where(k_for_bacco <= lbias.emulator['baryon']['k'].max()))

    k_emu = k_emu[mask_ks_for_bacco]
    k_for_bacco = k_for_bacco[mask_ks_for_bacco]

    combined_pars = {}
    for key in cosmopars.keys():
        combined_pars[key] = np.full((len(a)), cosmopars[key])

    combined_pars['expfactor'] = a

    pklin_emu = None
    pk_emu = np.array([lbias.get_nonlinear_pnn(k=k_for_bacco, expfactor=ai,
                                               **cosmopars)[1]/h**3 for ai in
                       a])

    # get_sigma8 always returns sigma8 at z=0
    print('sigma8 =', mpk.get_sigma8(**combined_pars, cold=False)[0])

    pk_linear = None
    pk_nonlinear = {'a': a, 'k': k_emu, 'pk_mm': pk_emu[:, 0, :],
                    'pk_md1': pk_emu[:, 1, :], 'pk_d1d1': pk_emu[:, 5, :]}


    # np.savez_compressed('pk_nonlinear_baccopt.npz', **pk_nonlinear)

    cosmo = ccl.CosmologyCalculator(Omega_c=cosmopars['omega_cold'] - cosmopars['omega_baryon'],
                                    Omega_b=cosmopars['omega_baryon'],
                                    h=cosmopars['hubble'],
                                    n_s=cosmopars['ns'],
                                    A_s=cosmopars['A_s'],
                                    m_nu = cosmopars['neutrino_mass'],
                                    w0=cosmopars['w0'],
                                    wa=cosmopars['wa'],
                                    pk_linear=None, # pk_linear,
                                    pk_nonlin=None, # pk_nonlinear,
                                    T_CMB=2.7255)
    fname = "linear_baccopt_5x2pt.fits"
elif args.non_linear == 'baccopkmm':
    import baccoemu

    nonlinear_emu_path =  None
    nonlinear_emu_details = None
    mpk = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
                                        nonlinear_emu_details=nonlinear_emu_details) # this generates many warnings from tensorflow, it's normal

    # Scale factor
    a = np.linspace(mpk.emulator['nonlinear']['bounds'][-1][0], 1, 100)

    # Hubble
    h = cosmopars['hubble']

    # Array of k's as in bacco.py to avoid interpolation issues that prevent us
    # from getting chi2 < 1e-3
    log10k_min=np.log10(0.0001)
    log10k_max=np.log10(50)
    nk_per_decade=20
    nk_total = int((log10k_max - log10k_min) * nk_per_decade)
    k_emu = np.logspace(log10k_min, log10k_max, nk_total)
    k_for_bacco = k_emu/h

    mask_ks_for_bacco = np.squeeze(np.where(k_for_bacco <= mpk.emulator['nonlinear']['k'].max()))

    k_emu = k_emu[mask_ks_for_bacco]
    k_for_bacco = k_for_bacco[mask_ks_for_bacco]

    combined_pars = {}
    for key in cosmopars.keys():
        combined_pars[key] = np.full((len(a)), cosmopars[key])

    combined_pars['expfactor'] = a

    pklin_emu = None
    # NOTE: get_nonlinear_pk is used only for baryons & shear. I get about 1-2%
    # difference with the
    pk_emu = mpk.get_nonlinear_pk(baryonic_boost=False, cold=False,
                                  k=k_for_bacco, **combined_pars)[1] / h**3

    # get_sigma8 always returns sigma8 at z=0
    print('sigma8 =', mpk.get_sigma8(**combined_pars, cold=False)[0])

    pk_linear = None
    pk_nonlinear = {'a': a, 'k': k_emu,
                    'delta_matter:delta_matter': pk_emu}

    # np.savez_compressed('pk_nonlinear_baccopt.npz', **pk_nonlinear)

    cosmo = ccl.CosmologyCalculator(Omega_c=cosmopars['omega_cold'] - cosmopars['omega_baryon'],
                                    Omega_b=cosmopars['omega_baryon'],
                                    h=cosmopars['hubble'],
                                    n_s=cosmopars['ns'],
                                    A_s=cosmopars['A_s'],
                                    m_nu = cosmopars['neutrino_mass'],
                                    w0=cosmopars['w0'],
                                    wa=cosmopars['wa'],
                                    pk_linear=None, # pk_linear,
                                    pk_nonlin=pk_nonlinear,
                                    T_CMB=2.7255)
    fname = "linear_baccopkmm_5x2pt.fits"

elif args.non_linear == 'halofit':
    cosmo = ccl.Cosmology(Omega_c=cosmopars['omega_cold'] - cosmopars['omega_baryon'],
                          Omega_b=cosmopars['omega_baryon'],
                          h=cosmopars['hubble'],
                          n_s=cosmopars['ns'],
                          A_s=cosmopars['A_s'],
                          m_nu = cosmopars['neutrino_mass'],
                          w0=cosmopars['w0'],
                          wa=cosmopars['wa'],
                          T_CMB=2.7255,
                          transfer_function='boltzmann_camb',
                          matter_power_spectrum='halofit')
    print(cosmo.sigma8())

    # cosmo.compute_nonlin_power()
    # pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
    # a_s, k, pkmm = pkmm.get_spline_arrays()
    # np.savez_compressed('pk_nonlinear_halofit.npz', a=a_s, k=k, pk=pkmm)

    fname = "linear_halofit_5x2pt.fits"
else:
    raise ValueError('non_linear must be one of halofit or baccopt')


zs = np.linspace(0., 1.5, 1024)
nz = interp1d(zs, np.exp(-0.5*((zs-0.5)/0.05)**2), bounds_error=False,
              fill_value=0)
bz = np.ones(zs.size)

# biases
biases = [1.2, 1.4]

# Nuisance parameters
dz_gc = [0.1, 0.15]
dz_sh =  [0.2, 0.4, 0.6]
m_sh =  [0.1, 0.3, 0.5]
# Intinsic Alingments
A0 = 0.1
eta = 1
A_IA = A0 * ((1+zs)/1.62)**eta
ia_bias = (zs, A_IA)

# CCL tracers
for dzi in dz_gc + dz_sh:
    plt.plot(zs, nz(zs - dzi))
plt.show()



tracers = [ccl.NumberCountsTracer(cosmo, dndz=(zs, nz(zs - dzi)), has_rsd=False,
                                  bias=(zs, bz)) for dzi in dz_gc] + \
          [ccl.WeakLensingTracer(cosmo, dndz=(zs, nz(zs - dzi)),
                                 ia_bias=ia_bias) for dzi in dz_sh] + \
          [ccl.CMBLensingTracer(cosmo, z_source=1100.)]

n_tracers = len(tracers)
# nx = (n_tracers * (n_tracers + 1)) // 2
pols = ['0', '0', 'e', 'e', 'e', '0']
tracer_names = ['gc0', 'gc1', 'sh0', 'sh1', 'sh2', 'kp']

# # Scale sampling
# # nl = 30 and 2048+1 as in limber.py
# ls = np.unique(np.geomspace(2, 2048+1, 30).astype(int)).astype(float)
# ls = np.concatenate((np.array([0.]), ls))
#
# l_all = np.arange(ls[-1])
# bpw_edges = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351,
#                       398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416,
#                       1608, 1826, 2048])
#
# fsky = 0.4
# dell = np.diff(bpw_edges)
# nbpw = bpw_edges.size - 1
# bpws = np.zeros([nbpw, l_all.size])
# for i in range(nbpw):
#     l0 = max(2, bpw_edges[i])
#     lf = bpw_edges[i+1]
#     bpws[i, l0:lf] = 1./dell[i]

# Scale sampling
ls = np.unique(np.geomspace(2, 2002, 90).astype(int)).astype(float)
ls = np.concatenate((np.array([0.]), ls))

l_all = np.arange(2002)
fsky = 0.4
dell = 10
nbpw = (l_all.size-2) // dell
bpws = np.zeros([nbpw, 2002])
for i in range(nbpw):
    l0 = 2 + i*dell
    lf = l0 + dell
    bpws[i, l0:lf] = 1./dell


def interp_bin(cl):
    # Interpolating in log space as in the code
    clf = interp1d(np.log(ls), cl)
    return np.dot(bpws, clf(np.log(l_all)))


def get_pairs():
    ix = 0
    for i1 in range(n_tracers):
        for i2 in range(i1, n_tracers):
            if (i1 in [0, 1]) and (i2 in [0, 1]) and (i1 != i2):
                continue
            typ = f'cl_{pols[i1]}{pols[i2]}'
            yield i1, i2, ix, typ
            ix += 1


lbpw = np.dot(bpws, l_all)

sls = np.zeros([n_tracers, n_tracers, nbpw])
for i1, i2, ix, _ in get_pairs():
    if args.non_linear == 'baccopt':
        # The extrapolation of pkm1d1 & pkd1d1 is different in the code and
        # this will cause a O(1%) difference at large scales.
        pkmm = pk_nonlinear['pk_mm']
        pkmd1 = pk_nonlinear['pk_md1']
        pkd1d1 = pk_nonlinear['pk_d1d1']
        bias1 = bias2 = 0
        if i1 in [0, 1]:
            bias1 = biases[i1] - 1
        if i2 in [0, 1]:
            bias2 = biases[i2] - 1
        if (bias1 != 0) and (bias2 != 0):
            pk = pkmm + (bias1+bias2)*pkmd1 + bias1*bias2*pkd1d1
        else:
            bias1 = max(bias1, bias2)
            pk = pkmm + bias1*pkmd1
        pk = ccl.Pk2D(a_arr=pk_nonlinear['a'],
                      lk_arr=np.log(pk_nonlinear['k']),
                      pk_arr=np.log(pk), is_logp=True)
        cl = interp_bin(ccl.angular_cl(cosmo, tracers[i1], tracers[i2], ls,
                                       p_of_k_a=pk))
    else:
        cl = interp_bin(ccl.angular_cl(cosmo, tracers[i1], tracers[i2], ls))
        if i1 in [0, 1]:
            cl *= biases[i1]
        if i2 in [0, 1]:
            cl *= biases[i2]

    if i1 in [2, 3, 4]:
        cl *= (1+m_sh[i1-2])
    if i2 in [2, 3, 4]:
        cl *= (1+m_sh[i2-2])
    sls[i1, i2, :] = cl
    sls[i2, i1, :] = cl
nls = np.zeros([n_tracers, n_tracers, nbpw])

# 4 gal per arcmin^2
# gc
nls[0, 0] = nls[1, 1] = 1/(4*(180*60/np.pi)**2)
# wl
for i in range(4):
    nls[2+i, 2+i] += 0.28**2/(16*(180*60/np.pi)**2)
# cmbk
nls[-1, -1] = 4e-8
cls = sls + nls

nx = len(list(get_pairs()))
cov = np.zeros([nx, nbpw, nx, nbpw])
nmodes = (2*lbpw+1)*dell*fsky
for i1, i2, ix, _ in get_pairs():
    for j1, j2, jx, _ in get_pairs():
        cov[ix, :, jx, :] = np.diag((cls[i1, j1]*cls[i2, j2] +
                                     cls[i1, j2]*cls[i2, j1])/nmodes)
cov = cov.reshape([nx*nbpw, nx*nbpw])

s = sacc.Sacc()
for i in range(2):
    s.add_tracer('NZ', f'gc{i}',
                 quantity='galaxy_density',
                 spin=0, z=zs, nz=nz(zs))

for i in range(3):
    s.add_tracer('NZ', f'sh{i}',
                 quantity='galaxy_shear',
                 spin=2, z=zs, nz=nz(zs))
s.add_tracer('Map', f'kp',
             quantity='cmb_convergence',
             spin=0, ell=l_all, beam=np.ones_like(l_all))
wins = sacc.BandpowerWindow(l_all, bpws.T)
for i1, i2, ix, typ in get_pairs():
    print(i1, i2, tracer_names[i1], tracer_names[i2], ix, typ)
    s.add_ell_cl(typ, tracer_names[i1], tracer_names[i2],
                 lbpw, sls[i1, i2], window=wins)
s.add_covariance(cov)

# for i1, i2, ix, typ in get_pairs():
#     l, cl, cov = s.get_ell_cl(typ, tracer_names[i1],
#                               tracer_names[i2], return_cov=True)
#     plt.figure()
#     plt.errorbar(l, cl, yerr=np.sqrt(np.diag(cov)))
#     plt.plot(l, nls[i1, i2], 'k--')
#     plt.loglog()
#     plt.title(f"{tracer_names[i1]}, {tracer_names[i2]}")
# plt.show()

s.save_fits(fname, overwrite=True)
os.system(f'gzip {fname}')
