import pyccl as ccl
import numpy as np
import sacc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# pip install git+https://bitbucket.org/rangulo/baccoemu.git@master
import baccoemu

nonlinear_emu_path =  None
nonlinear_emu_details = None
mpk = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
                                    nonlinear_emu_details=nonlinear_emu_details) # this generates many warnings from tensorflow, it's normal

# Scale factor
a = np.linspace(mpk.emulator['baryon']['bounds'][-1][0], 1, 100)

# Hubble
h = 0.67

# Array of k's as in bacco.py to avoid interpolation issues that prevent us
# from getting chi2 < 1e-3
log10k_sh_sh_min = np.log10(0.0001)
log10k_sh_sh_max = np.log10(50)
nk_sh_sh_per_decade=20
nk_sh_sh_total = int((log10k_sh_sh_max - log10k_sh_sh_min) * nk_sh_sh_per_decade)
k_emu = ks_sh_sh = np.logspace(log10k_sh_sh_min, log10k_sh_sh_max, nk_sh_sh_total)
k_sh_sh_for_bacco = ks_sh_sh/h

mask_ks_sh_sh_for_bacco = np.squeeze(np.where(k_sh_sh_for_bacco <= mpk.emulator['baryon']['k'].max()))

k_emu = k_emu[mask_ks_sh_sh_for_bacco]
k_sh_sh_for_bacco = k_sh_sh_for_bacco[mask_ks_sh_sh_for_bacco]

cosmopars = {'omega_cold' : 0.31, # this is cold = cdm + baryons
             'omega_baryon' : 0.05,
             'A_s' : 2.1265e-9,
             'ns' : 0.96,
             'hubble' : h,
             'neutrino_mass' : 0.15,
             'w0' : -1,
             'wa' : 0 }

bcmpars = {'M_c' : 14,
           'eta': -0.3,
           'beta' : -0.22,
           'M1_z0_cen' : 10.5,
           'theta_out' : 0.25,
           'theta_inn' : -0.86,
           'M_inn' : 13.4 }

combined_pars = {}
for key in cosmopars.keys():
    combined_pars[key] = np.full((len(a)), cosmopars[key])

for key in bcmpars.keys():
    combined_pars[key] = np.full((len(a)), bcmpars[key])
combined_pars['expfactor'] = a

pklin_emu = mpk.get_linear_pk(cold=False, k=k_sh_sh_for_bacco,
                              **combined_pars)[1] / h**3
pk_emu = mpk.get_nonlinear_pk(baryonic_boost=True, cold=False,
                              k=k_sh_sh_for_bacco, **combined_pars)[1] / h**3

# get_sigma8 always returns sigma8 at z=0
print('sigma8 =', mpk.get_sigma8(**combined_pars, cold=False)[0])

pk_linear = {'a': a, 'k': k_emu, 'delta_matter:delta_matter': pklin_emu}
pk_nonlinear = {'a': a, 'k': k_emu, 'delta_matter:delta_matter': pk_emu}

# np.savez_compressed('pk_nonlinear_baryons.npz', **pk_nonlinear)

cosmo = ccl.CosmologyCalculator(Omega_c=cosmopars['omega_cold'] - cosmopars['omega_baryon'],
                                Omega_b=cosmopars['omega_baryon'],
                                h=cosmopars['hubble'],
                                n_s=cosmopars['ns'],
                                A_s=cosmopars['A_s'],
                                m_nu = cosmopars['neutrino_mass'],
                                w0=cosmopars['w0'],
                                wa=cosmopars['wa'],
                                pk_linear=pk_linear,
                                pk_nonlin=pk_nonlinear,
                                T_CMB=2.7255)


pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
apk2d, logkpk2d, pkpk2d = pkmm.get_spline_arrays()
# np.savez_compressed('pk_nonlinear_baryons_ccl.npz', a=apk2d, k=np.exp(logkpk2d),
#                     pk=pkpk2d)



zs = np.linspace(0., 1.5, 1024)
nz = interp1d(zs, np.exp(-0.5*((zs-0.5)/0.05)**2), bounds_error=False,
              fill_value=0)
bz = np.ones(zs.size)

# Nuisance parameters
dz_sh =  [0.2, 0.4, 0.6]
m_sh =  [0.1, 0.3, 0.5]
# Intinsic Alingments
A0 = 0.1
eta = 1
A_IA = A0 * ((1+zs)/1.62)**eta
ia_bias = (zs, A_IA)

# CCL tracers
for dzi in dz_sh:
    plt.plot(zs, nz(zs - dzi))
plt.show()

tracers = [ccl.WeakLensingTracer(cosmo, dndz=(zs, nz(zs - dzi)),
                              ia_bias=ia_bias) for dzi in dz_sh]
n_tracers = len(tracers)
nx = (n_tracers * (n_tracers + 1)) // 2
pols = ['e', 'e', 'e']
tracer_names = [f'sh{i}' for i in range(len(dz_sh))]

# Scale sampling
ls = np.unique(np.geomspace(2, 8192, 90).astype(int)).astype(float)
ls = np.concatenate((np.array([0.]), ls))

l_all = np.arange(8192)
bpw_edges = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351,
                      398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416,
                      1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444,
                      5047, 5731, 6508, 7390, 8192])

fsky = 0.4
dell = np.diff(bpw_edges)
nbpw = bpw_edges.size - 1
bpws = np.zeros([nbpw, l_all.size])
for i in range(nbpw):
    l0 = max(2, bpw_edges[i])
    lf = bpw_edges[i+1]
    bpws[i, l0:lf] = 1./dell[i]


def interp_bin(cl):
    # Interpolating in log space as in the code
    clf = interp1d(np.log(ls), cl)
    return np.dot(bpws, clf(np.log(l_all)))


def get_pairs():
    ix = 0
    for i1 in range(n_tracers):
        for i2 in range(i1, n_tracers):
            typ = f'cl_{pols[i1]}{pols[i2]}'
            yield i1, i2, ix, typ
            ix += 1


lbpw = np.dot(bpws, l_all)

sls = np.zeros([n_tracers, n_tracers, nbpw])
for i1, i2, ix, _ in get_pairs():
    cl = interp_bin(ccl.angular_cl(cosmo, tracers[i1],
                                   tracers[i2], ls))
    cl *= (1+m_sh[i1])
    cl *= (1+m_sh[i2])
    sls[i1, i2, :] = cl
    sls[i2, i1, :] = cl
nls = np.zeros([n_tracers, n_tracers, nbpw])
# 4 gal per arcmin^2
for i in range(n_tracers):
    nls[i, i] += 0.28**2/(16*(180*60/np.pi)**2)
cls = sls + nls

cov = np.zeros([nx, nbpw, nx, nbpw])
nmodes = (2*lbpw+1)*dell*fsky
for i1, i2, ix, _ in get_pairs():
    for j1, j2, jx, _ in get_pairs():
        cov[ix, :, jx, :] = np.diag((cls[i1, j1]*cls[i2, j2] +
                                     cls[i1, j2]*cls[i2, j1])/nmodes)
cov = cov.reshape([nx*nbpw, nx*nbpw])

s = sacc.Sacc()
for i in range(n_tracers):
    s.add_tracer('NZ', f'sh{i}',
                 quantity='galaxy_shear',
                 spin=2, z=zs, nz=nz(zs))
wins = sacc.BandpowerWindow(l_all, bpws.T)
for i1, i2, ix, typ in get_pairs():
    print(i1, i2, tracer_names[i1], tracer_names[i2], ix, typ)
    s.add_ell_cl(typ, tracer_names[i1], tracer_names[i2],
                 lbpw, sls[i1, i2], window=wins)
s.add_covariance(cov)

for i1, i2, ix, typ in get_pairs():
    l, cl, cov = s.get_ell_cl(typ, tracer_names[i1],
                              tracer_names[i2], return_cov=True)
    plt.figure()
    plt.errorbar(l, cl, yerr=np.sqrt(np.diag(cov)))
    plt.plot(l, nls[i1, i2], 'k--')
    plt.loglog()
    plt.title(f"{tracer_names[i1]}, {tracer_names[i2]}")
plt.show()
s.save_fits("sh_baccoemu_baryons.fits", overwrite=True)
os.system('gzip sh_baccoemu_baryons.fits')
