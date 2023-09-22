import pyccl as ccl
import numpy as np
import sacc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

cosmo = ccl.Cosmology(Omega_c=0.26,
                      Omega_b=0.05,
                      h=0.67,
                      n_s=0.96,
                      A_s=2.1265e-9,
                      m_nu = 0.15,
                      baryons_power_spectrum='bcm',
                      bcm_log10Mc=14,
                      bcm_etab=0.6,
                      bcm_ks=0.6,
                      matter_power_spectrum='halofit')


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
s.save_fits("sh_bcm_baryons.fits", overwrite=True)
os.system('gzip sh_bcm_baryons.fits')
