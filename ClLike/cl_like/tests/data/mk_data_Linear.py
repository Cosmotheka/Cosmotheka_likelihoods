import pyccl as ccl
import numpy as np
import sacc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                      n_s=0.96, A_s=2.23E-9, T_CMB=2.7255)

zs = np.linspace(0., 1., 1024)
nz = np.exp(-0.5*((zs-0.5)/0.05)**2)
bz = np.ones(zs.size)

t_gc = ccl.NumberCountsTracer(cosmo, False, dndz=(zs, nz), bias=(zs, bz))
t_sh = ccl.WeakLensingTracer(cosmo, dndz=(zs, nz))
t_kp = ccl.CMBLensingTracer(cosmo, z_source=1100.)
n_tracers = 3
nx = (n_tracers * (n_tracers + 1)) // 2
tracers = [t_gc, t_kp, t_sh]
pols = ['0', '0', 'e']
tracer_names = ['gc1', 'kp', 'sh1']

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
    clf = interp1d(ls, cl)
    return np.dot(bpws, clf(l_all))


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
    sls[i1, i2, :] = cl
nls = np.zeros([n_tracers, n_tracers, nbpw])
# 4 gal per arcmin^2
nls[0, 0, :] = 1./(4*(180*60/np.pi)**2)
nls[1, 1, :] = 4E-8
nls[2, 2, :] = 0.28**2/(4*(180*60/np.pi)**2)
cls = sls + nls

cov = np.zeros([nx, nbpw, nx, nbpw])
nmodes = (2*lbpw+1)*dell*fsky
for i1, i2, ix, _ in get_pairs():
    for j1, j2, jx, _ in get_pairs():
        cov[ix, :, jx, :] = np.diag((cls[i1, j1]*cls[i2, j2] +
                                     cls[i1, j2]*cls[i2, j1])/nmodes)
cov = cov.reshape([nx*nbpw, nx*nbpw])

s = sacc.Sacc()
s.add_tracer('NZ', 'gc1',
             quantity='galaxy_density',
             spin=0, z=zs, nz=nz)
s.add_tracer('NZ', 'sh1',
             quantity='galaxy_shear',
             spin=2, z=zs, nz=nz)
s.add_tracer('Map', 'kp',
             quantity='cmb_convergence',
             spin=0, ell=l_all,
             beam=np.ones(l_all.shape))
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
plt.show()
s.save_fits("gc_kp_sh_linear.fits", overwrite=True)
os.system('gzip gc_kp_sh_linear.fits')
