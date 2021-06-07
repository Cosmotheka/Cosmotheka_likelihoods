#!/usr/bin/python3
import pyccl as ccl
import numpy as np
from scipy.interpolate import interp1d

def get_interpolated_cl(cosmo, ccl_tracer1, ccl_tracer2, ls):
    ls_nodes = np.unique(np.geomspace(2, ls[-1], 30).astype(int)).astype(float)
    cls_nodes = ccl.angular_cl(cosmo,
                               ccl_tracer1,
                               ccl_tracer2,
                               ls_nodes)
    cli = interp1d(np.log(ls_nodes), cls_nodes,
                   fill_value=0, bounds_error=False)
    msk = ls >= 2
    cls = np.zeros(len(ls))
    cls[msk] = cli(np.log(ls[msk]))
    return cls

def get_binned_cl(cosmo, ccl_tracer1, ccl_tracer2, ell_bpw, w_bpw, interp=True):
    # Get the bandpower window function.
    # Unbinned power spectrum.
    if interp:
        cl_unbinned = get_interpolated_cl(cosmo, ccl_tracer1, ccl_tracer2,
                                          ell_bpw)
    else:
        cl_unbinned = ccl.angular_cl(cosmo, ccl_tracer1, ccl_tracer2, ell_bpw)
    # Convolved with window functions.
    cl_binned = np.dot(w_bpw, cl_unbinned)

    return cl_binned
