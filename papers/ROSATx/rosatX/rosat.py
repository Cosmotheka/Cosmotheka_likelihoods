import pyatomdb as padb
import numpy as np
from astropy.io import fits
import os
import pickle


class ROSATResponse(object):
    def __init__(self, fname_rsp=None, fname_rmf=None,
                 fname_arf=None, Zmetal=0.3):
        if (fname_rsp is None) and (fname_rmf is None):
            raise ValueError("Need either RSP or RMF + ARF files")
        # We prioritise RMF + ARF
        if fname_rmf is not None:
            if fname_arf is None:
                raise ValueError("Need ARF with RMF")
            self.rsp_h = fits.open(fname_rmf)[1].data
            self.chans_h = fits.open(fname_rmf)[2].data
            self.arf_h = fits.open(fname_arf)[1].data
        else:
            self.rsp_h = fits.open(fname_rsp)[1].data
            self.chans_h = fits.open(fname_rsp)[2].data
            self.arf_h = None

        self.Zmetal = Zmetal

        # Check channels are monotonously increasing
        if np.any(np.diff(self.chans_h['E_MAX']) <= 0):
            raise ValueError("Energy channels have a weird ordering")

        # Number of channels
        self.n_chan = len(self.chans_h['E_MAX'])

        # Number of energy bins
        self.n_energ = len(self.rsp_h['ENERG_HI'])

        # Energy redistribution matrix times effective area
        self.m_x_A = np.zeros([self.n_energ, self.n_chan])

        # Construct matrix
        for i in range(self.n_energ):
            ngrp = self.rsp_h['N_GRP'][i]
            if ngrp == 0:
                continue
            # First channel for this energy bin (starts from 1!!).
            i_first = self.rsp_h['F_CHAN'][i]-1
            # Number of channels
            n_ch = self.rsp_h['N_CHAN'][i]
            row = self.rsp_h['MATRIX'][i]
            self.m_x_A[i, i_first:i_first+n_ch] = row[:n_ch]

        if self.arf_h is not None:
            self.m_x_A = self.m_x_A*self.arf_h['SPECRESP'][:, None]

        self.energ_bins = np.zeros(self.n_energ+1)
        self.energ_bins[:-1] = self.rsp_h['ENERG_LO']
        self.energ_bins[-1] = self.rsp_h['ENERG_HI'][-1]

        self.energ_chans = np.zeros(self.n_chan+1)
        self.energ_chans[:-1] = self.chans_h['E_MIN']
        self.energ_chans[-1] = self.chans_h['E_MAX'][-1]

        self.sess = padb.spectrum.CIESession()
        # Set metallicity
        Zs = np.ones(31)
        Zs[3:] = self.Zmetal
        self.sess.set_abund(np.arange(31, dtype=int), Zs)

    def get_energ_in_bins(self):
        return self.energ_bins

    def get_energ_in_mean(self):
        return 0.5*(self.energ_bins[:-1]+self.energ_bins[1:])

    def get_energ_out_bins(self):
        return self.energ_chans

    def get_energ_out_mean(self):
        return 0.5*(self.energ_chans[:-1]+self.energ_chans[1:])

    def get_spectrum_in(self, kT, z,
                        dolines=True, docont=True,
                        dopseudo=True):
        """ Returns incoming emissivity in ph cm^3 s^-1 bin^-1"""
        energ_o_bins = self.energ_bins
        energ_i_bins = energ_o_bins*(1+z)
        self.sess.set_response(energ_i_bins, raw=True)
        # This returns epsilon = ph cm^3 s^-1 bin^-1
        # (bin is range of E_in, hence we correct by (1+z))
        spec = self.sess.return_spectrum(kT, dolines=dolines,
                                         docont=docont,
                                         dopseudo=dopseudo)
        return spec/(1+z)

    def _get_channel_limits(self, emin, emax):
        if emin < self.chans_h['E_MIN'][0]:
            raise ValueError("Minimum energy below range")
        if emax > self.chans_h['E_MAX'][-1]:
            raise ValueError("Maximum energy above range")
        i_min = np.where(self.chans_h['E_MIN'] <= emin)[0][-1]
        i_max = np.where(self.chans_h['E_MAX'] >= emax)[0][0]
        return i_min, i_max

    def _get_convolved_spectrum(self, spec):
        return np.dot(spec, self.m_x_A)

    def get_convolved_spectrum_out(self, kT, z,
                                   dolines=True, docont=True,
                                   dopseudo=True):
        """ Returns outgoing emissivity in ph cm^5 s^-1 bin^-1"""
        spec_in = self.get_spectrum_in(kT, z, dolines=dolines,
                                       docont=docont,
                                       dopseudo=dopseudo)
        return self._get_convolved_spectrum(spec_in)

    def get_bandpass(self, emin, emax):
        imin, imax = self._get_channel_limits(emin, emax)
        bpass = np.sum(self.m_x_A[:, imin:imax+1], axis=1)
        return bpass

    def get_integrated_spectrum(self, kT, z, emin, emax,
                                dolines=True, docont=True,
                                dopseudo=True):
        """ Returns outgoing emissivity integrated
        over a band in ph cm^5 s^-1
        """
        spec_out = self.get_convolved_spectrum_out(kT, z, dolines=dolines,
                                                   docont=docont,
                                                   dopseudo=dopseudo)
        i_min, i_max = self._get_channel_limits(emin, emax)
        return np.sum(spec_out[i_min:i_max+1])

    def get_arbitrary_integrated_spectrum(self, fe, z, emin, emax):
        energ_o_bins = self.energ_bins
        energ_i_bins = energ_o_bins*(1+z)
        energ_i_bins = np.sqrt(energ_i_bins[1:]*energ_i_bins[:-1])
        farr = fe(energ_i_bins)*np.diff(energ_o_bins)/energ_i_bins
        f_M_A = self._get_convolved_spectrum(farr)
        i_min, i_max = self._get_channel_limits(emin, emax)
        return np.sum(f_M_A[i_min:i_max+1])

    def get_integrated_spectrum_interp(self, kT_min, kT_max, n_kT,
                                       zmax, n_z, emin, emax,
                                       dolines=True, docont=True,
                                       dopseudo=True):
        """ Returns interpolator for outgoing emissivity integrated
        over a band in ph cm^5 s^-1. Interpolation is done in log(kT)
        and z, and the quantity interpolated is log(J), so output
        object should be evaluated as:

        J(kT,z) = np.exp(interp(np.log(kT), z))

        wth kT in keV.
        """
        from scipy.interpolate import RegularGridInterpolator

        kT_arr = np.geomspace(kT_min, kT_max, n_kT)
        z_arr = np.linspace(0, zmax, n_z)
        specs = np.array([[self.get_integrated_spectrum(kT, z, emin, emax,
                                                        dolines=dolines,
                                                        docont=docont,
                                                        dopseudo=dopseudo)
                           for z in z_arr] for kT in kT_arr])
        interp = RegularGridInterpolator([np.log(kT_arr), z_arr],
                                         np.log(specs),
                                         method='linear', fill_value=None,
                                         bounds_error=False)
        return interp

    def get_spectrum_interp_cached(self, lines=True,
                                   kTmin=0.02, kTmax=50.0, nkT=32,
                                   zmax=4.0, nz=16, emin=0.5, emax=2.0):
        kTmin = 0.02
        kTmax = 50.0
        nkT = 32
        zmax = 4.0
        nz = 16
        emin = 0.5
        emax = 2.0

        fname = 'data/J'
        if lines:
            fname += 'tot'
        else:
            fname += 'cont'
        fname += '_Z%.2lf' % self.Zmetal
        fname += '_Emin%.2lf' % emin
        fname += '_Emax%.2lf' % emax
        if self.arf_h is not None:
            fname += '_padb'
        fname += '.pck'
        if os.path.isfile(fname):
            print(f"Found {fname}")
            with open(fname, "rb") as f:
                J = pickle.load(f)
        else:
            J = self.get_integrated_spectrum_interp(kTmin, kTmax, nkT,
                                                    zmax, nz, emin, emax,
                                                    dolines=lines,
                                                    dopseudo=lines)
            with open(fname, "wb") as f:
                pickle.dump(J, f)
        return J
