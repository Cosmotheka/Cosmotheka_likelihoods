import astropy.io.fits as pyfits
import numpy as np
import os
import hashlib
import pickle
from . import atomic, util, const, atomdb, apec
from scipy.sparse import bsr_array
import warnings


def __get_index(te, filename='$ATOMDB/apec_line.fits',
                teunits='keV', logscale=False):

    if teunits.lower() == 'kev':
        teval = te
    elif teunits.lower() == 'k':
        teval = te*const.KBOLTZ
    else:
        print("*** ERROR: unknown temeprature unit %s. \
              Must be keV or K. Exiting ***" %
              (teunits))
    if isinstance(filename, pyfits.fits.hdu.hdulist.HDUList):
        a = filename[1].data
    elif isinstance(filename, pyfits.fits.hdu.table.BinTableHDU):
        a = filename.data
    elif isinstance(filename, str):
        a = pyfits.open(os.path.expandvars(filename))[1].data
    if logscale:
        i = np.argmin(np.abs(np.log(a['kT'])-np.log(teval)))
    else:
        i = np.argmin(np.abs(a['kT']-teval))
    return i+2

# -------------------------------------------------------------------------------


def _expand_E_grid(eedges, n, Econt_in_full, cont_in_full):

    import scipy.integrate
    cont_in = cont_in_full[:n]
    Econt_in = Econt_in_full[:n]

    E_all = np.append(Econt_in, eedges)

    cont_tmp = np.interp(eedges, Econt_in, cont_in)
    C_all = np.append(cont_in, cont_tmp)

    iord = np.argsort(E_all)

    E_all = E_all[iord]
    C_all = C_all[iord]

    ihi = np.where(iord >= n)[0]
    cum_cont = scipy.integrate.cumtrapz(C_all, E_all, initial=0)
    C_out = np.zeros(len(eedges))

    C_out = cum_cont[ihi]

    cont = C_out[1:] - C_out[:-1]
    return cont

# -------------------------------------------------------------------------------


def _get_response_ebins(rmf):

    if type(rmf) == str:
        rmfdat = pyfits.open(rmf)
    elif type(rmf) == pyfits.hdu.hdulist.HDUList:
        rmfdat = rmf
    else:
        print("ERROR: unknown rmf type, %s" % (repr(type(rmf))))
        return
    try:
        k = rmfdat.index_of('MATRIX')
        matrixname = 'MATRIX'
    except KeyError:
        try:
            k = rmfdat.index_of('SPECRESP MATRIX')
            matrixname = 'SPECRESP MATRIX'
        except KeyError:
            print("Cannot find index for matrix in this data")
            raise

    specbins_in = rmfdat[matrixname].data['ENERG_LO']
    specbins_in = np.append(specbins_in, rmfdat[matrixname].data['ENERG_HI'][-1])

    specbins_out = rmfdat['EBOUNDS'].data['E_MIN']
    specbins_out = np.append(specbins_out, rmfdat['EBOUNDS'].data['E_MAX'][-1])

    return specbins_in, specbins_out

# -------------------------------------------------------------------------------


def __get_effective_area(rmf, arf=False):

    if arf:
        if type(arf) == str:
            arfdat = pyfits.open(arf)
        elif type(arf) == pyfits.hdu.hdulist.HDUList:
            arfdat = arf
        else:
            print("ERROR: unknown arf type, %s" % (repr(type(arf))))
            return
        arfarea = arfdat['SPECRESP'].data['SPECRESP']
    else:
        arfarea = 1.0

    if type(rmf) == str:
        rmfdat = pyfits.open(rmf)
    elif type(rmf) == pyfits.hdu.hdulist.HDUList:
        rmfdat = rmf
    else:
        print("ERROR: unknown rmf type, %s" % (repr(type(rmf))))
        return

    ebins_in, ebins_out = _get_response_ebins(rmf)

    area = np.zeros(len(ebins_in)-1, dtype=float)

    try:
        k = rmfdat.index_of('MATRIX')
        matrixname = 'MATRIX'
    except KeyError:
        try:
            k = rmfdat.index_of('SPECRESP MATRIX')
            matrixname = 'SPECRESP MATRIX'
        except KeyError:
            print("Cannot find index for matrix in this data")
            raise

    matname = 'MATRIX'
    if matname not in rmfdat[matrixname].data.names:
        matname = 'SPECRESP MATRIX'
        if matname not in rmfdat[matrixname].data.names:
            print("Error: Cannot find Matrix in rmf data")
            return
    for ibin, i in enumerate(rmfdat[matrixname].data):
        area[ibin] = sum(i[matname])

    area *= arfarea
    return ebins_in, area

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _Gaussian_CDF():

    def __init__(self):
        from scipy.stats import norm
        self.x = np.linspace(-6, 6, 2400)
        self.cdf = norm.cdf(self.x)
        self.broadentype = 'Gaussian'

    def broaden(self, centroid, width, ebins):

        etmp = (ebins - centroid) / width

        ret = np.interp(etmp, self.x, self.cdf)

        return ret

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _Lorentzian_CDF():

    def __init__(self):
        from scipy.stats import cauchy
        self.x = np.linspace(-12, 12, 4800)
        self.cdf = cauchy.cdf(self.x)
        self.broadentype = 'Lorentzian'

    def broaden(self, centroid, width, ebins):

        etmp = (ebins-centroid)/width

        ret = np.interp(etmp, self.x, self.cdf)

        return ret

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _Voigt_CDF():

    def __init__(self, sigma, gamma):
        from scipy.special import voigt_profile
        self.x = np.linspace(-12, 12, 2400)
        self.broadentype = 'Voigt'

    def __recalc(self, sigma, gamma):

        if ((self.sigma == sigma) &
                (self.gamma == gamma)):
            pass

        edges = (self.x[1:] + self.x[:-1]) / 2
        edges = np.append(edges[0]-(edges[1] - edges[0]),
                          edges,
                          edges[-1] + (edges[-1] - edges[-2]))

        dx = edges[1:] - edges[:-1]

        pdfmid = voigt_profile(self.x, sigma, gamma) * dx

        self.cdf = np.cumsum(pdfmid)

    def broaden(self, centroid, ebins, sigma, gamma, test_recalc=False):

        etmp = (ebins - centroid)

        if test_recalc:
            self.__recalc(sigma, gamma)

        ret = np.interp(etmp, self.x, self.cdf)

        return ret

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class CIESession():

    def __init__(self, linefile="$ATOMDB/apec_line.fits",
                 cocofile="$ATOMDB/apec_coco.fits",
                 elements=False,
                 abundset='AG89'):

        self.SessionType = 'CIE'
        self._session_initialise1(linefile, cocofile, elements, abundset)

        self.spectra = _CIESpectrum(self.linedata, self.cocodata)

        self._session_initialise2()

# -------------------------------------------------------------------------------

    def _session_initialise1(self, linefile, cocofile, elements, abundset):

        self.datacache = {}

        self._set_apec_files(linefile, cocofile)

        if util.keyword_check(elements):
            self.elements = elements
        else:
            if self.SessionType == 'CIE':
                self.elements = list(range(1, const.MAXZ_CIE + 1))
            elif self.SessionType in ['NEI', 'PShock', 'Kappa']:
                self.elements = list(range(1, const.MAXZ_NEI + 1))

        self.abundset = self.linedata[0].header['SABUND_SOURCE']
        self.default_abundset = self.linedata[0].header['SABUND_SOURCE']

        self.abundsetvector = np.zeros(const.MAXZ_CIE + 1)
        for Z in self.elements:
            self.abundsetvector[Z] = 1.0

        if util.keyword_check(abundset):
            self.set_abundset(abundset)

        self.abund = np.zeros(const.MAXZ_CIE + 1)

        for Z in self.elements:
            self.abund[Z] = 1.0

        self.response_set = False
        self.dolines = True
        self.docont = True
        self.dopseudo = True
        self.do_eebrems = True

        self.rmffile = False
        self.arffile = False
        self.raw_response = False

        self.verbose = False

# -------------------------------------------------------------------------------

    def _session_initialise2(self):

        self.set_broadening(False, broaden_limit=1e-18)
        self.cdf = _Gaussian_CDF()

    def set_broadening(self, thermal_broadening, broaden_limit=False,
                       velocity_broadening=0.0,
                       velocity_broadening_units='km/s',
                       thermal_broaden_temperature=None,
                       teunit='keV'):

        self.thermal_broadening = thermal_broadening
        if broaden_limit is not False:
            self.broaden_limit = broaden_limit

        if self.verbose:
            if self.thermal_broadening is True:
                print("Will thermally broaden lines with emissivity > %e ph cm3 s-1" % (self.broaden_limit))
            else:
                print("Will not thermally broaden lines")

        self.velocity_broadening = velocity_broadening

        allowed_velocity_broadening_units = ['km/s']

        if not velocity_broadening_units.lower() in ['km/s']:
            raise util.UnitsError("Error: velocity broadening units of %s is not in allowed set %s." %
                                  (velocity_broadening_units, repr(allowed_velocity_broadening_units)))
            return

        self.velocity_broadening_units = velocity_broadening_units

        self.spectra.thermal_broadening = self.thermal_broadening
        self.spectra.broaden_limit = self.broaden_limit

        self.spectra.velocity_broadening = self.velocity_broadening
        self.spectra.velocity_broadening_units = self.velocity_broadening_units
        if thermal_broaden_temperature is not None:
            T = util.convert_temp(thermal_broaden_temperature, teunit, 'K')
            self.thermal_broaden_temperature = T
            self.spectra.thermal_broaden_temperature = T
        else:
            self.thermal_broaden_temperature = None
            self.spectra.thermal_broaden_temperature = None

# -------------------------------------------------------------------------------

    def set_response(self, rmf, arf=False, raw=False, sparse=False):

        if raw is True:
            if sparse:
                warnings.warn("Sparse matrix requested with raw response. Ignoring.")

            self.specbins = rmf
            self.ebins_out = rmf
            self.specbin_units = 'keV'

            self.aeff = np.ones(len(rmf) - 1)

            self.response_set = True
            self.specbins_set = True
            self.arf = False
            self.ebins_checksum = hashlib.md5(self.specbins).hexdigest()
            self.raw_response = True
            self.response_type = 'raw'

        else:
            if util.keyword_check(arf):
                if type(arf) == str:
                    self.arffile = arf
                    self.arfdat = pyfits.open(arf)
                elif type(arf) == pyfits.hdu.hdulist.HDUList:
                    self.arfdat = arf
                    self.arffile = arf.filename()
                else:
                    print("ERROR: unknown arf type, %s" % (repr(type(arf))))
                    return

                self.arf = np.array(self.arfdat['SPECRESP'].data['SPECRESP'])

            else:
                self.arf = 1.0

            self.raw_response = False
            if type(rmf) == str:
                self.rmffile = rmf
                self.rmf = pyfits.open(rmf)
            elif type(rmf) == pyfits.hdu.hdulist.HDUList:
                self.rmf = rmf
                self.rmffile = rmf.filename()
            else:
                print("ERROR: unknown rmf type, %s" % (repr(type(rmf))))
                return

            ebins = self.rmf['EBOUNDS'].data['E_MIN']
            if ebins[-1] > ebins[0]:
                ebins = np.append(ebins, self.rmf['EBOUNDS'].data['E_MAX'][-1])
            else:
                ebins = np.append(self.rmf['EBOUNDS'].data['E_MAX'][0], ebins)

            try:
                k = self.rmf.index_of('MATRIX')
                matrixname = 'MATRIX'
            except KeyError:
                try:
                    k = self.rmf.index_of('SPECRESP MATRIX')
                    matrixname = 'SPECRESP MATRIX'
                except KeyError:
                    print("Cannot find index for matrix in this data")
                    raise

            if not sparse:
                chanoffset = self.rmf['EBOUNDS'].data['CHANNEL'][0]

                self.rmfmatrix = np.zeros([len(self.rmf[matrixname].data),
                                           len(self.rmf['EBOUNDS'].data)])
                for ibin, i in enumerate(self.rmf[matrixname].data):

                    lobound = 0

                    fchan = i['F_CHAN'] * 1
                    nchan = i['N_CHAN'] * 1

                    if np.isscalar(fchan):
                        fchan = np.array([fchan])
                    fchan -= chanoffset
                    if np.isscalar(nchan):
                        nchan = np.array([nchan])

                    for j in range(len(fchan)):
                        ilo = fchan[j]
                        if ilo < 0:
                            continue

                        ihi = fchan[j] + nchan[j]
                        self.rmfmatrix[ibin, ilo:ihi] = i['MATRIX'][lobound:lobound + nchan[j]]
                        lobound = lobound + nchan[j]

                self.specbins, self.ebins_out = _get_response_ebins(self.rmf)
                self.specbins = self.specbins

                if self.ebins_out[-1] < self.ebins_out[0]:
                    self.ebins_out = self.ebins_out[::-1]
                    self.rmfmatrix = self.rmfmatrix[:, ::-1]

                self.specbin_units = 'keV'
                self.aeff = self.rmfmatrix.sum(1)
                if util.keyword_check(self.arf):
                    self.aeff *= self.arf
                self.response_set = True
                self.specbins_set = True
                self.response_type = 'standard'

                self.ebins_checksum = hashlib.md5(self.specbins).hexdigest()

            else:
                data = []
                row = []
                col = []

                chanoffset = self.rmf['EBOUNDS'].data['CHANNEL'][0]

                for ibin, i in enumerate(self.rmf[matrixname].data):
                    lobound = 0
                    for ngrp in range(i['N_GRP']):
                        fchan = i['F_CHAN']*1
                        nchan = i['N_CHAN']*1

                        if np.isscalar(fchan):
                            fchan = np.array([fchan])
                        fchan -= chanoffset
                        if np.isscalar(nchan):
                            nchan = np.array([nchan])

                        for j in range(len(fchan)):
                            ilo = fchan[j]
                            if ilo < 0:
                                continue

                            ihi = fchan[j] + nchan[j]
                            data.extend(i['MATRIX'][lobound:lobound + nchan[j]])
                            row.extend(range(ilo, ihi))
                            col.extend([ibin] * nchan[j])
                            lobound = lobound + nchan[j]

                self.specbins, self.ebins_out = _get_response_ebins(self.rmf)
                self.specbins = self.specbins

                data = np.array(data)
                row = np.array(row)
                col = np.array(col)

                if self.ebins_out[-1] < self.ebins_out[0]:
                    self.ebins_out = self.ebins_out[::-1]
                    col = len(self.ebins_out) - col-1

                self.rmfmatrix = bsr_array((data, (row, col)),
                                           shape=(len(self.specbins) - 1,
                                           len(self.ebins_out) - 1),
                                           dtype=data.dtype)

                self.specbin_units = 'keV'
                self.aeff = self.rmfmatrix.sum(1)
                if util.keyword_check(self.arf):
                    self.aeff *= self.arf
                self.response_set = True
                self.specbins_set = True
                self.response_type = 'sparse'

                self.ebins_checksum = hashlib.md5(self.specbins).hexdigest()

        if self.specbins_set:
            if self.specbins[0] <= 0:
                warnings.warn('Response minimum energy is 0 keV, setting to small finite value (%e keV)' % (self.specbins[1] * 1e-6))
                self.specbins[0] = self.specbins[1] * 1e-6

# -------------------------------------------------------------------------------

    def return_spectrum(self, te, redshift=0.0, teunit='keV', nearest=False,
                        get_nearest_t=False, log_interp=True,
                        dolines=True, docont=True, dopseudo=True):

        if not self.response_set:
            raise util.ReadyError("Response not yet set: use set_response to set.")

        el_list = self.elements
        ab = {}
        for Z in el_list:
            ab[Z] = self.abund[Z]*self.abundsetvector[Z]

        self.spectra.ebins = self.specbins * (1 + redshift)
        self.spectra.ebins_checksum = hashlib.md5(self.spectra.ebins).hexdigest()
        s = self.spectra.return_spectrum(te, teunit=teunit, nearest=nearest,
                                         elements=el_list, abundance=ab,
                                         broaden_object=self.cdf,
                                         log_interp=log_interp, dolines=dolines,
                                         dopseudo=dopseudo, docont=docont,
                                         do_eebrems=self.do_eebrems)
        ss = self._apply_response(s)

        return ss / (1 + redshift)

# -------------------------------------------------------------------------------

    def return_integrated_spectrum(self, te, emin, emax, redshift=0.0, 
                                   teunit='keV', nearest=False,
                                   get_nearest_t=False, log_interp=True,
                                   dolines=True, docont=True, dopseudo=True):

        spec = self.return_spectrum(te, redshift=redshift, teunit=teunit, nearest=nearest,
                               get_nearest_t=get_nearest_t, log_interp=log_interp,
                               dolines=dolines, docont=docont, dopseudo=dopseudo)

        if emin < self.rmf['EBOUNDS'].data['E_MIN'][0]:
            raise ValueError("Minimum energy below range")
        if emax > self.rmf['EBOUNDS'].data['E_MAX'][-1]:
            raise ValueError("Maximum energy above range")
        i_min = np.where(self.rmf['EBOUNDS'].data['E_MIN'] <= emin)[0][-1]
        i_max = np.where(self.rmf['EBOUNDS'].data['E_MAX'] >= emax)[0][0]

        return np.sum(spec[i_min:i_max+1])

# -------------------------------------------------------------------------------

    def _apply_response(self, spectrum):

        if self.response_type == 'raw':
            return spectrum

        elif self.response_type == 'standard':
            arfdat = self.arf

            ret = spectrum * self.arf

            try:
                ret = np.matmul(ret, self.rmfmatrix)
            except ValueError:
                try:
                    ret = np.matmul(ret, self.rmfmatrix.transpose())
                except ValueError:
                    if ret == 0:
                        ret = np.zeros(len(self.ebins_out) - 1)
            return ret
        elif self.response_type == 'sparse':
            arfdat = self.arf
            ret = spectrum * self.arf
            ret = (self.rmfmatrix * ret).sum(1)
            pickle
            return ret

        else:
            raise util.OptionError('Unknown response type %s' % (self.response_type))

# -------------------------------------------------------------------------------

    def _set_apec_files(self, linefile, cocofile):

        if isinstance(linefile, str):
            lfile = os.path.expandvars(linefile)
            if not os.path.isfile(lfile):
                print("*** ERROR: no such file %s. Exiting ***" % (lfile))
                return -1
            self.linedata = pyfits.open(lfile)
            self.linefile = lfile

        elif isinstance(linefile, pyfits.hdu.hdulist.HDUList):
            self.linedata = linefile
            self.linefile = linefile.filename()

        else:
            print("Unknown data type for linefile. Please pass a string or an HDUList")

        if isinstance(cocofile, str):

            cfile = os.path.expandvars(cocofile)
            if not os.path.isfile(cfile):
                print("*** ERROR: no such file %s. Exiting ***" % (cfile))
                return -1
            self.cocodata = pyfits.open(cfile)
            self.cocofile = cfile

        elif isinstance(cocofile, pyfits.hdu.hdulist.HDUList):
            self.cocodata = cocofile
            self.cocofile = cocofile.filename()

        else:
            print("Unknown data type for cocofile. Please pass a string or an HDUList")

# -------------------------------------------------------------------------------

    def set_eebrems(self, do_eebrems):

        a = bool(do_eebrems)
        self.do_eebrems = a

# -------------------------------------------------------------------------------

    def set_abund(self, elements, abund):

        abundvec, aisvec = util.make_vec(abund)
        elementvec, eisvec = util.make_vec(elements)
        if (aisvec):
            if len(abundvec) != len(elementvec):

                print("abundance vector and element vector must have same number" +
                      " of elements")
                print('ab', abundvec)
                print('el', elementvec)
            else:

                self.abund[elementvec] = abundvec
        elif (eisvec):
            for el in elementvec:
                self.abund[el] = abund

        else:
            self.abund[elements] = abund

# -------------------------------------------------------------------------------

    def set_abundset(self, abundstring=None):

        if abundstring is None:

            abunddata = atomdb.get_data(False, False, 'abund', datacache=self.datacache)

            print("Possible abundance sets:")
            for name in abunddata[1].data.field('Source'):
                print("    %s" % (name))
            return

        old = atomdb.get_abundance(abundset=self.default_abundset,
                                   datacache=self.datacache)

        new = atomdb.get_abundance(abundset=abundstring,
                                   datacache=self.datacache)

        for Z in range(1, const.MAXZ_CIE + 1):
            try:
                self.abundsetvector[Z] = new[Z] / old[Z]
            except ZeroDivisionError:
                self.abundsetvector[Z] = 0.0
            except IndexError:
                pass

        self.abundset = abundstring

# -------------------------------------------------------------------------------

    def return_line_emissivity(self, Te, Z, z1, up, lo,
                               specunit='A', teunit='keV',
                               apply_aeff=False, apply_abund=True,
                               log_interp=True):

        Tevec, Teisvec = util.make_vec(Te)

        kTlist = util.convert_temp(Tevec, teunit, 'keV')
        if apply_abund:
            ab = self.abund[Z]*self.abundsetvector[Z]
        else:
            ab = 1.0

        eps = np.zeros(len(Tevec))
        ret = {}
        ret['wavelength'] = None
        for ikT, kT in enumerate(kTlist):
            e, lam = self.spectra.return_line_emissivity(kT, Z, z1,
                                                         up, lo,
                                                         specunit='A',
                                                         teunit='keV',
                                                         abundance=ab)

            eps[ikT] = e
            if lam is not False:
                ret['wavelength'] = lam * 1.0

        ret['Te'] = Te
        ret['teunit'] = teunit
        if ret['wavelength'] is not None:
            ret['energy'] = const.HC_IN_KEV_A / ret['wavelength']
        else:
            ret['energy'] = None

        if apply_aeff is True:
            e = ret['energy']
            ibin = np.where(self.specbins < e)[0][-1]

            eps = eps*self.aeff[ibin]

#         now correct for vectors

        if not Teisvec:
            eps = eps[0]

        ret['epsilon'] = eps

        return ret

# -------------------------------------------------------------------------------

    def return_linelist(self, Te, specrange, specunit='A',
                        teunit='keV', apply_aeff=False, nearest=False,
                        apply_binwidth=False):
        kT = util.convert_temp(Te, teunit, 'keV')

        el_list = self.elements
        ab = {}
        for Z in el_list:
            ab[Z] = self.abund[Z]*self.abundsetvector[Z]

        s = self.spectra.return_linelist(kT, specrange=specrange, teunit='keV',
                                         specunit=specunit, elements=el_list,
                                         abundance=ab, nearest=nearest)

        if apply_aeff is True:

            epsilon_aeff = self._apply_linelist_aeff(s, specunit, apply_binwidth)

            s['Epsilon'] = epsilon_aeff
        return s

# -------------------------------------------------------------------------------

    def _apply_linelist_aeff(self, linelist,
                             specunit, apply_binwidth):

        if specunit.lower() == 'kev':
            binwidth = self.ebins_out[1:] - self.ebins_out[:-1]
            factor = np.zeros(len(s), dtype=float)
            for i, ss in enumerate(linelist):
                e = const.HC_IN_KEV_A / ss['Lambda']
                if e > self.specbins[-1]:
                    factor[i] = 0.0
                elif e < self.specbins[0]:
                    factor[i] = 0.0
                else:
                    ibin = np.where(self.specbins < e)[0][-1]
                    factor[i] = self.aeff[ibin]
                    if apply_binwidth:
                        factor[i] /= binwidth[ibin]

            emiss_aeff = linelist['Epsilon'] * factor

        elif specunit.lower() == 'a':
            wvbins = 12.398425 / self.ebins_out[::-1]
            binwidth = wvbins[1:] - wvbins[:-1]
            factor = np.zeros(len(linelist), dtype=float)
            for i, ss in enumerate(linelist):
                e = ss['Lambda']
                if e > wvbins[-1]:
                    factor[i] = 0.0
                elif e < wvbins[0]:
                    factor[i] = 0.0
                else:
                    ibin = np.where(wvbins < e)[0][-1]
                    factor[i] = self.aeff[::-1][ibin]
                    if apply_binwidth:
                        factor[i] /= binwidth[ibin]
            emiss_aeff = linelist['Epsilon']*factor
        return emiss_aeff

# -------------------------------------------------------------------------------

    def _adjust_line(self, change, Z=0, z1=0, z1_drv=0,
                     upper=0, lower=0,
                     quantity="Epsilon",
                     method="Replace",
                     trackchanges=False):

        meth = method.lower()

        if Z == 0:
            Zlist = self.elements
        else:
            Zlist = [Z]
        for Zt in Zlist:
            if z1 == 0:
                z1list = range(1, z1 + 2)
            else:
                z1list = [z1]
            for z1t in z1list:
                if z1_drv == 0:
                    if self.SessionType == 'CIE':
                        z1_drvlist = [0]
                    else:
                        z1_drvlist = range(1, z1 + 2)
                else:
                    if self.SessionType == 'CIE':
                        z1_drvlist = [0]
                    else:
                        z1_drvlist = [z1_drv]

                for z1_drvt in z1_drvlist:

                    for ikT in range(len(self.spectra.kTlist)):

                        try:
                            ldat = self.spectra.spectra[ikT][Zt].lines.lines
                        except KeyError:
                            continue

                        tochange = np.ones(len(ldat), dtype=bool)

                        if upper != 0:
                            tochange[ldat['UpperLev'] != upper] = False
                        if lower != 0:
                            tochange[ldat['LowerLev'] != lower] = False

                        if trackchanges:
                            tochange[self.spectra.spectra[ikT][Zt].lines.changed is True] = False

                        if sum(tochange) > 0:

                            if meth == 'replace':
                                ldat[quantity][tochange] = change
                            if meth == 'add':
                                ldat[quantity][tochange] += change
                            if meth == 'subtract':
                                ldat[quantity][tochange] -= change
                            if meth == 'divide':
                                ldat[quantity][tochange] /= change
                            if meth == 'multiply':
                                ldat[quantity][tochange] *= change

                            self.spectra.spectra[ikT][Zt].lines.changed[tochange] = True

        return

# -------------------------------------------------------------------------------

    def _adjust_line_lambda(self, change, Z, z1,
                            upper, lower,
                            quantity="Epsilon",
                            method="Replace",
                            trackchanges=False):

        meth = method.lower()

        try:
            value = self.spectra.fixwavelength[Z][z1][upper][lower]
        except AttributeError:
            self.spectra.fixwavelength = {}

        if Z not in self.spectra.fixwavelength.keys():
            self.spectra.fixwavelength[Z] = {}

        if z1 not in self.spectra.fixwavelength[Z].keys():
            self.spectra.fixwavelength[Z][z1] = {}

        if upper not in self.spectra.fixwavelength[Z][z1].keys():
            self.spectra.fixwavelength[Z][z1][upper] = {}

        if lower not in self.spectra.fixwavelength[Z][z1][upper].keys():
            self.spectra.fixwavelength[Z][z1][upper][lower] = quantity

        if Z == 0:
            Zlist = self.elements
        else:
            Zlist = [Z]
        for Zt in Zlist:
            if z1 == 0:
                z1list = range(1, z1 + 2)
            else:
                z1list = [z1]
            for z1t in z1list:
                if z1_drv == 0:
                    if self.SessionType == 'CIE':
                        z1_drvlist = [0]
                    else:
                        z1_drvlist = range(1, z1 + 2)
                else:
                    if self.SessionType == 'CIE':
                        z1_drvlist = [0]
                    else:
                        z1_drvlist = [z1_drv]

                for z1_drvt in z1_drvlist:

                    for ikT in range(len(self.spectra.kTlist)):

                        try:
                            ldat = self.spectra.spectra[ikT][Zt].lines.lines
                        except KeyError:
                            continue

                        tochange = np.ones(len(ldat), dtype=bool)

                        if upper != 0:
                            tochange[ldat['UpperLev'] != upper] = False
                        if lower != 0:
                            tochange[ldat['LowerLev'] != lower] = False

                        if trackchanges:
                            tochange[self.spectra.spectra[ikT][Zt].lines.changed is True] = False

                        if sum(tochange) > 0:

                            if meth == 'replace':
                                ldat[quantity][tochange] = change
                            if meth == 'add':
                                ldat[quantity][tochange] += change
                            if meth == 'subtract':
                                ldat[quantity][tochange] -= change
                            if meth == 'divide':
                                ldat[quantity][tochange] /= change
                            if meth == 'multiply':
                                ldat[quantity][tochange] *= change

                            self.spectra.spectra[ikT][Zt].lines.changed[tochange] = True

        return

# -------------------------------------------------------------------------------


class _CIESpectrum():

    def __init__(self, linedata, cocodata):

        self.datacache = {}
        self.SessionType = 'CIE'

        picklefname = os.path.expandvars('$ATOMDB/spectra_%s_%s.pkl' %
                                         (linedata[0].header['CHECKSUM'],
                                          cocodata[0].header['CHECKSUM']))
        havepicklefile = False
        if os.path.isfile(picklefname):
            havepicklefile = True

        if havepicklefile:
            try:
                self.spectra = pickle.load(open(picklefname, 'rb'))
                self.kTlist = self.spectra['kTlist']
            except AttributeError:
                havepicklefile = False
                print("pre-stored data in %s is out of date. This can be caused by updates to the data " % (picklefname) +
                      "or, more likely, changes to pyatomdb. Regenerating...")

                if os.path.isfile(picklefname):
                    os.remove(picklefname)

        if not havepicklefile:
            self.spectra = {}
            self.kTlist = np.array(linedata[1].data['kT'].data)
            self.spectra['kTlist'] = np.array(linedata[1].data['kT'].data)

            for ihdu in range(len(self.kTlist)):
                self.spectra[ihdu] = {}
                self.spectra[ihdu]['kT'] = self.kTlist[ihdu]
                ldat = np.array(linedata[ihdu + 2].data.data)
                cdat = np.array(cocodata[ihdu + 2].data.data)

                Zarr = np.zeros([len(ldat), const.MAXZ_CIE + 1], dtype=bool)
                Zarr[np.arange(len(ldat), dtype=int), ldat['Element']] = True

                for Z in range(1, const.MAXZ_CIE + 1):
                    ccdat = cdat[(cdat['Z'] == Z) & (cdat['rmJ'] == 0)]

                    if len(ccdat) == 1:
                        c = ccdat[0]
                    else:
                        c = False

                    self.spectra[ihdu][Z] = _ElementSpectrum(ldat[Zarr[:, Z]], c, Z)
            pickle.dump(self.spectra, open(picklefname, 'wb'))

        self.logkTlist = np.log(self.kTlist)

# -------------------------------------------------------------------------------

    def get_nearest_Tindex(self, Te, teunit='keV', nearest=False,
                           log_interp=True):

        kT = util.convert_temp(Te, teunit, 'keV')

        if kT < self.kTlist[0]:
            print("kT = %f is below minimum range of %f. Returning lowest kT spectrum available" %
                  (kT, self.kTlist[0]))
            ikT = [0]
            f = [1.0]

        elif kT > self.kTlist[-1]:
            print("kT = %f is above maximum range of %f. Returning highest kT spectrum available" %
                  (kT, self.kTlist[-1]))
            ikT = [len(self.kTlist) - 1]
            f = [1.0]
        else:

            if log_interp:
                if nearest:
                    ikT = [np.argmin(np.abs(self.logkTlist - np.log(kT)))]
                    f = [1.0]
                else:
                    ikT = np.where(self.kTlist < kT)[0][-1]
                    ikT = [ikT, ikT + 1]

                    f = 1 - (np.log(kT) - self.logkTlist[ikT[0]]) / (self.logkTlist[ikT[1]] - self.logkTlist[ikT[0]])
                    f = [f, 1 - f]

            else:
                if nearest:
                    ikT = [np.argmin(np.abs(self.kTlist - kT))]
                    f = [1.0]
                else:
                    ikT = np.where(self.kTlist < kT)[0][-1]
                    ikT = [ikT, ikT + 1]

                    f = 1 - (kT-self.kTlist[ikT[0]]) / (self.kTlist[ikT[1]] - self.kTlist[ikT[0]])
                    f = [f, 1 - f]

        return ikT, f

# -------------------------------------------------------------------------------

    def calc_ee_brems_spec(ebins, Te, dens, teunit='keV'):

        kT = util.convert_temp(Te, teunit, 'keV')

        eespec = apec.calc_ee_brems(ebins, kT, dens)

        ee = (ebins[1:]-ebins[:-1]) * (eespec[1:] + eespec[:-1]) / 2

        return ee

# -----------------------------------------------------------------------

    def return_spectrum(self, Te, teunit='keV', nearest=False,
                        elements=False, abundance=False, log_interp=True,
                        broaden_object=False,
                        dolines=True, docont=True, dopseudo=True,
                        do_eebrems=False):

        kT = util.convert_temp(Te, teunit, 'keV')

        ikT, f = self.get_nearest_Tindex(kT, teunit='keV', nearest=nearest, log_interp=log_interp)

        if elements is False:
            elements = range(1, const.MAXZ_CIE + 1)

        if abundance is False:
            abundance = {}
            for Z in elements:
                abundance[Z] = 1.0

        s = 0.0

        if do_eebrems:
            nel = 0.0
            rawabund = atomdb.get_abundance(datacache=self.datacache)

        if self.thermal_broaden_temperature is not None:
            Tb = util.convert_temp(self.thermal_broaden_temperature, 'K', 'keV')
        else:
            Tb = kT
        for Z in elements:
            abund = abundance[Z]
            if abund > 0:
                epslimit = self.broaden_limit / abund

                if len(ikT) == 1:
                    ss = self.spectra[ikT[0]][Z].return_spectrum(self.ebins,
                                                                 Tb,
                                                                 ebins_checksum=self.ebins_checksum,
                                                                 thermal_broadening=self.thermal_broadening,
                                                                 broaden_limit=epslimit,
                                                                 velocity_broadening=self.velocity_broadening,
                                                                 broaden_object=broaden_object,
                                                                 dolines=dolines,
                                                                 docont=docont,
                                                                 dopseudo=dopseudo) *\
                        abund

                else:
                    ss1 = self.spectra[ikT[0]][Z].return_spectrum(self.ebins,
                                                                  Tb,
                                                                  ebins_checksum=self.ebins_checksum,
                                                                  thermal_broadening=self.thermal_broadening,
                                                                  broaden_limit=epslimit,
                                                                  velocity_broadening=self.velocity_broadening,
                                                                  broaden_object=broaden_object,
                                                                  dolines=dolines,
                                                                  docont=docont,
                                                                  dopseudo=dopseudo) *\
                        abund

                    ss2 = self.spectra[ikT[1]][Z].return_spectrum(self.ebins,
                                                                  Tb,
                                                                  ebins_checksum=self.ebins_checksum,
                                                                  thermal_broadening=self.thermal_broadening,
                                                                  broaden_limit=epslimit,
                                                                  velocity_broadening=self.velocity_broadening,
                                                                  broaden_object=broaden_object,
                                                                  dolines=dolines,
                                                                  docont=docont,
                                                                  dopseudo=dopseudo) *\
                        abund

                    ss = self._merge_spectra_temperatures(f, ss1, ss2, log_interp)

                s += ss
                if do_eebrems:
                    ionpop = apec.return_ionbal(Z, kT, datacache=self.datacache, teunit='keV')
                    Zabundance = rawabund[Z]*abund

                    tmp = sum(ionpop*np.arange(Z + 1)) * Zabundance
                    nel += tmp

        if do_eebrems:
            eespec = calc_ee_brems_spec(self.ebins, kT, nel)

            try:
                nH = rawabund[1] * abundance[1]
            except KeyError:
                print("Warning: as this plasma has no hydrogen in it, assuming nH=1 for electron-electron bremstrahlung renorm")
                nH = 1.0

            s += eespec / (nel / nH)

        return s

# -------------------------------------------------------------------------------

    def return_line_emissivity(self, Te, Z, z1, up, lo, specunit='A',
                               teunit='keV', abundance=1.0,
                               log_interp=True):

        kT = util.convert_temp(Te, teunit, 'keV')

        ikT, f = self.get_nearest_Tindex(kT,
                                         teunit='keV',
                                         nearest=False,
                                         log_interp=log_interp)

        eps_in = np.zeros(len(ikT))
        eps = 0.0
        lam = 0.0
        for i in range(len(ikT)):
            iikT = ikT[i]

            llist = self.spectra[iikT][Z].return_linematch(Z, z1, up, lo)
            for line in llist:
                eps_in[i] += line['Epsilon']
                lam = line['Lambda']

        if log_interp:
            eps_out = 0.0
            for i in range(len(ikT)):
                eps_out += f[i] * np.log(eps_in[i] + const.MINEPSOFFSET)
            eps += np.exp(eps_out - const.MINEPSOFFSET) * abundance
        else:
            eps_out = 0.0
            for i in range(len(ikT)):
                eps_out += f[i] * eps_in[i]
            eps += eps_out * abundance

        if specunit == 'keV':
            lam = const.HC_IN_KEV_A / lam
        return eps, lam

# -------------------------------------------------------------------------------

    def _merge_spectra_temperatures(self, f, spec1, spec2, log_interp):

        if log_interp:
            out = np.log(spec1 + const.MINEPSOFFSET) * f[0] +\
                np.log(spec2 + const.MINEPSOFFSET) * f[1]
            spec = np.exp(out) - const.MINEPSOFFSET

        else:
            spec = spec1 * f[0] + spec2 * f[1]

        spec[spec < 0] = 0.0

        return spec

# -------------------------------------------------------------------------------

    def _merge_linelist_duplicates(self, llist, by_ion_drv=False):

        if by_ion_drv:
            llist = np.sort(llist, order=['Ion', 'Ion_drv', 'UpperLev', 'LowerLev'])
        else:
            llist = np.sort(llist, order=['Ion', 'UpperLev', 'LowerLev'])

        keep = np.ones(len(llist), dtype=bool)

        if by_ion_drv:
            j = np.where((llist[1:]['Ion'] == llist[:-1]['Ion']) &
                         (llist[1:]['Ion_drv'] == llist[:-1]['Ion_drv']) &
                         (llist[1:]['UpperLev'] == llist[:-1]['UpperLev']) &
                         (llist[1:]['LowerLev'] == llist[:-1]['LowerLev']))[0]
        else:
            j = np.where((llist[1:]['Ion'] == llist[:-1]['Ion']) &
                         (llist[1:]['UpperLev'] == llist[:-1]['UpperLev']) &
                         (llist[1:]['LowerLev'] == llist[:-1]['LowerLev']))[0]

        for jj in j:
            llist['Epsilon'][jj + 1] += llist['Epsilon'][jj]
            keep[jj] = False

        llist_out = llist[keep]

        return llist_out

# -------------------------------------------------------------------------------

    def _merge_linelists_temperatures(self, f, llist1, llist2, log_interp, by_ion_drv=False):

        llist1 = self._merge_linelist_duplicates(llist1, by_ion_drv=by_ion_drv)
        llist2 = self._merge_linelist_duplicates(llist2, by_ion_drv=by_ion_drv)

        llist = np.append(llist1, llist2)

        if log_interp:
            llist[:len(llist1)]['Epsilon'] = np.log(llist[:len(llist1)]['Epsilon'])*f[0]
            llist[len(llist1):]['Epsilon'] = np.log(llist[len(llist1):]['Epsilon'])*f[1]
        else:
            llist[:len(llist1)]['Epsilon'] = llist[:len(llist1)]['Epsilon']*f[0]
            llist[len(llist1):]['Epsilon'] = llist[len(llist1):]['Epsilon']*f[1]

        if by_ion_drv:
            llist = np.sort(llist, order=['Ion', 'Ion_drv', 'UpperLev', 'LowerLev'])
        else:
            llist = np.sort(llist, order=['Ion', 'UpperLev', 'LowerLev'])

        keep = np.zeros(len(llist), dtype=bool)

        if by_ion_drv:
            j = np.where((llist[1:]['Ion'] == llist[:-1]['Ion']) &
                         (llist[1:]['Ion_drv'] == llist[:-1]['Ion_drv']) &
                         (llist[1:]['UpperLev'] == llist[:-1]['UpperLev']) &
                         (llist[1:]['LowerLev'] == llist[:-1]['LowerLev']))[0]
        else:
            j = np.where((llist[1:]['Ion'] == llist[:-1]['Ion']) &
                         (llist[1:]['UpperLev'] == llist[:-1]['UpperLev']) &
                         (llist[1:]['LowerLev'] == llist[:-1]['LowerLev']))[0]

        for jj in j:
            llist['Epsilon'][jj+1] += llist['Epsilon'][jj]

            keep[jj + 1] = True

        llist = llist[keep]
        if log_interp:
            llist['Epsilon'] = np.exp(llist['Epsilon'])

        return llist

# -------------------------------------------------------------------------------

    def return_linelist(self, Te, teunit='keV', nearest=False,
                        specrange=False, specunit='A',
                        elements=False, abundance=False,
                        log_interp=True):

        kT = util.convert_temp(Te, teunit, 'keV')

        ikT, f = self.get_nearest_Tindex(kT, teunit='keV', nearest=nearest)

        if elements is False:
            elements = range(1, const.MAXZ_CIE + 1)

        if abundance is False:
            abundance = {}
            for Z in elements:
                abundance[Z] = 1.0

        linelist = np.zeros(0, dtype=apec.generate_datatypes('linelist_cie_spectrum'))

        for Z in elements:
            abund = abundance[Z]
            if abund > 0:

                if len(ikT) > 1:

                    llist1 = self.spectra[ikT[0]][Z].return_linelist(specrange,
                                                                     specunit=specunit)
                    llist2 = self.spectra[ikT[1]][Z].return_linelist(specrange,
                                                                     specunit=specunit)

                    elemlinelist = self._merge_linelists_temperatures(f, llist1, llist2,
                                                                      log_interp)

                else:
                    elemlinelist = self.spectra[ikT[0]][Z].return_linelist(specrange,
                                                                           specunit=specunit)

                elemlinelist['Epsilon'] *= abund

                if len(linelist) == 0:
                    linelist = elemlinelist
                else:
                    linelist = np.append(linelist, elemlinelist)

        return linelist

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _ElementSpectrum():

    def __init__(self, linedata, cocodata, Z, z1_drv=0):

        if z1_drv != 0:
            tmp = linedata[(linedata['Element'] == Z) &
                           (linedata['Ion_drv'] == z1_drv)]
            self.lines = _LineData(tmp)
            self.continuum = _ContinuumData(cocodata)

        else:
            self.lines = _LineData(linedata)
            self.continuum = _ContinuumData(cocodata)

# -------------------------------------------------------------------------------

    def return_spectrum(self, eedges, Te, ebins_checksum=False,
                        thermal_broadening=False,
                        broaden_limit=False,
                        velocity_broadening=0.0,
                        teunit='keV',
                        broaden_object=False,
                        dolines=True,
                        docont=True,
                        dopseudo=True):

        T = util.convert_temp(Te, teunit, 'K')

        if ebins_checksum is False:
            ebins_checksum = hashlib.md5(eedges).hexdigest()

        self.ebins_checksum = ebins_checksum
        self.T = T
        spec = np.zeros(len(eedges)-1)

        if dolines:
            spec += self.lines.return_spec(eedges, T,
                                           ebins_checksum=ebins_checksum,
                                           thermal_broadening=thermal_broadening,
                                           broaden_limit=broaden_limit,
                                           velocity_broadening=velocity_broadening,
                                           broaden_object=broaden_object)
        if dopseudo + docont > 0:
            spec += self.continuum.return_spec(eedges, ebins_checksum=ebins_checksum,
                                               dopseudo=dopseudo, docont=docont)

        self.spectrum = spec

        return self.spectrum

# -------------------------------------------------------------------------------

    def return_linelist(self, specrange, specunit='A'):

        wave = util.convert_spec(specrange, specunit, 'A')

        llist = self.lines.lines[(self.lines.lines['Lambda'] >= wave[0]) &
                                 (self.lines.lines['Lambda'] <= wave[1])]

        return llist

# -------------------------------------------------------------------------------

    def return_linematch(self, Z, z1, up, lo, z1_drv=0):

        llist = self.lines.lines[(self.lines.lines['Element'] == Z) &
                                 (self.lines.lines['Ion'] == z1) &
                                 (self.lines.lines['UpperLev'] == up) &
                                 (self.lines.lines['LowerLev'] == lo)]
        if z1_drv != 0:
            llist = llist[llist['Ion_drv'] == z1_drv]

        return llist

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _LineData():

    def __init__(self, linelist):

        self.lines = linelist
        self.lineenergies = const.HC_IN_KEV_A/self.lines['Lambda']
        self.spectrum_calculated = False
        self.T = 0.0
        self.v = 0.0
        self.ebins_checksum = False

# -------------------------------------------------------------------------------

    def return_spec(self, eedges, T, ebins_checksum=False,
                    thermal_broadening=False,
                    velocity_broadening=0.0,
                    broaden_limit=1e-18,
                    broaden_object=False):

        if ebins_checksum is False:
            ebins_checksum = hashlib.md5(eedges).hexdigest()
        if velocity_broadening is None:
            velocity_broadening = 0.0
        if ((thermal_broadening is False) &
                (velocity_broadening is False)):
            if ((self.ebins_checksum == ebins_checksum) &
                    (self.spectrum_calculated is True)):
                pass

            else:
                spec, z = np.histogram(self.lineenergies,
                                       bins=eedges,
                                       weights=self.lines['Epsilon'])
                self.spectrum = spec
                self.spectrum_calculated = True

        else:
            if ((thermal_broadening is True) &
                    (velocity_broadening <= 0)):
                if ((self.ebins_checksum == ebins_checksum) &
                        (self.spectrum_calculated is True) &
                        (self.T is T) &
                        (self.v <= 0.0)):
                    pass
                else:
                    recalc = True

            if ((thermal_broadening is False) &
                    (velocity_broadening > 0)):
                if ((self.ebins_checksum == ebins_checksum) &
                        (self.spectrum_calculated is True) &
                        (self.T is T) &
                        (self.v == velocity_broadening)):
                    pass
                else:
                    recalc = True

            if ((thermal_broadening is True) &
                    (velocity_broadening > 0)):

                if ((self.ebins_checksum == ebins_checksum) &
                        (self.spectrum_calculated is True) &
                        (self.T is T) &
                        (self.v == velocity_broadening)):
                    pass
                else:
                    recalc = True
            if recalc is True:
                ind = self.lines['Epsilon'] > broaden_limit
                nonind = ~ind

                llist = self.lines[ind]

                masslist = atomic.Z_to_mass(1, raw=True)

                if thermal_broadening is False:
                    T = 0.0
                    Tb = 0.0
                else:
                    Tb = util.convert_temp(T, 'K', 'keV')*const.ERG_KEV/(masslist[llist['Element']]*1e3*const.AMUKG)

                if velocity_broadening < 0:
                    vb = 0.0
                else:
                    vb = (velocity_broadening * 1e5)**2

                wcoeff = np.sqrt(Tb+vb) / (const.LIGHTSPEED*1e2)

                elines = self.lineenergies[ind]
                width = wcoeff*elines
                NSIGMALIMIT = 4
                eplu = elines+NSIGMALIMIT*width
                eneg = elines-NSIGMALIMIT*width
                emax = max(eedges)
                emin = min(eedges)
                igood = np.where(((elines >= emin) & (eneg < emax)) |
                                 ((elines < emin) & (eplu < emin)))[0]
                spec = np.zeros(len(eedges))

                for iline in igood:
                    spec += broaden_object.broaden(const.HC_IN_KEV_A / llist['Lambda'][iline],
                                                   width[iline], eedges) * llist['Epsilon'][iline]

                spec = spec[1:] - spec[:-1]
                s, z = np.histogram(self.lineenergies[nonind],
                                    bins=eedges,
                                    weights=self.lines['Epsilon'][nonind])
                spec += s
                self.spectrum = spec
                self.spectrum_calculated = True

                if thermal_broadening:
                    self.T = T
                else:
                    self.T = 0.0

                self.v = velocity_broadening
        return self.spectrum

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class _ContinuumData():

    def __init__(self, cocoentry):

        if type(cocoentry) == bool:
            if cocoentry is False:
                cocoentry = {}
                cocoentry['N_Cont'] = 2
                cocoentry['N_Pseudo'] = 2
                cocoentry['E_Cont'] = np.array([0.0001, 10000])
                cocoentry['Continuum'] = np.array([0.0, 0.0])
                cocoentry['E_Pseudo'] = np.array([0.0001, 10000])
                cocoentry['Pseudo'] = np.array([0.0, 0.0])

        nEC = cocoentry['N_Cont']
        nEP = cocoentry['N_Pseudo']

        self.ECont = cocoentry['E_Cont'][:nEC]
        self.EPseudo = cocoentry['E_Pseudo'][:nEP]

        self.Cont = cocoentry['Continuum'][:nEC]
        self.Pseudo = cocoentry['Pseudo'][:nEP]

        self.spectrum_calculated = False
        self.ebins_checksum = False

# -------------------------------------------------------------------------------

    def return_spec(self, eedges, ebins_checksum=False,
                    docont=True, dopseudo=True):

        if ebins_checksum is False:
            ebins_checksum = hashlib.md5(eedges).hexdigest()

        if docont:
            cont = _expand_E_grid(eedges, len(self.ECont),
                                  self.ECont,
                                  self.Cont)
        else:
            cont = 0.0

        if dopseudo:
            pseudo = _expand_E_grid(eedges, len(self.EPseudo),
                                    self.EPseudo,
                                    self.Pseudo)
        else:
            pseudo = 0.0

        self.ebins_checksum = ebins_checksum
        self.spectrum = cont + pseudo

        return self.spectrum

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def calc_ee_brems_spec(ebins, Te, dens, teunit='keV'):

    kT = util.convert_temp(Te, teunit, 'keV')

    eespec = apec.calc_ee_brems(ebins, kT, dens)

    ee = (ebins[1:]-ebins[:-1]) * (eespec[1:] + eespec[:-1]) / 2

    return ee
