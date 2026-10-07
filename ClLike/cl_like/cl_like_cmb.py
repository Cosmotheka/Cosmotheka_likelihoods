import numpy as np
import sacc
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


class ClLikeCMB(Likelihood):
    """Gaussian likelihood for binned CMB power spectra from a SACC file.

    Theory spectra come from the CLASS theory code (via `provider.get_Cl`),
    in muK^2 and without ell(ell+1)/2pi, and are binned with the SACC
    bandpower windows.
    """
    # Input sacc file
    input_file: str = ""
    # List of bin (tracer) names, e.g. [{'name': 'T'}, {'name': 'E'}]
    bins: list = []
    # Default settings (lmin, lmax)
    defaults: dict = {}
    # List of two-point functions: [{'bins': ['T', 'T']}, ...]
    twopoints: list = []
    # Map from tracer name to CLASS field ('t' or 'e')
    tracer_fields: dict = {'T': 't', 'E': 'e'}
    # Null negative covariance eigenvalues when computing inverse cov?
    null_negative_cov_eigvals_in_icov: bool = False

    def initialize(self):
        self.defaults = dict(self.defaults)
        self._read_data()

    def _get_cl_type(self, tn1, tn2):
        f1 = self.tracer_fields[tn1]
        f2 = self.tracer_fields[tn2]
        code = {'t': '0', 'e': 'e'}
        cltyp = f'cl_{code[f1]}{code[f2]}'
        if cltyp == 'cl_e0':  # sacc only stores cl_0e
            cltyp = 'cl_0e'
        return cltyp

    def _class_key(self, tn1, tn2):
        # CLASS keys: 'tt', 'te', 'ee' (te == et)
        key = self.tracer_fields[tn1] + self.tracer_fields[tn2]
        return 'te' if key == 'et' else key

    def _read_data(self):
        """Read sacc file, apply scale cuts, and build the data vector."""
        self.sacc_file = s = sacc.Sacc.load_fits(self.input_file)

        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            if b['name'] not in self.tracer_fields:
                raise LoggedError(self.log,
                                  "No CLASS field for tracer %s" % b['name'])

        indices = []
        self.cl_meta = []
        id_sofar = 0
        for cl in self.twopoints:
            tn1, tn2 = cl['bins']
            cltyp = self._get_cl_type(tn1, tn2)
            if cltyp == 'cl_0e' and self.tracer_fields[tn1] == 'e':
                tn1, tn2 = tn2, tn1  # sacc stores T first
            l, c_ell, ind = s.get_ell_cl(cltyp, tn1, tn2,
                                         return_cov=False,
                                         return_ind=True)
            if c_ell.size == 0:
                continue

            # Scale cuts
            lmin = cl.get('lmin', self.defaults.get('lmin', 0))
            lmax = cl.get('lmax', self.defaults.get('lmax', 1E30))
            sel = (l >= lmin) * (l <= lmax)
            l = l[sel]
            c_ell = c_ell[sel]
            ind = ind[sel]

            bpw = s.get_bandpower_windows(ind)
            self.cl_meta.append({'bin_1': tn1,
                                 'bin_2': tn2,
                                 'class_key': self._class_key(tn1, tn2),
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size, dtype=int)),
                                 'l_bpw': bpw.values,
                                 'w_bpw': bpw.weight.T})  # (nbin, nell)
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)

        self.data_vec = s.mean[indices]
        self.cov = s.covariance.dense[indices][:, indices]
        self.inv_cov = self.get_inv_cov(self.cov)
        self.ndata = len(self.data_vec)
        self.indices = indices

        # Maximum multipole needed from CLASS, per spectrum type
        self.lmax_class = {}
        for clm in self.cl_meta:
            k = clm['class_key']
            lm = int(np.ceil(np.max(clm['l_bpw'])))
            self.lmax_class[k] = max(self.lmax_class.get(k, 0), lm)

    def get_inv_cov(self, cov):
        if self.null_negative_cov_eigvals_in_icov:
            evals, evecs = np.linalg.eigh(cov)
            inv_evals = 1/evals
            inv_evals[evals < 0] = 0
            return evecs.dot(np.diag(inv_evals).dot(evecs.T))
        return np.linalg.inv(cov)

    def get_requirements(self):
        return {"Cl": dict(self.lmax_class)}

    def get_cl_theory(self):
        """Binned theory data vector."""
        cls = self.provider.get_Cl(ell_factor=False, units="muK2")
        ell = cls['ell']
        t = np.zeros(self.ndata)
        for clm in self.cl_meta:
            cl_th = np.interp(clm['l_bpw'], ell, cls[clm['class_key']],
                              left=0., right=0.)
            t[clm['inds']] = clm['w_bpw'] @ cl_th
        return t

    def logp(self, **params_values):
        r = self.get_cl_theory() - self.data_vec
        chi2 = r @ self.inv_cov @ r
        return -0.5 * chi2

    def get_cl_data_sacc(self):
        s = self.sacc_file.copy()
        s.keep_indices(self.indices)
        return s
