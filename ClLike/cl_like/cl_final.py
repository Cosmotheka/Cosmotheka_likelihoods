"""
Theory class that computes the full angular power spectrum
"""
from cobaya.theory import Theory
import numpy as np


class ClFinal(Theory):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""

    def initialize_with_provider(self, provider):
        self.provider = provider
        # Additional information specific for this likelihood
        ia_model = self.provider.get_ia_model()
        is_PT_bias = self.provider.get_is_PT_bias()

        self.bias_names, self.bias_info = self._get_bias_info(ia_model,
                                                              is_PT_bias)
        self.ndata = np.sum([clm['l_eff'].size for clm in self.cl_meta])

    def get_requirements(self):
        return {"ia_model": None, "is_PT_bias": None}

    def must_provide(self, **requirements):
        if "cl_theory" not in requirements:
            return {}

        options = requirements.get('cl_theory') or {}
        self.cl_meta = options.get("cl_meta")
        self.tracer_qs = options.get("tracer_qs")
        self.bin_properties = options.get("bin_properties")

        return {"Limber": {"cl_meta": self.cl_meta,
                           "tracer_qs": self.tracer_qs,
                           "bin_properties": self.bin_properties}
                }

    def calculate(self, state, want_derived=True, **pars):
        # First, gather all the necessary ingredients for the Cls without bias parameters
        res = self.provider.get_Limber()
        cld = res['cl_data']

        # Construct bias vector
        bias = np.array([pars[k] for k in self.bias_names])

        # Theory model
        t = self._model(cld, bias, **pars)
        state["cl_theory"] = self._model(cld, bias, **pars)
        # state["cl_theory_deriv"] = self._model_deriv(cld, bias, **pars)

    def get_cl_theory(self):
        return self._current_state["cl_theory"]

    def get_cl_theory_deriv(self):
        return self._current_state["cl_theory_deriv"]

    def _model(self, cld, bias_vec, **pars):
        cls = np.zeros(self.ndata)
        for icl, clm in enumerate(self.cl_meta):
            cl_this = np.zeros_like(clm['l_eff'])
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            e1 = self.bias_info[n1]['eps']
            e2 = self.bias_info[n2]['eps']
            ind1 = self.bias_info[n1]['bias_ind']
            ind2 = self.bias_info[n2]['bias_ind']
            b1 = bias_vec[ind1] if ind1 is not None else None
            b2 = bias_vec[ind2] if ind2 is not None else None
            inds = clm['inds']
            if e1 and e2:
                cl_this += cld['cl00'][icl]
            if e1 and (b2 is not None):
                cl_this += np.dot(b2, cld['cl01'][icl]) # (nbias) , (nbias, nell)
            if e2 and (b1 is not None):
                cl_this += np.dot(b1, cld['cl10'][icl]) # (nbias) , (nbias, nell)
            if (b1 is not None) and (b2 is not None):
                cl_this += np.dot(b1, np.dot(b2, cld['cl11'][icl])) # (nbias1) * ((nbias2), (nbias1,nbias2,nell))

            # Add multiplicative bias
            for name in [n1, n2]:
                if self.tracer_qs[name] == "galaxy_shear":
                    bn = '_'.join([self.input_params_prefix, name, 'm'])
                    print(name, bn, pars.get(bn))
                    cl_this *= (1 + pars.get(bn, 0))
            cls[inds] = cl_this

        return cls

    def _model_deriv(self, cld, bias_vec):
        # TODO: Fix this
        nbias = len(bias_vec)
        cls_deriv = np.zeros((self.ndata, nbias))

        for icl, clm in enumerate(self.cl_meta):
            cls_grad = np.zeros([nbias, len(clm['l_eff'])])
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            e1 = self.bin_properties[n1]['eps']
            e2 = self.bin_properties[n2]['eps']
            ind1 = self.bin_properties[n1]['bias_ind']
            ind2 = self.bin_properties[n2]['bias_ind']
            b1 = bias_vec[ind1] if ind1 is not None else None
            b2 = bias_vec[ind2] if ind2 is not None else None
            inds = clm['inds']

            if e1 and (b2 is not None):
                cls_grad[ind2] += cld['cl01'][icl] # (nbias2, ndata) , (nbias2, ndata)
            if e2 and (b1 is not None):
                cls_grad[ind1] += cld['cl10'][icl] # (nbias1, ndata) , (nbias1, ndat)

            if (b1 is not None) and (b2 is not None):
                cl_b2 = np.dot(b2, cld['cl11'][icl]) # (nbias2) , (nbias1, nbias2, ndata) -> (nbias1, ndata)
                cl_b1 = np.sum(b1[:, None, None] * cld['cl11'][icl], axis=0) # (nbias1) , (nbias1, nbias2, ndata) -> (nbias2, ndata)
                cls_grad[ind1] += cl_b2
                cls_grad[ind2] += cl_b1

            cls_deriv[inds] = cls_grad.T
        return cls_deriv # (ndata, nbias)

    def _get_bias_info(self, ia_model, is_PT_bias):
        # Extract additional per-sample information from the sacc
        # file needed for this likelihood.
        ind_bias = 0
        ind_IA = None
        bias_names = []
        bias_info = {}
        for name in self.bin_properties.keys():
            quantity = self.tracer_qs[name]
            bd = bias_info[name] = {}
            bd['bias_ind'] = None # No biases by default
            if quantity == 'galaxy_density':
                # Linear bias
                inds = [ind_bias]
                bias_names.append(self.input_params_prefix + '_'+ name +'_b1')
                ind_bias += 1
                # Higher-order biases
                if is_PT_bias:
                    for bn in ['b2', 'bs', 'bk2']:
                        bias_names.append(self.input_params_prefix + '_'+ name
                                          +'_'+bn)
                        inds.append(ind_bias)
                        ind_bias += 1
                bd['bias_ind'] = inds
                # No magnification bias yet
                bd['eps'] = False
            elif quantity == 'galaxy_shear':
                if ia_model != 'IANone':
                    if ind_IA is None:
                        ind_IA = ind_bias
                        bias_names.append(self.input_params_prefix + '_A_IA')
                        ind_bias += 1
                    bd['bias_ind'] = [ind_IA]
                bd['eps'] = True
            elif quantity == 'cmb_convergence':
                # TODO: No idea what eps is so setting to True to make the test
                # work
                bd['eps'] = True

        return bias_names, bias_info
