"""
Theory class that computes the full angular power spectrum
"""
from cobaya.theory import Theory
from cobaya.log import LoggedError
import numpy as np


class ClFinal(Theory):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
    shape_model: str = "ShapeNone"

    def initialize_with_provider(self, provider):
        self.provider = provider
        # Additional information specific for this likelihood
        ia_model = self.provider.get_ia_model()
        bias_model = self.provider.get_bias_model()
        is_PT_bias = self.provider.get_is_PT_bias()

        self.bias_names, self.bias_info = self._get_bias_info(ia_model,
                                                              bias_model,
                                                              is_PT_bias)
        self.ndata = np.sum([clm['l_eff'].size for clm in self.cl_meta])

    def get_requirements(self):
        return {"ia_model": None, "bias_model": None, "is_PT_bias": None}

    def must_provide(self, **requirements):
        self.calculate_things = ("cl_theory" in requirements)
        if 'cl_with_bias' not in requirements and self.calculate_things is False:
            return {}

        if self.calculate_things:
            options = requirements.get('cl_theory') or {}
        else:
            options = requirements.get('cl_with_bias') or {}

        self.cl_meta = options.get("cl_meta")
        self.tracer_qs = options.get("tracer_qs")
        self.bin_properties = options.get("bin_properties")

        return {"Limber": {"cl_meta": self.cl_meta,
                           "tracer_qs": self.tracer_qs,
                           "bin_properties": self.bin_properties}
                }

    def calculate(self, state, want_derived=True, **pars):
        if not self.calculate_things:
            return

        # Construct bias vector
        bias = np.zeros(len(self.bias_names))
        for i, k in enumerate(self.bias_names):
            if k[-2:] == "_s":
                # Magnification i.e. (2 - 5s)
                bias[i] = 2 - 5*pars[k]
            else:
                bias[i] = pars[k]

        # Construct global bias vector
        global_bias = self._get_global_bias(**pars)

        # Theory model
        state["cl_theory"], state["cl_theory_deriv"] = self.get_cl_with_bias(bias, global_bias)

    def get_cl_with_bias(self, bias, global_bias):
        # First, gather all the necessary ingredients for the Cls without bias parameters
        res = self.provider.get_Limber()
        cld = res['cl_data']
        # Theory model
        cl = self._model(cld, bias, global_bias)
        dcl = self._model_deriv(cld, bias, global_bias)
        return cl, dcl

    def get_bias_info(self):
        return self.bias_names, self.bias_info

    def get_cl_theory(self):
        return self._current_state["cl_theory"]

    def get_cl_theory_deriv(self):
        return self._current_state["cl_theory_deriv"]

    def get_cl_theory_dderiv(self, bias, global_bias):
        # First, gather all the necessary ingredients for the Cls without bias parameters
        res = self.provider.get_Limber()
        cld = res['cl_data']
        return self._model_dderiv(cld, bias, global_bias)

    def _get_global_bias(self, **pars):
        global_bias = {}
        for name in self.bin_properties.keys():
            global_bias[name] = 1

            # Add multiplicative bias
            if self.tracer_qs[name] == "galaxy_shear":
                if self.shape_model == 'ShapeMultiplicative':
                    bn = '_'.join([self.input_params_prefix, name, 'm'])
                    global_bias[name] = (1 + pars.get(bn, 0))

        return global_bias

    def _model(self, cld, bias_vec, global_bias):
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

            # Multiply global biases (e.g. multiplicative bias)
            cl_this *= global_bias[n1]
            cl_this *= global_bias[n2]

            cls[inds] = cl_this

        return cls

    def _model_deriv(self, cld, bias_vec, global_bias):
        nbias = len(bias_vec)
        cls_deriv = np.zeros((self.ndata, nbias))

        for icl, clm in enumerate(self.cl_meta):
            cls_grad = np.zeros([nbias, len(clm['l_eff'])])
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            e1 = self.bias_info[n1]['eps']
            e2 = self.bias_info[n2]['eps']
            ind1 = self.bias_info[n1]['bias_ind']
            ind2 = self.bias_info[n2]['bias_ind']
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

            # Multiply global biases (e.g. multiplicative bias)
            cls_grad[ind1] *= global_bias[n1]
            cls_grad[ind2] *= global_bias[n2]

            cls_deriv[inds] = cls_grad.T
        return cls_deriv # (ndata, nbias)

    def _model_dderiv(self, cld, bias_vec, global_bias):
        nbias = len(bias_vec)
        cls_dderiv = np.zeros((self.ndata, nbias, nbias))

        for icl, clm in enumerate(self.cl_meta):
            n1 = clm['bin_1']
            n2 = clm['bin_2']
            ind1 = self.bias_info[n1]['bias_ind']
            ind2 = self.bias_info[n2]['bias_ind']
            inds = clm['inds']

            if (ind1 is not None) and (ind2 is not None):
                cls_hess = np.zeros([nbias, nbias, len(clm['l_eff'])])
                cls_hess[np.ix_(ind1, ind2)] += cld['cl11'][icl]
                cls_hess[np.ix_(ind2, ind1)] += np.transpose(cld['cl11'][icl], axes=(1, 0, 2))
                cls_dderiv[inds] = np.transpose(cls_hess, axes=(2, 0, 1))
        return cls_dderiv # (ndata, nbias)


    def _get_bias_info(self, ia_model, bias_model, is_PT_bias):
        # Extract additional per-sample information from the sacc
        # file needed for this likelihood.
        ind_bias = 0
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
                # Magnification bias
                if self.bin_properties[name]['mag_bias']:
                    pn = self.input_params_prefix + '_'+ name +'_s'
                    bias_names.append(pn)
                    inds.append(ind_bias)
                    ind_bias += 1
                bd['bias_ind'] = inds

                # In the lagrangian picture there's an unbiased term.
                bd['eps'] = (bias_model in ['LagrangianPT', 'BaccoPT'])
                # No magnification bias yet

            elif quantity == 'galaxy_shear':
                bd['eps'] = True
                if ia_model == 'IAPerBin':
                    pn = '_'.join([self.input_params_prefix, name, 'A_IA'])
                elif ia_model == 'IADESY1':
                    pn = '_'.join([self.input_params_prefix, 'A_IA'])
                elif ia_model == 'IADESY1_PerSurvey':
                    # This assumes that name = survey__zbin
                    survey = name.split('__')[0]
                    pn = '_'.join([self.input_params_prefix, survey, 'A_IA'])
                else:
                    continue

                if pn in bias_names:
                    bd['bias_ind'] = [bias_names.index(pn)]
                else:
                    bias_names.append(pn)
                    bd['bias_ind'] = [ind_bias]
                    ind_bias += 1
            elif quantity == 'cmb_convergence':
                bd['eps'] = True

        return bias_names, bias_info
