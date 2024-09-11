import numpy as np
import pyccl as ccl
import baccoemu
import warnings
import copy
from scipy import optimize
from scipy.misc import derivative  # it is a deprecated function

class BaccoCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
    """
    def __init__(self, log10k_min=np.log10(0.008), log10k_max=np.log10(0.5), nk_per_decade=20,
                 log10k_sh_sh_min=np.log10(0.0001), log10k_sh_sh_max=np.log10(50), nk_sh_sh_per_decade=20,
                 a_arr=None, nonlinear_emu_path=None, nonlinear_emu_details=None, use_baryon_boost=False,
                 ignore_lbias=False, allow_bcm_emu_extrapolation_for_shear=True,
                 allow_halofit_extrapolation_for_shear=False,
                 allow_halofit_extrapolation_for_shear_on_k=False):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        nk_sh_sh_total = int((log10k_sh_sh_max - log10k_sh_sh_min) * nk_sh_sh_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        self.ks_sh_sh = np.logspace(log10k_sh_sh_min, log10k_sh_sh_max, nk_sh_sh_total)
        self.use_baryon_boost = use_baryon_boost
        self.ignore_lbias = ignore_lbias
        self.allow_bcm_emu_extrapolation_for_shear = allow_bcm_emu_extrapolation_for_shear
        self.allow_halofit_extrapolation_for_shear = allow_halofit_extrapolation_for_shear
        self.allow_halofit_extrapolation_for_shear_on_k = allow_halofit_extrapolation_for_shear_on_k

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self.lbias = baccoemu.Lbias_expansion()
            
            # Original code 3/9/2024
            
            #self.mpk = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
            #                                         nonlinear_emu_details=nonlinear_emu_details)
            
            # My modification
            self.mpk = baccoemu.Matter_powerspectrum() # so that all mpk-dependent quantities will only see baccoemu's default limits
            self.mpmatter = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
                                                     nonlinear_emu_details=nonlinear_emu_details) # this is added exclusively for the handling of
                                                     # the halofit/matter power spectrum
            
            
                                                     
                                                     

        # check with the currently loaded version of baccoemu if the a array is
        # all within the allowed ranges
        emu_kind = 'baryon' if self.use_baryon_boost else 'nonlinear'
        amin = self.mpk.emulator[emu_kind]['bounds'][-1][0]
        if a_arr is None:
            zmax = 1/amin - 1
            # Only 20 a's to match the a's in the other PT classes with
            # a < ~0.275
            a_arr = 1./(1+np.linspace(0., zmax, 20)[::-1])
        if np.any(a_arr < amin):
            # This check is done by baccoemu but is not printed by Cobaya, so I
            # add the test here.
            raise ValueError("baccoemu only defined for scale factors between "
                             f"1 and {amin}")
        self.a_s = a_arr

    def _check_baccoemu_baryon_pars_for_extrapolation(self, cosmopars_in):
        """ Check passed parameters, if pars for baryon emu out of range,
        return a new dictionary apt for extrapolation.

        Extrapolation of the bcm emulator in cosmology is done by evaluating
        the emu at the closest cosmology within the allowed parameter space,
        while modifying Ob and Oc to keep the baryon fraction fixed
        """
        cosmopars = copy.deepcopy(cosmopars_in)

        # Return cosmopars to get the sigma8_cold equivalent to the input As
        within_bounds, cosmopars = self._check_within_bounds(cosmopars,
                                                             return_cosmopars=True)
        if (not self.allow_bcm_emu_extrapolation_for_shear) or \
            within_bounds['baryon']:
            return copy.deepcopy(cosmopars), copy.deepcopy(self.a_s)

        cosmopars_out = copy.deepcopy(cosmopars)

        b_frac_orig = cosmopars['omega_baryon']/cosmopars['omega_cold']

        emulator = self.mpk.emulator['baryon']

        for i, par in enumerate(emulator['keys']):
            if (par in self.mpk.emulator['nonlinear']['keys']):
                if par in cosmopars:
                    if cosmopars[par] is None:
                        del cosmopars_out[par]
                    else:
                        if (cosmopars[par] < emulator['bounds'][i][0]):
                            cosmopars_out[par] = emulator['bounds'][i][0]
                        elif (cosmopars[par] > emulator['bounds'][i][1]):
                            cosmopars_out[par] = emulator['bounds'][i][1]

        b_frac = cosmopars_out['omega_baryon']/cosmopars_out['omega_cold']
        if np.round(b_frac_orig, 4) != np.round(b_frac, 4):
            min_func = lambda o: np.abs(o[1] / o[0] - b_frac_orig)
            Oc_bounds = emulator['bounds'][0]
            Ob_bounds = emulator['bounds'][2]
            res = optimize.minimize(min_func,
                                    np.array([cosmopars_out['omega_cold'], cosmopars_out['omega_baryon']]),
                                    bounds=(Oc_bounds, Ob_bounds))
            cosmopars_out['omega_cold'] = res.x[0]
            cosmopars_out['omega_baryon'] = res.x[1]

        a_s_out = copy.deepcopy(self.a_s)
        a_s_out[a_s_out < emulator['bounds'][-1][0]] = emulator['bounds'][-1][0]

        return cosmopars_out, a_s_out

    def _check_within_bounds(self, cosmopars, return_cosmopars=False):
        """
        Check if cosmological parameters are within bounds

        Return: dict with keys 'nonlinear' and 'baryon'. If return_cosmopars is
        True, returns the cosmopars with sigma8_cold instead of A_s.
        """
        cosmopars = copy.deepcopy(cosmopars)
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpk.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']
        within_bounds = []
        within_bounds_mpk = []
        for i, parname in enumerate(self.mpk.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds.append(np.all(val >= self.mpk.emulator['baryon']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['baryon']['bounds'][i][1]))
            within_bounds_mpk.append(np.all(val >= self.mpk.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['nonlinear']['bounds'][i][1]))

        output = {'nonlinear': np.all(within_bounds_mpk),
                'baryon': np.all(within_bounds)}
        if return_cosmopars:
            return output, cosmopars

        return output
    
    def _check_within_bounds_biased_tracers(self, cosmopars, return_cosmopars=False):
        """
        Check if cosmological parameters are within bounds

        Return: dict with keys 'nonlinear' and 'baryon'. If return_cosmopars is
        True, returns the cosmopars with sigma8_cold instead of A_s.
        
        THIS IS FOR BIASED TRACERS AND FOR BARYONS. 
        
        """

        '''

        # Carlos' original code 
        cosmopars = copy.deepcopy(cosmopars)
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpk.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']
        within_bounds = []
        #without_bounds = []
        within_bounds_mpk = []
        for i, parname in enumerate(self.mpk.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds.append(np.all(val >= self.mpk.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['nonlinear']['bounds'][i][1]))	    
            within_bounds_mpk.append(np.all(val >= self.mpk.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['nonlinear']['bounds'][i][1]))

        output = {'nonlinear': np.all(within_bounds_mpk),
                'baryon': np.all(within_bounds)}
        if return_cosmopars:
            return output, cosmopars

        return output
        '''
        
        cosmopars = copy.deepcopy(cosmopars) 
        
        # Do always it for baryons... 
        cosmopars_predict_baryons = copy.deepcopy(cosmopars) # store cosmopars with the parameters at the value I want to predict in case it is necessary
        cosmopars_start_baryons = copy.deepcopy(cosmopars)   # here store the cosmopars by replacing the out-of-bound parameters with the to-start value 
        # in the Taylor expansion
        
        
        # And for matter
        cosmopars_predict_mpk = copy.deepcopy(cosmopars) # store cosmopars with the parameters at the value I want to predict in case it is necessary
        cosmopars_start_mpk = copy.deepcopy(cosmopars)   # here store the cosmopars by replacing the out-of-bound parameters with the to-start value 
        # in the Taylor expansion
          
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpk.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']

        within_bounds_baryons = []  
        within_bounds_mpk = []

        ######################  My addition, add names of parameters which are out of the bounds  #################
        without_bounds_baryons = []
        without_bounds_mpk = []

        # baryon
        to_predict_value_baryons = {}  # here the out-of-bound parameter will be stored
        to_start_value_baryons = {}  # the fictious parameter within baccoemu, acting as a pivot for the Taylor extrapolation, will be stored here
        
        # matter
        to_predict_value_mpk = {} # here the out-of-bound parameter will be stored
        to_start_value_mpk = {} # the fictious parameter within baccoemu, acting as a pivot for the Taylor extrapolation, will be stored here
        
        ####################### End of my addition ###############################################################

    
        for i, parname in enumerate(self.mpk.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
                #print('parameter_value',val)
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds_baryons.append(np.all(val >= self.mpk.emulator['baryon']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['baryon']['bounds'][i][1]))	    
            #print('i',i,'val',val,'', self.mpk.emulator['nonlinear']['bounds'][i][0])
            within_bounds_mpk.append(np.all(val >= self.mpk.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['nonlinear']['bounds'][i][1]))
            
            ####   My addition, store parameters out of bounds, which require extrapolation 

            ## NOTICE the +- 0.02 displacement I am adding on the bounds, this is needed to ensure the derivative calculator does not offset the bounds when calculating the four points derivative, one point of which (either the leftmost or the rightmost, depending whether we are on the lower or upper bound, might be overshooting the bound)
            
            # do it twice, for baryons and nonlinear for consistency, I actually just need nonlinear
            if (within_bounds_baryons[i] == False): #and (parname != 'expfactor')  :
            #print(parname)
                without_bounds_baryons.append(parname)

                to_predict_parameter = val#cosmopars[parname]#within_bounds[i]
                to_predict_parameter_name = parname
        
                if np.all(to_predict_parameter > self.mpk.emulator['baryon']['bounds'][i][1]): #if the parameter is LARGER than the upper bound:
                    to_predict_value_baryons[parname] = to_predict_parameter # store to-predict value
                    to_start_value_baryons[parname] = self.mpk.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                    cosmopars_predict_baryons[parname] = to_predict_value_baryons[parname]
                    cosmopars_start_baryons[parname] = to_start_value_baryons[parname]

                if np.all(to_predict_parameter < self.mpk.emulator['baryon']['bounds'][i][0]): #if the parameter is SMALLER than the lower bound:
                    to_predict_value_baryons[parname] = to_predict_parameter # store to-predict value
                    to_start_value_baryons[parname] = self.mpk.emulator['baryon']['bounds'][i][0] + 0.02 # store to-start value (= lower bound + 0.02 )
                    cosmopars_predict_baryons[parname] = to_predict_value_baryons[parname]
                    cosmopars_start_baryons[parname] = to_start_value_baryons[parname]
            else:  #and (parname != 'expfactor')  :#print(parname)
                
                #to_predict_parameter = cosmopars[parname]#within_bounds[i]
                #to_predict_parameter_name = parname
        
                #if to_predict_parameter > self.mpk.emulator['baryon']['bounds'][i][1]: #if the parameter is LARGER than the upper bound:
                #    to_predict_value[parname] = to_predict_parameter # store to-predict value
                #    to_start_value[parname] = self.mpk.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                
                cosmopars_predict_baryons[parname] = val
                cosmopars_start_baryons[parname] =   val #to_start_value[parname]   

            if (within_bounds_mpk[i] == False): #and (parname != 'expfactor')  :
                #print(parname)
                without_bounds_mpk.append(parname)

                to_predict_parameter = val#cosmopars[parname]#within_bounds_mpk[i]
                to_predict_parameter_name = parname
        
                if np.all(to_predict_parameter > self.mpk.emulator['nonlinear']['bounds'][i][1]): #if the parameter is LARGER than the upper bound:
                    to_predict_value_mpk[parname] = to_predict_parameter # store to-predict value
                    to_start_value_mpk[parname] = self.mpk.emulator['nonlinear']['bounds'][i][1] - 0.02 # store to-start value (= upper bound - 0.02 )
                    cosmopars_predict_mpk[parname] = to_predict_value_mpk[parname]
                    cosmopars_start_mpk[parname] = to_start_value_mpk[parname]

                if np.all(to_predict_parameter < self.mpk.emulator['nonlinear']['bounds'][i][0]): #if the parameter is SMALLER than the lower bound:
                    to_predict_value_mpk[parname] = to_predict_parameter # store to-predict value
                    to_start_value_mpk[parname] = self.mpk.emulator['nonlinear']['bounds'][i][0] + 0.02 # store to-start value (= lower bound + 0.02 )
                    cosmopars_predict_mpk[parname] = to_predict_value_mpk[parname]
                    cosmopars_start_mpk[parname] = to_start_value_mpk[parname]

            else:  #and (parname != 'expfactor')  :#print(parname)
                
                #to_predict_parameter = cosmopars[parname]#within_bounds[i]
                #to_predict_parameter_name = parname
        
                #if to_predict_parameter > self.mpk.emulator['baryon']['bounds'][i][1]: #if the parameter is LARGER than the upper bound:
                #    to_predict_value[parname] = to_predict_parameter # store to-predict value
                #    to_start_value[parname] = self.mpk.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                
                cosmopars_predict_mpk[parname] = val
                cosmopars_start_mpk[parname] =   val #to_start_value[parname]   
    
        ######## end of my addition ############
    
        output = {'nonlinear': np.all(within_bounds_mpk),
                'baryon': np.all(within_bounds_baryons)}
        
        #if return_cosmopars:
        #    return output, cosmopars
            
        ## my addition again
	#print('Sono qui')
        #print('output', output)
        #print('within_bounds',within_bounds)
        #print('within_bounds_mpk',within_bounds_mpk)
        #print('without_bounds',without_bounds)
        #print('without_bounds_mpk',without_bounds_mpk)
        #print('to_predict_value',to_predict_value)
        #print('to_start_value',to_start_value)
        #print('to_predict_value_mpk',to_predict_value_mpk)
        #print('to_start_value_mpk',to_start_value_mpk)
        #print('cosmopars_predict',cosmopars_predict)
        #print('cosmopars_start',cosmopars_start)
        #print('cosmopars_predict_mpk',cosmopars_predict_mpk)
        #print('cosmopars_start_mpk',cosmopars_start_mpk)
        
      
        return output, within_bounds_baryons,within_bounds_mpk,without_bounds_baryons,without_bounds_mpk, to_predict_value_baryons,to_start_value_baryons,to_predict_value_mpk, to_start_value_mpk,cosmopars_predict_baryons,cosmopars_start_baryons,cosmopars_predict_mpk,cosmopars_start_mpk
        
        ### end of my addition
        
        if return_cosmopars:
            return output, within_bounds_baryons,within_bounds_mpk,without_bounds_baryons,without_bounds_mpk, to_predict_value_baryons,to_start_value_baryons,
        to_predict_value_mpk, to_start_value_mpk,cosmopars_predict_baryons,cosmopars_start_baryons,cosmopars_predict_mpk,cosmopars_start_mpk,cosmopars
        

    def _check_within_bounds_matter_power_spectrum(self, cosmopars, return_cosmopars=False):
        
        """
        Check if cosmological parameters are within bounds
        
        ONLY FOR MATTER POWER SPECTRUM; DEPENDING IF THEY ARE WITHIN BOUNDS OR NOT, USE STANDARD NON LINEAR POWER SPECTRUM OR HALOFIT

        Return: dict with keys 'nonlinear' and 'baryon'. If return_cosmopars is
        True, returns the cosmopars with sigma8_cold instead of A_s.
        """

        '''

        # Carlos' original code 
        cosmopars = copy.deepcopy(cosmopars)
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpmatter.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']
        within_bounds = []
        #without_bounds = []
        within_bounds_mpmatter = []
        for i, parname in enumerate(self.mpmatter.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds.append(np.all(val >= self.mpmatter.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpmatter.emulator['nonlinear']['bounds'][i][1]))	    
            within_bounds_mpmatter.append(np.all(val >= self.mpmatter.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpmatter.emulator['nonlinear']['bounds'][i][1]))

        output = {'nonlinear': np.all(within_bounds_mpmatter),
                'baryon': np.all(within_bounds)}
        if return_cosmopars:
            return output, cosmopars

        return output
        '''

        
        cosmopars = copy.deepcopy(cosmopars) 
        
        # Do always it for baryons... 
        cosmopars_predict_baryons_mpmatter = copy.deepcopy(cosmopars) # store cosmopars with the parameters at the value I want to predict in case it is necessary
        cosmopars_start_baryons_mpmatter = copy.deepcopy(cosmopars)   # here store the cosmopars by replacing the out-of-bound parameters with the to-start value 
        # in the Taylor expansion
        
        
        # And for matter
        cosmopars_predict_mpmatter = copy.deepcopy(cosmopars) # store cosmopars with the parameters at the value I want to predict in case it is necessary
        cosmopars_start_mpmatter = copy.deepcopy(cosmopars)   # here store the cosmopars by replacing the out-of-bound parameters with the to-start value 
        # in the Taylor expansion
        
          
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpmatter.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']

        within_bounds_baryons_mpmatter = []  
        within_bounds_mpmatter = []

        ######################  My addition, add names of parameters which are out of the bounds  #################
        without_bounds_baryons_mpmatter = []
        without_bounds_mpmatter = []

        # baryon
        to_predict_value_baryons_mpmatter = {}  # here the out-of-bound parameter will be stored
        to_start_value_baryons_mpmatter = {}  # the fictious parameter within baccoemu, acting as a pivot for the Taylor extrapolation, will be stored here
        
        # matter
        to_predict_value_mpmatter = {} # here the out-of-bound parameter will be stored
        to_start_value_mpmatter = {} # the fictious parameter within baccoemu, acting as a pivot for the Taylor extrapolation, will be stored here
        
        ####################### End of my addition ###############################################################

    
        for i, parname in enumerate(self.mpmatter.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds_baryons_mpmatter.append(np.all(val >= self.mpmatter.emulator['baryon']['bounds'][i][0]) & np.all(val <= self.mpmatter.emulator['baryon']['bounds'][i][1]))	    
            within_bounds_mpmatter.append(np.all(val >= self.mpmatter.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpmatter.emulator['nonlinear']['bounds'][i][1]))
            
            ####   My addition, store parameters out of bounds, which require extrapolation 
            ## NOTICE the +- 0.02 displacement I am adding on the bounds, this is needed to ensure the derivative calculator does not offset the bounds when calculating the four points derivative, one point of which (either the leftmost or the rightmost, depending whether we are on the lower or upper bound, might be overshooting the bound)
             
            # do it twice, for baryons and nonlinear for consistency, I actually just need nonlinear
            if (within_bounds_baryons_mpmatter[i] == False): #and (parname != 'expfactor')  :
            #print(parname)
                without_bounds_baryons_mpmatter.append(parname)

                to_predict_parameter = val#cosmopars[parname]#within_bounds[i]
                to_predict_parameter_name = parname
        
                if np.all(to_predict_parameter > self.mpmatter.emulator['baryon']['bounds'][i][1]): #if the parameter is LARGER than the upper bound:
                    to_predict_value_baryons_mpmatter[parname] = to_predict_parameter # store to-predict value
                    to_start_value_baryons_mpmatter[parname] = self.mpmatter.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                    cosmopars_predict_baryons_mpmatter[parname] = to_predict_value_baryons_mpmatter[parname]
                    cosmopars_start_baryons_mpmatter[parname] = to_start_value_baryons_mpmatter[parname]

                if np.all(to_predict_parameter < self.mpmatter.emulator['baryon']['bounds'][i][0]): #if the parameter is SMALLER than the upper bound:
                    to_predict_value_baryons_mpmatter[parname] = to_predict_parameter # store to-predict value
                    to_start_value_baryons_mpmatter[parname] = self.mpmatter.emulator['baryon']['bounds'][i][0] + 0.02 # store to-start value (= lower bound + 0.02 )
                    cosmopars_predict_baryons_mpmatter[parname] = to_predict_value_baryons_mpmatter[parname]
                    cosmopars_start_baryons_mpmatter[parname] = to_start_value_baryons_mpmatter[parname]
            else:  #and (parname != 'expfactor')  :#print(parname)
                
                #to_predict_parameter = cosmopars[parname]#within_bounds[i]
                #to_predict_parameter_name = parname
        
                #if to_predict_parameter > self.mpmatter.emulator['baryon']['bounds'][i][1]: #if the parameter is LARGER than the upper bound:
                #    to_predict_value[parname] = to_predict_parameter # store to-predict value
                #    to_start_value[parname] = self.mpmatter.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                
                cosmopars_predict_mpmatter[parname] = val
                cosmopars_start_mpmatter[parname] =   val #to_start_value[parname]   

            if (within_bounds_mpmatter[i] == False): #and (parname != 'expfactor')  :
                #print(parname)
                without_bounds_mpmatter.append(parname)

                to_predict_parameter = val#cosmopars[parname]#within_bounds_mpmatter[i]
                to_predict_parameter_name = parname
        
                if np.all(to_predict_parameter > self.mpmatter.emulator['nonlinear']['bounds'][i][1]): #if the parameter is LARGER than the upper bound:
                    to_predict_value_mpmatter[parname] = to_predict_parameter # store to-predict value
                    to_start_value_mpmatter[parname] = self.mpmatter.emulator['nonlinear']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                    cosmopars_predict_mpmatter[parname] = to_predict_value_mpmatter[parname]
                    cosmopars_start_mpmatter[parname] = to_start_value_mpmatter[parname]

                if np.all(to_predict_parameter < self.mpmatter.emulator['nonlinear']['bounds'][i][0]): #if the parameter is SMALLER than the upper bound:
                    to_predict_value_mpmatter[parname] = to_predict_parameter # store to-predict value
                    to_start_value_mpmatter[parname] = self.mpmatter.emulator['nonlinear']['bounds'][i][0] + 0.02 # store to-start value (= lower bound + 0.02 )
                    cosmopars_predict_mpmatter[parname] = to_predict_value_mpmatter[parname]
                    cosmopars_start_mpmatter[parname] = to_start_value_mpmatter[parname]

            else:  #and (parname != 'expfactor')  :#print(parname)
                
                #to_predict_parameter = cosmopars[parname]#within_bounds[i]
                #to_predict_parameter_name = parname
        
                #if to_predict_parameter > self.mpmatter.emulator['baryon']['bounds'][i][1]: #if the parameter is LARGER than the upper bound:
                #    to_predict_value[parname] = to_predict_parameter # store to-predict value
                #    to_start_value[parname] = self.mpmatter.emulator['baryon']['bounds'][i][1] - 0.02 # store to-start value (= upper bound -0.02 )
                
                cosmopars_predict_mpmatter[parname] = val
                cosmopars_start_mpmatter[parname] =   val #to_start_value[parname]   
    
        ######## end of my addition ############
    
        output_mpmatter = {'nonlinear': np.all(within_bounds_mpmatter),
                'baryon': np.all(within_bounds_baryons_mpmatter)}
        
        #if return_cosmopars:
        #    return output, cosmopars
            
        ## my addition again
	#print('Sono qui')
        #print('output', output)
        #print('within_bounds',within_bounds)
        #print('within_bounds_mpmatter',within_bounds_mpmatter)
        #print('without_bounds',without_bounds)
        #print('without_bounds_mpmatter',without_bounds_mpmatter)
        #print('to_predict_value',to_predict_value)
        #print('to_start_value',to_start_value)
        #print('to_predict_value_mpmatter',to_predict_value_mpmatter)
        #print('to_start_value_mpmatter',to_start_value_mpmatter)
        #print('cosmopars_predict',cosmopars_predict)
        #print('cosmopars_start',cosmopars_start)
        #print('cosmopars_predict_mpmatter',cosmopars_predict_mpmatter)
        #print('cosmopars_start_mpmatter',cosmopars_start_mpmatter)
        
      
        return output_mpmatter, within_bounds_baryons_mpmatter,within_bounds_mpmatter,without_bounds_baryons_mpmatter,without_bounds_mpmatter, to_predict_value_baryons_mpmatter,to_start_value_baryons_mpmatter,to_predict_value_mpmatter, to_start_value_mpmatter,cosmopars_predict_baryons_mpmatter,cosmopars_start_baryons_mpmatter,cosmopars_predict_mpmatter,cosmopars_start_mpmatter
        
        ### end of my addition
        
        if return_cosmopars:
            return output_mpmatter, within_bounds_baryons_mpmatter,within_bounds_mpmatter_baryons_mpmatter,without_bounds_baryons_mpmatter,without_bounds_mpmatter, to_predict_value_baryons_mpmatter,to_start_value_baryons_mpmatter,
        to_predict_value_mpmatter, to_start_value_mpmatter,cosmopars_predict_baryons_mpmatter,cosmopars_start_baryons_mpmatter,cosmopars_predict_mpmatter,cosmopars_start_mpmatter,cosmopars
    
    
        

    def _sigma8tot_2_sigma8cold(self, emupars, sigma8tot):
        """Use baccoemu to convert sigma8 total matter to sigma8 cdm+baryons
        """
        if hasattr(emupars['omega_cold'], '__len__'):
            _emupars = {}
            for pname in emupars:
                _emupars[pname] = emupars[pname][0]
        else:
            _emupars = emupars
        A_s_fid = 2.1e-9
        
        
        sigma8tot_fid = self.mpk.get_sigma8(cold=False,
                                            A_s=A_s_fid, **_emupars)
        
                                                    
        A_s = (sigma8tot / sigma8tot_fid)**2 * A_s_fid
        return self.mpk.get_sigma8(cold=True, A_s=A_s, **_emupars)


    def update_pk(self, cosmo, bcmpar=None, **kwargs):
    
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        
        ######################################################################
        #  Calculations are within the EFT framework, what we want are the values of the 15 perturbative terms to get the biased tracer power specturm
        
        #  If the parameter set is within  baccoemu bounds, no extrapolation is performed and just run Carlos' cosmotheka
        #  If the parameter set is without baccoemu bounds, that is it is dtheta away from baccoemu full dynamic range
        	# - The power spectrum is calculated at a baccoemu boundary parameter set theta and extrapolated up to theta+dtheta by means of a first order Taylor expansion in two steps:
        		## P_{alpha,beta}(theta+dtheta,k,z) = r_{alpha,beta}(theta+dtheta,k,z) * P_mm(theta+dtheta,k,z),   		
        		#  with r(k,theta+delta_theta) = r(k,theta) + sum_i [dr^{i}_mm(theta_i)/dtheta_i * (delta_theta)_i] (https://arxiv.org/pdf/2103.09820
        		#, eq. 3.13,3.14)
        
        #  Notice that this influences EXCLUSIVELY NONLINEAR, pnn is always calculated within the standard baccoemu bounds.
        # In other words: 
        	# - One thing is directly calling baccoemu's get_nonlinear_pnn function. This function returns the 15 perturbative terms of the EFT, which allow to build a BIASED TRACER (= galaxies) power spectrum once the 15 terms are combined with the respective bias (user-defined terms) and summed. The output of the summation is equal to emulator.get_galaxy_real_pk(bias=bias_params, k=k, **params) for a given bias_params. This function MUST NOT receive baccoemu extended parameters
        	# The nonlinear MATTER power spectrum, instead, is calculated by means of get_nonlinear_pk and MUST RECEIVE baccoemu extended parameters.
      
        # The function treats differently the P_mm term and the r(k) correction. 
        
        #### P_mm term:
        #	- If all parameters are within (the extended) bounds, P_mm term is baccoemu P non linear (different from perturbative expansion).
        #	- If AT LEAST ONE PARAMETER IS NOT WITHIN BOUNDS (I.E., IF WITHOUT_BOUNDS_MPK is NOT EMPTY ) directly substitute it with halofit, without extrapolating, because it is more trustworthy. P_mm
        
        
        cospar = self._get_bacco_pars_from_cosmo(cosmo)
        h = cospar['hubble']  # this is the Prediction VALUE OF H (or the "True value" at every step), EVEN IF IT IS OVERSHOOTING BACCOEMU LIMITS, USE IT FOR CONVERSION
        cospar_and_a = self._get_pars_and_a_for_bacco(cospar, self.a_s)

        # HEFT
        k_for_bacco = self.ks/h  # this is in h/Mpc^{-1}
        self.k_for_bacco = k_for_bacco
        
        # TODO: Use lbias.emulator['nonlinear']['k'].max() instead of 0.75?
        
        self.mask_ks_for_bacco = np.squeeze(np.where(k_for_bacco <= 0.75))
        k_for_bacco = k_for_bacco[self.mask_ks_for_bacco]
        
        dx = 0.01  # HARD CODED DERIVATIVE SIZE, CONSIDER WHETHER TO MAKE IT ADAPTIVE.
        
        #within_bounds_mpk = self._check_within_bounds(cospar)['nonlinear']
        #within_bounds_mpk = self._my_check_within_bounds(cospar)['nonlinear']
        
        ################# My addition 1) ####################################
        
        # Step 1: check how many parameters are within or without bounds

        #output, within_bounds, within_bounds_mpk, without_bounds, without_bounds_mpk, to_predict_value,to_start_value, to_predict_value_mpk, to_start_value_mpk, cosmopars_predict, cosmopars_start, cosmopars_predict_mpk, cosmopars_start_mpk = self._check_within_bounds_biased_tracers(cospar)  # new output
                
        output, within_bounds_baryons,within_bounds_mpk,without_bounds_baryons,without_bounds_mpk, to_predict_value_baryons,to_start_value_baryons,to_predict_value_mpk, to_start_value_mpk,cosmopars_predict_baryons,cosmopars_start_baryons,cosmopars_predict_mpk,cosmopars_start_mpk  = self._check_within_bounds_biased_tracers(cospar)  # new output
                       
                 
        within_bounds_mpk = output['nonlinear'] # equivalent to previous within_bounds_mpk = self._check_within_bounds(cospar)['nonlinear']
        
        ##### Do the same separatedly for the matter power spectrum 
        
        
        # Note, I am using a temporary variable naming, not the best I can think of, just for symmetry:
        
        # - within_bounds_matter corresponds to within_matter, within_bounds_mpk_matter corresponds to within_bounds_mpk. All quantities with _mpk_matter are the ones of interest so far, 	BECAUSE THE ONES WITHOUT MPK REFER TO BARYONS, WHICH REQUIRE FURTHER DUPLICATION OF THE CODE. 
        
        #output_mpmatter, within_bounds_baryons_mpmatter,within_bounds_mpmatter,without_bounds_baryons_mpmatter,without_bounds_mpmatter, to_predict_value_baryons_mpmatter,to_start_value_baryons_mpmatter,to_predict_value_mpmatter, to_start_value_mpmatter,cosmopars_predict_baryons_mpmatter,cosmopars_start_baryons_mpmatter,cosmopars_predict_mpmatter,cosmopars_start_mpmatter = self._check_within_bounds_matter_power_spectrum(cospar)  # new output
        
        output_mpmatter, within_bounds_baryons_mpmatter,within_bounds_mpk_mpmatter,without_bounds_baryons_mpmatter,without_bounds_mpk_mpmatter, to_predict_value_baryons_mpmatter,to_start_value_baryons_mpmatter,to_predict_value_mpk_mpmatter, to_start_value_mpk_mpmatter,cosmopars_predict_baryons_mpmatter,cosmopars_start_baryons_mpmatter,cosmopars_predict_mpk_mpmatter,cosmopars_start_mpk_mpmatter = self._check_within_bounds_matter_power_spectrum(cospar)  # new output
        
                
        within_bounds_mpk_mpmatter = output_mpmatter['nonlinear'] # equivalent to previous within_bounds_mpk = self._check_within_bounds(cospar)['nonlinear']
        
        
        
        # TEMP RIMUOVIMI: SERVE SOLO PER FAR FUNZIONARE IL CODICE SUCCESSIVAMENTE  
        
        without_bounds_mpk = np.array(['sigma8_cold','omega_cold'])
        to_start_value_mpk['sigma8_cold'] = 0.75
        to_start_value_mpk['omega_cold'] = 0.2 + cospar['omega_baryon'] #+ 0.02
        cosmopars_start_mpk['sigma8_cold'] = to_start_value_mpk['sigma8_cold']
        cosmopars_start_mpk['omega_cold'] = to_start_value_mpk['omega_cold']
        cosmopars_predict_mpk['sigma8_cold'] = cospar['sigma8_cold']   #to_start_value_mpk['sigma8_cold']
        cosmopars_predict_mpk['omega_cold'] =  cospar['omega_cold'] #-0.02  #to_start_value_mpk['omega_cold']
        
        
        #without_bounds_mpk_mpmatter = np.array(['sigma8_cold','omega_cold'])
        #to_start_value_mpk_mpmatter['sigma8_cold'] = 0.75
        #to_start_value_mpk_mpmatter['omega_cold'] = 0.25 + cospar['omega_baryon']
        #cosmopars_start_mpk_mpmatter['sigma8_cold'] = to_start_value_mpk_mpmatter['sigma8_cold']
        #cosmopars_start_mpk_mpmatter['omega_cold'] = to_start_value_mpk_mpmatter['omega_cold']
        #cosmopars_predict_mpk_mpmatter['sigma8_cold'] = cospar['sigma8_cold'] #to_start_value_mpk['sigma8_cold']
        #cosmopars_predict_mpk_mpmatter['omega_cold'] = cospar['omega_cold']  #to_start_value_mpk['omega_cold']
       
        #print('len(without_bounds_mpk)',len(without_bounds_mpk))
      
        if len(without_bounds_mpk) == 0: # if all parameters are within baccoemu narrower bounds, run Carlos' code.
        
        
            if self.ignore_lbias:
                self.pk_temp = None
            else:
                #print('allora sono qui, riga 636')
                self.pk_temp = self.lbias.get_nonlinear_pnn(k=k_for_bacco,
                                                        **cospar_and_a)[1]/h**3
                #print('self.pktemp',self.pk_temp)
                self.k_can_be_useful = k_for_bacco
                
            self.pk2d_computed = {}
        else: # otherwise, handle extrapolation, which involves two different parts 
            
            #print('Sono qui, è falso')
            ########################  1) P_mm part ############################################################
        
            if len(without_bounds_mpk_mpmatter) == 0:  # if there are points beyond the narrower limits but not within the broader ones: 
            
               P_mm = self.mpmatter.get_nonlinear_pk(baryonic_boost=False, cold=False, k=k_for_bacco, **cosmopars_predict_mpk_mpmatter)[1]/h**3 # since the parameters are within the extended bounds, do the calculation with the prediction points, no the starting point ones. 
               
               # that is, P_mm is 1) a pure matter power spectrum, 2) nonlinear, 3) calculated at the prediction point 4) without baryonic boost and with cold = False, 
            
             
            else: # worst-case scenario, if the parameters are overshooting even the extended limits, then use halofit. 
            
               P_mm = np.zeros((len(self.a_s),len(k_for_bacco)))
               # notice, halofit comes from pyccl, hence it receives k in units of Mpc^{-1}, not h/Mpc. Since I want to calculate it at k for bacco, I have to take 
               # the k_for bacco scales and convert them to the equivalent Mpc^{-1} values. H is set at the 
               
               for a_index, a_val in enumerate(self.a_s):
                            P_mm[a_index] = cosmo.nonlin_matter_power(k_for_bacco*h,a_val) # thanks to halofit larger dynamic range, I can use again the true parameters
            
            self.P_mm = P_mm  # for utility, promote to self.
            
            ##################  2) EXTRAPOLATION PART: GET r(k, theta+dtheta), that is, Taylor-expanded biased-tracers correction factor 
            
            print('Extrapolating')
                            
            # 2-1) Get r(k,theta), that is r(k) at the last useful point

            k_at_start = k_for_bacco * h/cosmopars_start_mpk['hubble'] #this is h/Mpc  -> I can't do calculations with baccoemu on biased tracers beyond the parameter limits ,hence k must be within the bounds 
            
            
            k_at_start, pnn_reference_term_by_term = self.lbias.get_nonlinear_pnn(k=k_at_start, **cosmopars_start_mpk.copy())#[1]/h**3
            
            print('cosmopars_start_mpk',cosmopars_start_mpk)
            print('cosmopars_predict_mpk',cosmopars_predict_mpk)
            
            ######### CHECK THIS!    
            
            #self.mpk.get_nonlinear_pnn(k=k_for_bacco, **cosmopars_start_mpk) 
            
            # 15 perturbative terms 
            r_k_reference = np.array(pnn_reference_term_by_term / pnn_reference_term_by_term[0]) # r_k reference is pnn_reference divided by the p_11 term, which corresponds to the matter nonlinear power spectrum when all other bias terms but the '1' operator are set to zero.
        
            self.r_k_reference = r_k_reference
                            
                            
            # Transforming it into an array allows for subsequent matrix manipulation
        
            ################## extrapolation process   ####################################
         
            #  - Calculate partial derivatives and perturbative terms 
        
        
            # - ## Maybe this can be further improved with use of gradient for the first order expansion
            #if true: 
            total_perturbation_rk = np.zeros(len(k_for_bacco))
            
            drk_dvariable_dictionary = {} # storing all derivatives
            
            rk_perturbation_term_dictionary = {}  # storing derivatives*delta_theta for all parameters
            
            # for now, just focus on the pk of interest, do not duplicate for baryons:
            
            for par_name in without_bounds_mpk:  
                
                drk_dvariable_dictionary['drk_d' + par_name]  = self.central_derivator(self._get_drk_dparameter,to_start_value_mpk[par_name],dx=dx,args=(k_at_start,cosmopars_start_mpk.copy(),par_name))#[1]
                             # array of length k with the power spectrum derivative with respect to the desired parameter
                rk_perturbation_term_dictionary[par_name] = drk_dvariable_dictionary['drk_d' + par_name] * ( cosmopars_predict_mpk.copy()[par_name] - cosmopars_start_mpk.copy()[par_name] ) # calculate dp/dpar * delta_par 
                
                print('deltah',( cosmopars_predict_mpk.copy()[par_name] - cosmopars_start_mpk.copy()[par_name] ))
                total_perturbation_rk = total_perturbation_rk + rk_perturbation_term_dictionary[par_name] # add all the perturbations together, at first order it is linear	    
            
            self.drk_dvariable_dictionary = drk_dvariable_dictionary
            self.rk_perturbation_term_dictionary = rk_perturbation_term_dictionary
        
            r_k_predicted = r_k_reference + total_perturbation_rk # store predicted Taylor-expanded BIASED TERMS EFT EXPANSION,
            # A MATRIX OF SIZE 15 ROWS x len(k) columns, the total r_k has to be calculated separatedly
        
            self.r_k_predicted = r_k_predicted
            self.total_perturbation_rk = total_perturbation_rk
            # Get all corrected terms (matrix again)
        
            P_corrected_predicted = self.r_k_predicted * self.P_mm # a matrix of 15 rows (nr. of EFT terms) each of               n_columns = len(k), calculated at scale factor a_s and with parameters theta+dtheta obtained via Taylor expansion
            #
                           
        
            self.P_corrected_predicted = P_corrected_predicted
            
            if self.ignore_lbias:
                self.pk_temp = None
            
            else: 
                        self.pk_temp = self.P_corrected_predicted
                        #print('sono qui')
                        #print(self.pk_temp)
                        #self.k_for_bacco = k_for_bacco
                        self.k_at_start = k_at_start
         
            self.pk2d_computed = {}
                             
            ###########################à######  end of my addition 1  ##############################
            ############## The code below does not involve my case, it might not be necessary at the moment and could be silenced ###
        
        
        '''

        if self.ignore_lbias:
            self.pk_temp = None
        else:
            self.pk_temp = self.lbias.get_nonlinear_pnn(k=k_for_bacco,
                                                        **cospar_and_a)[1]/h**3

        # Shear - Shear (and baryons)
        
        
        baryonic_boost = self.use_baryon_boost and (bcmpar is not None)

        k_sh_sh_for_bacco = self.ks_sh_sh/h
        emu_type_for_setting_kmax = 'baryon' if baryonic_boost else 'nonlinear'
        self.mask_ks_sh_sh_for_bacco = np.squeeze(np.where(k_sh_sh_for_bacco <= self.mpk.emulator[emu_type_for_setting_kmax]['k'].max()))
        k_sh_sh_for_bacco = k_sh_sh_for_bacco[self.mask_ks_sh_sh_for_bacco]

        within_bounds_mpk = self._check_within_bounds(cospar)['nonlinear']

        if (not within_bounds_mpk) & self.allow_halofit_extrapolation_for_shear:
            cosmo.compute_nonlin_power()
            pknl = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            pk = np.array([pknl.eval(self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco], a, cosmo) for a in self.a_s])
        else:
            # TODO: This is going to be called even if no baryons are
            # requested. Shouldn't it have a flag?
            pk = self.mpk.get_nonlinear_pk(baryonic_boost=False, cold=False,
                                           k=k_sh_sh_for_bacco,
                                           **cospar_and_a)[1]/h**3

        if baryonic_boost:
            Sk = self.get_baryonic_boost(cosmo, bcmpar, k_sh_sh_for_bacco)
        else:
            Sk = np.ones_like(pk)

        if self.allow_halofit_extrapolation_for_shear_on_k:
            cosmo.compute_nonlin_power()
            pknl = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            kix = self.mask_ks_sh_sh_for_bacco[-1] + 1
            pkhfit = [pknl(self.ks_sh_sh[kix:], a) for a in self.a_s]
            pk = np.concatenate([pk, pkhfit], axis=1)
            # Extrapolating as in CCL. We could come up with different
            # extrapolation schemes (e.g. Sk = constant?)
            Sk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]),
                            pk_arr=np.log(Sk), is_logp=True)
            Sk = np.array([Sk2d(self.ks_sh_sh, ai) for ai in self.a_s])

        self.pk_temp_sh_sh = pk * Sk
        self.Sk_temp = Sk
        self.pk2d_computed = {}
        
        '''
        
    def _get_dpk_dparameter(self,param_evaluation_point,k,params_dict,param_sensitivity_name):
   
                       
        #Arguments: 
   	
   	#- param_evaluation_point: the function is being passed to the derivative calculator, which provides the evalutation at param_evaluation_point (i.e., the point called theta_star, theta_reference, depending on notation)
   	
   	#- k: array of Fourier modes
   	
   	#- params_dict: a COPY of the full cosmology parameters at the REFERENCE cosmology point, only the parameter whose name is in param_sensitivity_name is getting modified
   	#- param_sensitivity_name: the parameter name with respect to which the derivative is taken. 
   	
   	#'''
        params_dict[param_sensitivity_name] = param_evaluation_point  #S_8 * ((S_8_sensitivity_params['omega_cold'])/0.3)**(-alpha)
        
        k, dpk_dparameter_nonlinear  = self.mpk.get_nonlinear_pk(baryonic_boost=False, cold=False,k=k, **params_dict)
        return dpk_dparameter_nonlinear/params_dict['hubble']**3
    
    def _get_drk_dparameter(self,param_evaluation_point,k,params_dict,param_sensitivity_name):

        #'''
        #Arguments: 
   	
   	    #- param_evaluation_point: the function is being passed to the derivative calculator, which provides the evalutation at 		param_evaluation_point (theta_star, theta_reference, depending on notation)
   	
   	    #- k: array of Fourier modes
   	
   	    #- params_dict: the full cosmology parameters at the REFERENCE cosmology point, only the parameter whose name is in param_sensitivity_name
   	    #- param_sensitivity_name: the parameter name with respect to which the derivative is taken .
        
        params_dict[param_sensitivity_name] = param_evaluation_point  #S_8 * ((S_8_sensitivity_params['omega_cold'])/0.3)**(-alpha)
        k, dpk_dparameter_biased  =  self.lbias.get_nonlinear_pnn(k=k, **params_dict)  #[1]/h**3#emulator.get_nonlinear_pnn(k=k, **params_dict) 
        return np.array(dpk_dparameter_biased / dpk_dparameter_biased[0])  #MIND THE RATIO and check the index! 
        

    def _get_pars_and_a_for_bacco(self, pars, a):
        combined_pars = {}
        for key in pars.keys():
            combined_pars[key] = np.full((len(a)), pars[key])
        combined_pars['expfactor'] = a

        return combined_pars

    def _get_bacco_pars_from_cosmo(self, cosmo):
        cospar = {
            'omega_cold': cosmo['Omega_c'] + cosmo['Omega_b'],
            'omega_baryon': cosmo['Omega_b'],
            'ns': cosmo['n_s'],
            'hubble': cosmo['h'],
            'neutrino_mass': np.sum(cosmo['m_nu']),
            'w0': cosmo['w0'],
            'wa': cosmo['wa']}
        if np.isnan(cosmo['A_s']):
            cospar['sigma8_cold'] = self._sigma8tot_2_sigma8cold(cospar, cosmo.sigma8())
        
            #try: 
            #            cospar['sigma8_cold'] = self._sigma8tot_2_sigma8cold(cospar, cosmo.sigma8())
            #except:
            #            cospar['sigma8_cold'] = self._sigma8tot_2_sigma8cold(cospar,cosmo.sigma8())
                        
        else:
            cospar['A_s'] = cosmo['A_s']

        return cospar

    def get_baryonic_boost(self, cosmo, bcmpar, k_arr):
        cospar = self._get_bacco_pars_from_cosmo(cosmo)
        cospar_for_bcm, these_a_s = self._check_baccoemu_baryon_pars_for_extrapolation(cospar)
        cospar_for_bcm.update(bcmpar)
        cospar_for_bcm = self._get_pars_and_a_for_bacco(cospar_for_bcm,
                                                        these_a_s)
        Sk = self.mpk.get_baryonic_boost(k=k_arr, **cospar_for_bcm)[1]
        return Sk
    
    def central_derivator(self, f_x,x_0,args=(), dx=0.01,) :
    
        '''
        Scipy's derivator is deprecated, this is my personal version, central difference with o(dx^2) precision
        '''
    
        right_edge = f_x(x_0+dx,*args)
        left_edge = f_x(x_0-dx,*args)
        num = right_edge - left_edge
        den = 2*dx
        return num/den
    
    
    def get_pk(self, kind, pnl=None, cosmo=None, sub_lowk=False, alt=None):
        # Clarification:
        # We are expanding the galaxy overdensity as:
        #   1+ d_g = 1 + b1 d + b2 d2^2/2 + bs s^2/2 + bk k^2 d
        # But Bacco assumes
        #   1+d_g = 1 + b1 d + b2 d2^2 + bs s^2 + bk k^2 d
        # The order of pk_Temp:
        #  11, 1d, 1d2, 1s2, 1k2, dd, dd2, ds2, dk2, d2d2, d2s2, d2k2, s2s2, s2k2, k2k2
        # This will return
        # mm -> <1*1> (from bacco)
        # md1 -> <1*d> (from bacco)
        # md2 -> <1*d^2/2>
        # ms2 -> <1*s^2/2>
        # mk2 -> <k2*k2> (with <d*d> as pnl)
        # d1d1 -> <d*d> (returns pnl)
        # d1d2 -> <d*d^2/2>
        # d1s2 -> <d*s^2/2>
        # d1k2 -> k^2 <d*d> (with <d*d> as pnl)
        # d2d2 -> <d^2/2*d^2/2>
        # d2s2 -> <d^2/2*s^2/2>
        # d2k2 -> k^2 <d*d^2/2>, not provided
        # s2s2 -> <s^2/2*s^2/2>
        # s2k2 -> k^2 <d*s^2/2>, not provided
        # k2k2 -> k^4 <d*d>, not provided
        # When not provided, this function just returns `alt`

        if kind in self.pk2d_computed:
            return self.pk2d_computed[kind]

        inds = {'mm': 0,
                'md1': 1,
                'md2': 2,
                'ms2': 3,
                'mk2': 4,
                'd1d1': 5,
                'd1d2': 6,
                'd1s2': 7,
                'd1k2': 8,
                'd2d2': 9,
                'd2s2': 10,
                'd2k2': 11,
                's2s2': 12,
                's2k2': 13,
                'k2k2': 14}
        pfac = {'mm': 1.0,
                'md1': 1.0,
                'md2': 0.5,
                'ms2': 0.5,
                'mk2': 1.0,
                'd1d1': 1.0,
                'd1d2': 0.5,
                'd1s2': 0.5,
                'd1k2': 1.0,
                'd2d2': 0.25,
                'd2s2': 0.25,
                'd2k2': 0.5,
                's2s2': 0.25,
                's2k2': 0.5,
                'k2k2': 1.0}

        if kind == 'Sk':
            pk = np.log(self.Sk_temp)
            if self.allow_halofit_extrapolation_for_shear_on_k:
                k = self.ks_sh_sh
            else:
                k = self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=pk,
                            is_logp=True)
            self.pk2d_computed[kind] = pk2d
        elif kind == 'mm_sh_sh':
            if self.allow_halofit_extrapolation_for_shear_on_k:
                k = self.ks_sh_sh
            else:
                k = self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]
            pk = np.log(self.pk_temp_sh_sh)
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=pk,
                            is_logp=True)
            self.pk2d_computed[kind] = pk2d
        else:
            if not self.ignore_lbias:
                pk = pfac[kind]*self.pk_temp[inds[kind], :, :]
                if kind in ['mm']:
                    pk = np.log(pk)
                pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks[self.mask_ks_for_bacco]),
                                pk_arr=pk, is_logp=kind in ['mm'])
                self.pk2d_computed[kind] = pk2d
            else:
                pk2d = None

        return pk2d
