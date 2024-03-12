import numpy as np
import pickle
import sklearn.decomposition
import os
import tensorflow as tf


class PCA(sklearn.decomposition.PCA):
    @property
    def normalizer(self):
        return self._normalizer 

    def output(self, X):
        ''' Decompress data into higher dimensional space. 
        Parameters
        ----------
        X : 1d array
        '''
            
        X =  self.inverse_transform(X[np.newaxis,:]).flatten()

        if (self.normalizer is not None):
            
            X = np.array(self.normalizer.input(X))

        return X
    
class Znormalize:

    @property
    def means(self):
        return self._means   
    
    @means.setter
    def means(self, mu):
        self._means = mu

    @property
    def stds(self):
        return self._stds

    @stds.setter
    def stds(self, sigma):
        self._stds = sigma

    def input(self, X):
        '''Z-normalize data
        '''

        X_normalize = (X - self.means)/self.stds

        return X_normalize

    def output(self, X):
        '''Un-normalize data
        '''

        X_unnormalize = (X *self.stds) + self.means

        return X_unnormalize

    

    pass

class MAREN_standalone:

    @property
    def load_message(self):
        return self._load_message

    @property
    def savedir(self):
        return self._savedir

    @property
    def nuclide_normalizer(self):
        '''
        Z-normalizes for the following nucldie identities
            nuclides = [
                    'u-234',
                    'u-235',
                    'u-236',
                    'u-238',
                    'pu-238',
                    'pu-239',
                    'pu-240',
                    'pu-241',
                    'pu-242',
                ]
        '''
        return self._nuclide_normalizer

    @property
    def isp_normalizer(self):
        '''Z normalize for the instantaneous state parameters
            Tf, Tm, Rhom, B
        '''
        return self._isp_normalizer

    @property
    def nuclide_input_min(self):
        return self._nuclide_input_min
    
    @property
    def nuclide_input_max(self):
        return self._nuclide_input_max
    
    @property
    def isp_input_min(self):
        return self._isp_input_min
    
    @property
    def isp_input_max(self):
        return self._isp_input_max       


    @property
    def fuel(self):
        return self._fuel

    @property
    def bh2o(self):
        return self._bh2o
    
    @property
    def air(self):
        return self._air

    @property
    def ss(self):
        return self._ss

    @property
    def b4c(self):
        return self._b4c

    @property   
    def he(self):
        return self._he
    
    @property
    def zirc4(self):
        return self._zirc4

    @property
    def flux(self):
        return self._flux

    @classmethod
    def load_from_dir(cls, dir,) -> 'MAREN_standalone':
        
        with open(f'{dir}/maren_standalone.pkl','rb') as f:
            maren_standalone = pickle.load(f)
        print(f'\tLoaded MAREN standalone from directory: {dir}')

        print(f'\nLoad Message:\n{maren_standalone.load_message}\n\n')
        
        return maren_standalone

    @staticmethod
    def nuclide_input_list():
        '''U and Pu isotopes used as input parameters for the various cross-section types
        '''
        result= [
        'U234',
        'U235',
        'U236',
        'U238',
        'Pu238',
        'Pu239',
        'Pu240',
        'Pu241',
        'Pu242',
        ]
        return result

    @staticmethod
    def fuelpellet_reconstruction_nuclide_list():
        
        '''Returns a list of nuclide id's used for fuel pellet reconstruction. 
        '''

        UPuTh = [
            'U232',
            'U233',
            'U234',
            'U235',
            'U236',
            'U237',
            'U238',
            'U239',
            'Pu238',
            'Pu239',
            'Pu240',
            'Pu241',
            'Pu242',
            'Th232',
            'Th233',
            'Th234',
            'Pa231',
            'Pa232',
            'Pa233',
        ]

        MA = [
            'Am241',
            'Am242',
            'Am242_m1',
            'Am243',
            'Cm242',
            'Cm243',
            'Cm244',
            'Cm245',
            'Cm246',
            'Cm247',
            'Cm248',
            'Np237',
            'Np238',
            'Np239',
        ]

        # ignore burnable poisons for now
        BP = [
            'Gd152',
            'Gd154',
            'Gd155',
            'Gd156',
            'Gd157',
            'Gd158',
            'Gd160',
        ]

        FP = [
            'Xe135',
            'Sm149',
            'Nd143',
            'Rh103',
            'Pm147',
            'Xe131',
            'Cs133',
            'Tc99',
            'Sm152',
            'Sm151',
            'Nd145',
            'Pm148_m1',
            'Mo95',
            'Eu153',
            'Ru101',
            'Sm150',
            'Pm148',
            'Eu154',
            'Cs134',
            'Eu155',
            'Pr141',
            'Mo98',
            'Kr83',
            'La139',
            'Zr93',
            'Sm147',
            'Rh105',
            'Ag109',
            'Cs135',
            'Pd105',
            'Mo97',
            'I129',
            'Cd113',
            'Xe133',
            'Nd144',
            'Pd107',
            'Zr91',
            'Nd148',
            'Pm149',
            'Eu151',
            'Sm148',
            'I135',
            'Ru105',
            'Ba140',
            'La140',
            'Nd147',
        ]

        # Light nuclides (O-16, N-14, N-15)
        LN = [
            'H1',
            'H2',
            'O16',
            'N14',
            'N15',
        ]

        nuc = UPuTh + MA + BP + FP + LN
        return nuc

    @staticmethod
    def scale_56group_library():
        '''Energy group edges of the SCALE 56 group library (SCALE user manual)
        '''
        return np.array([
            2.00000E+07,
            6.43400E+06,
            4.30400E+06,
            3.00000E+06,
            1.85000E+06,
            1.50000E+06,
            1.20000E+06,
            8.61100E+05,
            7.50000E+05,
            6.00000E+05,
            4.70000E+05,
            3.30000E+05,
            2.70000E+05,
            2.00000E+05,
            5.00000E+04,
            2.00000E+04,
            1.70000E+04,
            3.74000E+03,
            2.25000E+03,
            1.91500E+02,
            1.87700E+02,
            1.17500E+02,
            1.16000E+02,
            1.05000E+02,
            1.01200E+02,
            6.75000E+01,
            6.50000E+01,
            3.71300E+01,
            3.60000E+01,
            2.17500E+01,
            2.12000E+01,
            2.05000E+01,
            7.00000E+00,
            6.87500E+00,
            6.50000E+00,
            6.25000E+00,
            5.00000E+00,
            1.13000E+00,
            1.08000E+00,
            1.01000E+00,
            6.25000E-01,
            4.50000E-01,
            3.75000E-01,
            3.50000E-01,
            3.25000E-01,
            2.50000E-01,
            2.00000E-01,
            1.50000E-01,
            1.00000E-01,
            8.00000E-02,
            6.00000E-02,
            5.00000E-02,
            4.00000E-02,
            2.53000E-02,
            1.00000E-02,
            4.00000E-03,
            1.00000E-05,
        ])

    def get_nuclide_inputs(self, nuclide_concentrations):
        nuc_inp_ind = [MAREN_standalone.fuelpellet_reconstruction_nuclide_list().index(t) for t in MAREN_standalone.nuclide_input_list() ]

        NOI_input = nuclide_concentrations[nuc_inp_ind]

        return NOI_input

    def normalize_nuclide_inputs(self, nuclide_input):
        # clamp nuclide inputs between min and max values
        for n in range(len(nuclide_input)):
            nuclide_input[n] = np.clip(nuclide_input[n], self.nuclide_input_min[n], self.nuclide_input_max[n])

        nuclide_input = self.nuclide_normalizer.input(nuclide_input)

        return nuclide_input

    def normalize_isp_inputs(self, isp):
        # clamp nuclide inputs between min and max values
        for n in range(len(isp)):
            isp[n] = np.clip(isp[n], self.isp_input_min[n], self.isp_input_max[n])

        isp = self.isp_normalizer.input(isp)

        return isp       

    def predict_multigroup_xslibraries(self, fuelpellet_nuclide_concentrations, instantaneous_state_parameters, bh2o_nuclide_concentrations) -> dict:
        '''Evaluates the cross-section libraries for the following materials in the fuel pin, control rod and instrumentation tube geometry model. Terms in brackets are the dictionary keys: 
        fuel pin geometry model
        fuel pellet ("fuel"), he gap ("he"), zirc4 clad ("zirc4"), borated water ("bh2o"), 
        instrumentation tube: stainless steel ("ss"), air ("air")
        control rod: boron-carbide ("b4c")

        Parameters
        ----------

        fuelpellet_nuclide_concentrations : np.ndarray of shape (91,) 
            Fuel pellet nuclide concentrations in atom/b-cm. See MAREN_standalone.fuelpellet_reconstruction_nuclide_list() for ordering of nuclides


        instantaneous_state_parameters : np.ndarray of shape (4,). 
            In the order: Tf (fuel temperature [K]), Tm (moderator temperature[K]) Rhom (moderator density [g/cm3]), B (boron concentration [ppm])

        
        bh2o_nuclide_concentrations : np.ndarray of shape (6,)
            Nuclide concentration in borated water, see BH2O.bh2o_nuclides() for the ordering of nuclides

        Returns
        -------
        mat_xs_library : dict with the following keys: fuel, he, zirc4, bh2o
            'fuel' has the following macroscopic cross-section types: fission, total, nu, chi, p0scatter, all other materials only have total and p0scatter cross-section types. 

        '''

        mat_xs_library = {}

        mat_xs_library['fuel'] = self.fuel.output(fuelpellet_nuclide_concentrations, instantaneous_state_parameters,)

        mat_xs_library['bh2o'] = self.bh2o.output(fuelpellet_nuclide_concentrations, bh2o_nuclide_concentrations, instantaneous_state_parameters)

        for mat in ['he','zirc4','air','ss','b4c']:
            if getattr(self, mat).use_Tf == True:
                isp = instantaneous_state_parameters
            else:
                isp = instantaneous_state_parameters[1:]

            mat_xs_library[mat] = getattr(self, mat).output(isp)

        return mat_xs_library

    def predict_fewgroup_xslibraries(self, fuelpellet_nuclide_concentrations, instantaneous_state_parameters, bh2o_nuclide_concentrations, fewgroup_bin_edge_index=[40]) -> dict:
        '''Similar to predict_multigroup_fuelpin_geometry_xslibraries() method, but use fuel flux to collapse fuel pellet cross-section libraries and use bh2o flux to collapse the rest

        fewgroup_bin_edge_index : list of integers
            group edge index demarcating few-group. i.e. fewgroup_bin_edge_index=[40] groups 0-39 into few group index 0 and groups 40-56 into few group index 1. 

        '''

        fuel_flux, bh2o_flux = self.flux.output(fuelpellet_nuclide_concentrations, instantaneous_state_parameters)

        mgxs = self.predict_multigroup_xslibraries(fuelpellet_nuclide_concentrations, instantaneous_state_parameters, bh2o_nuclide_concentrations)

        fgxs = {}
        for mat in ['fuel','he','zirc4','bh2o','air','ss','b4c']:
            fgxs[mat] = {}
            if mat == 'fuel':
                weighting_flux = fuel_flux
            else:
                weighting_flux = bh2o_flux

            for xs in mgxs[mat].keys():
                if xs != 'chi':
                    fgxs[mat][xs] = self.collapse_multigroup_to_fewgroup(mgxs[mat][xs], weighting_flux, fewgroup_bin_edge_index)
                else:
                    fgxs[mat][xs] = self.collapse_chi_to_fewgroup(mgxs[mat][xs], fewgroup_bin_edge_index)

        return fgxs, fuel_flux, bh2o_flux

    def collapse_multigroup_to_fewgroup(self, xs:np.ndarray, weighting_flux:np.ndarray, fewgroup_bin_edge_index=[40]):
        '''Collapse multigroup libraries to fewgroup libraries using weighting flux 

        Parameters
        ----------
        xs : np.ndarray
            Must be of shape (56,) or (56,56)

        weighting_flux : np.ndarray
            Shape (56,)

        fewgroup_bin_edge_index : list
            group edge index demarcating few-group. i.e. fewgroup_bin_edge_index=[40] groups 0-39 into few group index 0 and groups 40-56 into few group index 1. 

        Returns
        -------
        fg_xs : np.ndarray of shape (len(fewgroup_bin_edge_index)+1)

        '''

        if len(xs.shape) == 1:
            assert xs.shape[0] == 56
        else:
            assert xs.shape == (56, 56)

        
        fewgroup_bin_edge_index = np.array(fewgroup_bin_edge_index, dtype=int)
        fewgroup_bin_edge_index = np.insert(fewgroup_bin_edge_index, 0, 0)
        fewgroup_bin_edge_index = np.append(fewgroup_bin_edge_index, len(weighting_flux))

        Ngroups = len( fewgroup_bin_edge_index)-1
        
        if len(xs.shape) == 1:
            fgXS = np.zeros(Ngroups)
            for g in range(Ngroups):
                fgXS[g] = np.dot( xs[ fewgroup_bin_edge_index[g]: fewgroup_bin_edge_index[g+1]], weighting_flux[ fewgroup_bin_edge_index[g]: fewgroup_bin_edge_index[g+1]] ) /np.sum(weighting_flux[ fewgroup_bin_edge_index[g]: fewgroup_bin_edge_index[g+1]])
            return fgXS
        else:
            fgXS = np.atleast_2d(np.zeros((Ngroups, Ngroups)))

            for gFrom in range(Ngroups):

                source = np.matmul(xs[ fewgroup_bin_edge_index[gFrom]: fewgroup_bin_edge_index[gFrom+1],:].T, weighting_flux[ fewgroup_bin_edge_index[gFrom]: fewgroup_bin_edge_index[gFrom+1]])
                for gTo in range(Ngroups):
                    numerator = np.sum(source[ fewgroup_bin_edge_index[gTo]: fewgroup_bin_edge_index[gTo+1]])
                    fgXS[gFrom, gTo] = numerator/np.sum(weighting_flux[ fewgroup_bin_edge_index[gFrom]: fewgroup_bin_edge_index[gFrom+1]])

            return fgXS

    def collapse_chi_to_fewgroup(self, chi:np.ndarray,fewgroup_bin_edge_index=[40] ):
        fewgroup_bin_edge_index = np.array(fewgroup_bin_edge_index, dtype=int)
        fewgroup_bin_edge_index = np.insert(fewgroup_bin_edge_index, 0, 0)
        fewgroup_bin_edge_index = np.append(fewgroup_bin_edge_index, len(chi))
        Ngroups = len(fewgroup_bin_edge_index)-1
        FG_chi = np.zeros(Ngroups)
        
        for g in range(Ngroups):
            FG_chi[g] = np.sum(chi[fewgroup_bin_edge_index[g]:fewgroup_bin_edge_index[g+1]])

        return FG_chi
        pass

class Fuel:

    @property
    def maren_standalone(self):
        return self._maren_standalone 

    @property
    def o16_normalizer(self):
        return self._o16_normalizer
    
    @property
    def o16_input_min(self):
        return self._o16_input_min
    
    @property
    def o16_input_max(self):
        return self._o16_input_max

    @property
    def fission_nuc_pca_list(self):
        return self._fission_nuc_pca_list

    @property
    def fission_pca_num_el(self):
        '''Number of LD coeffs per nuclide. Value of 0 means that for that nuclide there is no variance in the cross-section values 
        '''
        return self._fission_pca_num_el
    
    @property
    def fission_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._fission_dnn_min_output

    @property
    def fission_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._fission_dnn_max_output
    
    @property
    def fission_dnn_output_normalizer(self):
        return self._fission_dnn_output_normalizer

    @property
    def total_nuc_pca_list(self):
        return self._total_nuc_pca_list
    
    @property
    def total_pca_num_el(self):
        '''Number of LD coeffs per nuclide. Value of 0 means that for that nuclide there is no variance in the cross-section values 
        '''
        return self._total_pca_num_el

    @property
    def total_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_min_output

    @property
    def total_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_max_output

    @property
    def total_dnn_output_normalizer(self):
        return self._total_dnn_output_normalizer


    @property
    def p0scatter_pca(self):
        return self._p0scatter_pca

    @property
    def p0scatter_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_min_output

    @property
    def p0scatter_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_max_output    


    @property
    def p0scatter_dnn_output_normalizer(self):
        return self._p0scatter_dnn_output_normalizer

    @property
    def nu_pca(self):
        return self._nu_pca
    
    @property
    def nu_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._nu_dnn_min_output

    @property
    def nu_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._nu_dnn_max_output
    
    @property
    def nu_dnn_output_normalizer(self):
        return self._nu_dnn_output_normalizer    

    @property
    def chi_pca(self):
        return self._chi_pca
    
    @property
    def chi_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._chi_dnn_min_output

    @property
    def chi_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._chi_dnn_max_output

    @property
    def chi_dnn_output_normalizer(self):
        return self._chi_dnn_output_normalizer    

    def normalize_o16_input(self, o16_concentration):
        o16_concentration = np.clip(o16_concentration, self.o16_input_min, self.o16_input_max)
        o16_concentration = self.o16_normalizer.input(o16_concentration)
        return o16_concentration

    def output(self, nuclide_concentrations, instantaneous_state_parameters, return_microxs:bool = False) -> dict:
        '''Outputs 56-group cross-section libraries for the fuel pellet material

        Parameters
        ----------
        nuclide_concentrations : 1d np.ndarray
            The nuclide concentrations of the fuel pellet material in [atom/b-cm]

        instantaneous_state_parameters : 1d np.ndarray
            The instantaneous state parameters in the order:
                Tf : fuel temperature [K]
                Tm : moderator temperature [K]
                Rhom : moderator density [g/cm3]
                B : natural boron concentration in water [ppm]

        return_microxs : bool
            if True, then xs_library will contain the microscopic cross-section values in the key ['microxs_total', 'microxs_fission']

        Returns
        -------
        xs_library : dict
            xs_library containing the following keys
                'total'     : macroscopic total cross section. shape: (56,)
                'fission'   : macroscopic fission cross section. shape: (56,)
                'nu'        : neutron multiplicity of fuel pellet. shape: (56,)
                'chi'       : neutron fission distribution. shape: (56,)
                'p0scatter' : (n,xn) scattering matrix. scatter[i,j] represents the cross-section value from i'th energy group to j'th energy group. shape (56,56)

                The following fields are returned if option microxs == True
                'microxs_total'   : microscopic total cross section. shape: (56, Nnuc). microxs_total[:, n] are the cross-section values for the n'th nuclide
                'microxs_fission' : microscopic fission cross section. shape: (56, Nnuc). microxs_fission[:, n] are the cross-section values for the n'th nuclide

        '''

        assert len(nuclide_concentrations) == len(MAREN_standalone.fuelpellet_reconstruction_nuclide_list()), f"Expected vector of size {len(MAREN_standalone.fuelpellet_reconstruction_nuclide_list())} but received vector of length {len(nuclide_concentrations)}. See fuel.nuclide_list for list of nuclides for fuel pellet reconstruction"
        assert len(instantaneous_state_parameters) == 4

        # extract nuclide inputs
        NOI_input = self.maren_standalone.get_nuclide_inputs(nuclide_concentrations)
        # normalize nuclides of interest inputs
        NOI_input = self.maren_standalone.normalize_nuclide_inputs(NOI_input)

        # normalize isp inputs
        isp_input = self.maren_standalone.normalize_isp_inputs(instantaneous_state_parameters)

        o16_concentration = nuclide_concentrations[MAREN_standalone.fuelpellet_reconstruction_nuclide_list().index('O16')]
        o16_input = self.normalize_o16_input(o16_concentration)

        xs_library = {} 

        # total and fission cross-sections
        for xs in ['fission','total']:

            if return_microxs == True:
                micro_xs = np.zeros((self.fuelpellet_reconstruction_nuclide_list + 1, 56))

            macro_xs = np.zeros(56)

            # load tensorflow model
            dnn = tf.keras.models.load_model(f'{self.maren_standalone.savedir}/fuel/DNN_{xs}')
            min_output = getattr(self, f"{xs}_dnn_min_output")
            max_output = getattr(self, f"{xs}_dnn_max_output")
            output_normalizer = getattr(self, f"{xs}_dnn_output_normalizer")

            ldcoeff = forwardpass_dnn(input=np.concatenate([NOI_input, isp_input]), dnn=dnn, min_output=min_output, max_output=max_output, output_normalizer=output_normalizer)

            # number of ldcoeff per nuclide
            num_el = getattr(self, f'{xs}_pca_num_el')

            count = 0

            for c,t in enumerate(getattr(self,f'{xs}_nuc_pca_list')):
                if isinstance(t, PCA):
                    tmp = t.output(ldcoeff[count:count+num_el[c]])
                    tmp = np.clip(tmp, 0, None) # microXS cannot be < 0, therefore floor to 0
                    
                else:
                    # no PCA has been performed i.e. xs values are constant. This should be the case only for xs == fission and nuclide is a non-fissile
                    tmp = t

                if return_microxs == True:
                    micro_xs[c,:] = tmp
                
                if c != len(nuclide_concentrations):
                    macro_xs = macro_xs + tmp*nuclide_concentrations[c]
                else:
                    # `BASE` cross-section
                    macro_xs = macro_xs + tmp
                count += num_el[c]

            xs_library[f'{xs}'] = macro_xs
            if return_microxs == True:
                xs_library[f'micro_{xs}'] = micro_xs

        for xs in ['p0scatter','chi','nu',]:

            if xs == 'p0scatter':
                # o-16 concentration is an additional input to the model 
                dnn_input = np.concatenate([NOI_input, o16_input, isp_input])
            else:
                dnn_input = np.concatenate([NOI_input, isp_input])

            dnn = tf.keras.models.load_model(f'{self.maren_standalone.savedir}/fuel/DNN_{xs}')
            output_normalizer = getattr(self, f"{xs}_dnn_output_normalizer")
            min_output = getattr(self, f"{xs}_dnn_min_output")
            max_output = getattr(self, f"{xs}_dnn_max_output")

            ldcoeff = forwardpass_dnn(dnn_input, dnn, min_output, max_output, output_normalizer)

            pca = getattr(self, f'{xs}_pca')
            tmp = pca.output(ldcoeff)

            tmp = np.clip(tmp, 0, None)

            if xs == 'chi': # distribution of chi should sum to 1
                tmp = tmp/np.sum(tmp)

            if xs == 'p0scatter':
                tmp = np.reshape(tmp, (56,-1))

            xs_library[f'{xs}'] = tmp

        return xs_library

class BH2O:

    @property
    def maren_standalone(self):
        return self._maren_standalone 

    @property
    def total_ldcoeff_normalizer(self):
        return self._total_ldcoeff_normalizer
    
    @property
    def total_nuc_pca_list(self):
        return self._total_nuc_pca_list
    
    @property
    def total_pca_num_el(self):
        '''Number of LD coeffs per nuclide. Value of 0 means that for that nuclide there is no variance in the cross-section values 
        Nuclides to reconstruct bh2o: 'h-1', 'h-2', 'b-10', 'b-11', 'o-16','o-17'
        '''
        return self._total_pca_num_el
    
    @property
    def total_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_min_output

    @property
    def total_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_max_output
    
    @property
    def total_dnn_output_normalizer(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_output_normalizer

    @property
    def p0scatter_pca(self):
        return self._p0scatter_pca
    
    @property
    def p0scatter_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_min_output

    @property
    def p0scatter_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_max_output
    
    @property
    def p0scatter_dnn_output_normalizer(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_output_normalizer


    @staticmethod
    def bh2o_nuclides():
        return ['H1', 'H2', 'B10', 'B11', 'O16','O17']


    def output(self, fuelpellet_nuclide_concentrations, bh2o_nuclide_concentrations, instantaneous_state_parameters, return_microxs:bool = False):
        '''Outputs 56-group cross-section libraries for the fuel pellet material

        Parameters
        ----------
        fuelpellet_nuclide_concentrations : 1d np.ndarray
            The nuclide concentration of the fuel pellet material in [atom/b-cm]

        bh2o_nuclide_concentrations : 1d np.ndarray
            The nuclide concentration of the borated water in [atom/b-cm]

        instantaneous_state_parameters : 1d np.ndarray
            The instantaneous state parameters in the order:
                Tf : fuel temperature [K]
                Tm : moderator temperature [K]
                Rhom : moderator density [g/cm3]
                B : natural boron concentration in water [ppm]

        return_microxs : bool
            if True, then xs_library will contain the microscopic cross-section values in the key ['microxs_total',]

        Returns
        -------
        xs_library : dict
            xs_library containing the following keys
                'total'     : macroscopic total cross section. shape: (56,)
                'fission'   : macroscopic fission cross section. shape: (56,)
                'nu'        : neutron multiplicity of fuel pellet. shape: (56,)
                'chi'       : neutron fission distribution. shape: (56,)
                'p0scatter' : (n,xn) scattering matrix. scatter[i,j] represents the cross-section value from i'th energy group to j'th energy group. shape (56,56)

                The following fields are returned if option microxs == True
                'microxs_total'   : microscopic total cross section. shape: (56, Nnuc). microxs_total[:, n] are the cross-section values for the n'th nuclide
                'microxs_fission' : microscopic fission cross section. shape: (56, Nnuc). microxs_fission[:, n] are the cross-section values for the n'th nuclide

        '''

        assert len(fuelpellet_nuclide_concentrations) == len(MAREN_standalone.fuelpellet_reconstruction_nuclide_list()), f"Expected vector of size {len(MAREN_standalone.fuelpellet_reconstruction_nuclide_list())} but received vector of length {len(fuelpellet_nuclide_concentrations)}. See MAREN_standalone.nuclide_list for list of nuclides for fuel pellet reconstruction"

        assert len(bh2o_nuclide_concentrations) == len(self.bh2o_nuclides()), f"Expected vector of size {len(self.bh2o_nuclides())} but received vector of length {len(bh2o_nuclide_concentrations)}. See BH2O.bh2o_nuclides for list of nuclides for borated water reconstruction"

        assert len(instantaneous_state_parameters) == 4

        # extract nuclide inputs
        NOI_input = self.maren_standalone.get_nuclide_inputs(fuelpellet_nuclide_concentrations)

        # normalize nuclides of interest inputs
        NOI_input = self.maren_standalone.normalize_nuclide_inputs(NOI_input)

        # normalize isp inputs
        isp_input = self.maren_standalone.normalize_isp_inputs(instantaneous_state_parameters)

        xs_library = {} 

        # total cross-section
        for xs in ['total', 'p0scatter']:

            # load tensorflow model    
            dnn = tf.keras.models.load_model(f'{self.maren_standalone.savedir}/bh2o/DNN_{xs}')
            dnn_input = np.concatenate([NOI_input, isp_input])
            min_output = getattr(self, f"{xs}_dnn_min_output")
            max_output = getattr(self, f"{xs}_dnn_max_output")
            output_normalizer = getattr(self, f"{xs}_dnn_output_normalizer")

            ldcoeff = forwardpass_dnn(dnn_input, dnn, min_output, max_output, output_normalizer)

            if xs == 'total':
                if return_microxs == True:
                    micro_xs = np.zeros((len(self.bh2o_nuclides()), 56))

                macro_xs = np.zeros(56)

                # number of ldcoeff per nuclide
                num_el = getattr(self, f'total_pca_num_el')
                count = 0

                for c,t in enumerate(getattr(self,f'{xs}_nuc_pca_list')):
                    
                    tmp = t.output(ldcoeff[count:count+num_el[c]])
                    tmp = np.clip(tmp, 0, None) # microXS cannot be < 0, therefore floor to 0

                    if return_microxs == True:
                        micro_xs[c,:] = tmp
                    
                    macro_xs = macro_xs + tmp*bh2o_nuclide_concentrations[c]

                    count += num_el[c]

                xs_library[f'total'] = macro_xs
                if return_microxs == True:
                    xs_library[f'micro_total'] = micro_xs
            else:
                # p0scatter

                p0scatter = self.p0scatter_pca.output(ldcoeff)
                p0scatter = np.reshape(p0scatter, (56,-1))

                xs_library['p0scatter'] = p0scatter

                pass

        return xs_library
    
class Nonfuelpellet:

    @property
    def name(self):
        return self._name

    @property
    def maren_standalone(self):
        return self._maren_standalone 
    
    @property
    def total_pca(self):
        return self._total_pca

    @property
    def total_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_min_output

    @property
    def total_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_max_output
    
    @property
    def total_dnn_output_normalizer(self):
        '''for clamping the outputs 
        '''
        return self._total_dnn_output_normalizer

    @property
    def p0scatter_pca(self):
        return self._p0scatter_pca
    
    @property
    def p0scatter_dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_min_output

    @property
    def p0scatter_dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_max_output
    
    @property
    def p0scatter_dnn_output_normalizer(self):
        '''for clamping the outputs 
        '''
        return self._p0scatter_dnn_output_normalizer

    @property
    def use_Tf(self):
        '''Materials in fuel pin model should use Tf 
        '''
        return self._use_Tf

    def output(self, instantaneous_state_parameters):
        '''Outputs 56-group cross-section libraries for the material

        Parameters
        ----------

        instantaneous_state_parameters : 1d np.ndarray of length 3 or 4
            if self.use_Tf == True then 
            The instantaneous state parameters in the order:
                Tf : fuel temperature [K]
                Tm : moderator temperature [K]
                Rhom : moderator density [g/cm3]
                B : natural boron concentration in water [ppm]

            if self.use_Tf == False, then no Tf and the vector should contain only 3 parameters
            
        Returns
        -------
        xs_library : dict
            xs_library containing the following keys
                'total'     : macroscopic total cross section. shape: (56,)
                'p0scatter' : (n,xn) scattering matrix. scatter[i,j] represents the cross-section value from i'th energy group to j'th energy group. shape (56,56)
        '''
    
        pass

        if self.use_Tf == True:
            assert len(instantaneous_state_parameters) == 4, "Inputs to this model are the instantaneous state parameters: Tf, Tm, Rhom, B"
        else:
            assert len(instantaneous_state_parameters) == 3, "Inputs to this model are the instantaneous state parameters: Tm, Rhom, B"

        # normalize isp inputs
        if self.use_Tf == True:
            isp_input = self.maren_standalone.normalize_isp_inputs(instantaneous_state_parameters)
        else:
            isp_input = self.maren_standalone.normalize_isp_inputs(np.concatenate([[0],instantaneous_state_parameters]))
            isp_input = isp_input[1:]

        xs_library = {} 

        # total cross-section
        for xs in ['total', 'p0scatter']:

            # load tensorflow model    
            dnn = tf.keras.models.load_model(f'{self.maren_standalone.savedir}/{self.name}/DNN_{xs}')
            dnn_input = np.concatenate([isp_input])
            output_normalizer = getattr(self, f"{xs}_dnn_output_normalizer")
            min_output = getattr(self, f"{xs}_dnn_min_output")
            max_output = getattr(self, f"{xs}_dnn_max_output")
            ldcoeff = forwardpass_dnn(dnn_input, dnn, min_output, max_output, output_normalizer)

            pca = getattr(self, f"{xs}_pca")
            t = pca.output(ldcoeff)

            if xs == 'p0scatter':
                t = np.reshape(t, (56,-1))
            
            xs_library[f'{xs}'] = t

        return xs_library
            
class Flux:
    @property
    def maren_standalone(self):
        return self._maren_standalone 
    
    @property
    def pca(self):
        return self._pca

    @property
    def dnn_min_output(self):
        '''for clamping the outputs 
        '''
        return self._dnn_min_output

    @property
    def dnn_max_output(self):
        '''for clamping the outputs 
        '''
        return self._dnn_max_output
    
    @property
    def dnn_output_normalizer(self):
        '''for clamping the outputs 
        '''
        return self._dnn_output_normalizer
    
    @property
    def min_flux(self):
        return self._min_flux
    
    @property
    def max_flux(self):
        return self._max_flux

    def output(self, fuelpellet_nuclide_concentrations:np.ndarray, instantaneous_state_parameters:np.ndarray):
        '''
        Parameters
        ----------
        nuclide_concentrations : 1d np.ndarray
            The nuclide concentrations of the fuel pellet material in [atom/b-cm]

        instantaneous_state_parameters : 1d np.ndarray
            The instantaneous state parameters in the order:
                Tf : fuel temperature [K]
                Tm : moderator temperature [K]
                Rhom : moderator density [g/cm3]
                B : natural boron concentration in water [ppm]

        Returns
        -------
        fuel_flux : np.ndarray (56,) note: units are 1/cm2-s (converting to 1/cm2-s-eV i.e. dividing by energy bin width, can be done by np.diff(MAREN_standalone.scale_56group_library()) )

        bh2o_flux : np.ndarray (56,) note: units are 1/cm2-s
        
        '''

        # extract nuclide inputs
        NOI_input = self.maren_standalone.get_nuclide_inputs(fuelpellet_nuclide_concentrations)
        # normalize nuclides of interest inputs
        NOI_input = self.maren_standalone.normalize_nuclide_inputs(NOI_input)

        # normalize isp inputs
        isp_input = self.maren_standalone.normalize_isp_inputs(instantaneous_state_parameters)

        # load tensorflow model
        dnn = tf.keras.models.load_model(f'{self.maren_standalone.savedir}/flux/DNN')
        min_output = getattr(self, f"dnn_min_output")
        max_output = getattr(self, f"dnn_max_output")
        output_normalizer = getattr(self, f"dnn_output_normalizer")

        ldcoeff = forwardpass_dnn(input=np.concatenate([NOI_input, isp_input]), dnn=dnn, min_output=min_output, max_output=max_output, output_normalizer=output_normalizer)

        flux = self.pca.output(ldcoeff)

        # floor to min and max
        for n in range(len(flux)):
            flux[n] = np.clip(flux[n], self.min_flux[n], self.max_flux[n])
        

        fuel_flux = flux[0:56]
        bh2o_flux = flux[56:]

        return fuel_flux, bh2o_flux


def forwardpass_dnn(input:np.ndarray, dnn:tf.keras.Model, min_output:np.ndarray, max_output:np.ndarray, output_normalizer:Znormalize):

    # note: use __call__ instead of dnn.predict() in-order to avoid re-tracing warning (see also: https://www.tensorflow.org/guide/function#controlling_retracing), if dnn.predict method is preferred, use the following:
    # ldcoeff = dnn.predict(input[np.newaxis,:], verbose=0).flatten()
    ldcoeff = dnn(input[np.newaxis,:]).numpy().flatten()
    
    ldcoeff = output_normalizer.output(ldcoeff)

    for i in range(len(ldcoeff)):
        ldcoeff[i] = np.clip(ldcoeff[i], min_output[i], max_output[i])

    return ldcoeff