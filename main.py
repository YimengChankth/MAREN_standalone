from MAREN_standalone import MAREN_standalone

import numpy as np
import tabulate
import matplotlib.pyplot as plt


'''
This MAREN standalone was created in 2023-2024 by Yi Meng Chan for research conducted at the KTH Royal Institute of Technology. Further information on usage can be found at: https://github.com/YimengChankth/MAREN_standalone. Details and explanations on MAREN can be found at: \n \tYi Meng Chan and Jan Dufek. A deep-learning representation of multi-group cross sections in lattice calculations. Annals of Nuclear Energy, 195:110123, 2024. ISSN 0306-4549. doi: https://doi.org/10.1016/j.anucene.2023.110123. URL: https://www.sciencedirect.com/science/article/pii/S0306454923004425

'''


def example_generate_xs_library():
    '''This example shows how to generate 56-group libraries as well as a conventional fast-thermal library for a typical fresh 3.1% enriched fuel pellet material and associated materials, including materials present in control rods and instrumentation tubes. By default cross-section libraries are generated for the following materials (Terms in brackets are the dictionary keys)
        
        fuel pellet ("fuel"), he gap ("he"), zirc4 clad ("zirc4"), borated water ("bh2o"), 
        instrumentation tube: stainless steel ("ss"), air ("air")
        control rod: boron-carbide ("b4c")

    The instantaneous state parameters are set to: 

    Fuel temperature (Tf)      : 900 K
    Moderator temperature (Tm) : 550 K
    Moderator density (Rhom)   : 0.75 g/cm3
    Boron concentration (B)    : 300 ppm

    Fresh 3.1% fuel pellet nuclide composition (see: BEAVRS reference):
    ┌───────────┬────────────────────────────────────┐
    │ Nuclide   │ Atomic concentration (atom/b-cm)   │
    ├───────────┼────────────────────────────────────┤
    │ U234      │ 5.798700e-06                       │
    ├───────────┼────────────────────────────────────┤
    │ U235      │ 7.217500e-04                       │
    ├───────────┼────────────────────────────────────┤
    │ U238      │ 2.225300e-02                       │
    ├───────────┼────────────────────────────────────┤
    │ O16       │ 4.585300e-02                       │
    └───────────┴────────────────────────────────────┘

    borated water 300 ppm nuclide composition
    ┌───────────┬────────────────────────────────────┐
    │ Nuclide   │ Atomic concentration (atom/b-cm)   │
    ├───────────┼────────────────────────────────────┤
    │ H1        │ 5.013417e-02                       │
    ├───────────┼────────────────────────────────────┤
    │ H2        │ 7.809111e-06                       │
    ├───────────┼────────────────────────────────────┤
    │ B10       │ 2.484668e-06                       │
    ├───────────┼────────────────────────────────────┤
    │ B11       │ 1.005150e-05                       │
    ├───────────┼────────────────────────────────────┤
    │ O16       │ 2.501133e-02                       │
    ├───────────┼────────────────────────────────────┤
    │ O17       │ 9.501904e-06                       │
    └───────────┴────────────────────────────────────┘

    '''

    # MAREN model reconstructs fission and total macroscopic cross-section by summing individual nuclide concentrations. This version of MAREN uses 91 nuclides to reconstruct the fuel pellet
    reconstruction_nuclide_list = MAREN_standalone.fuelpellet_reconstruction_nuclide_list()

    fuelpellet_composition = {
        'U234':5.7987e-06,
        'U235':7.2175e-04,
        'U238':2.2253e-02,
        'O16' :4.5853e-02,
    }

    fuelpellet_comp_input = np.zeros(len(reconstruction_nuclide_list))

    for nuc in fuelpellet_composition.keys():
        ind = reconstruction_nuclide_list.index(nuc)
        fuelpellet_comp_input[ind] = fuelpellet_composition[nuc]

    bh2o_composition = [5.01341657e-02, 7.80911115e-06, 2.48466803e-06, 1.00514976e-05, 2.50113335e-02, 9.50190422e-06]

    isp = [900, 550, 0.75, 300]

    # Load maren model
    maren = MAREN_standalone.load_from_dir('maren_sa_01042024')

    # predict the mgxs with the original 56 energy group edges (see maren.scale_56group_library() for the values of the energy edges)
    mgxs = maren.predict_multigroup_xslibraries(fuelpellet_nuclide_concentrations=fuelpellet_comp_input, instantaneous_state_parameters=isp, bh2o_nuclide_concentrations=bh2o_composition)

    print(f'MGXS contains the following materials:')
    for mat in mgxs.keys():
        print(f'\t{mat}')
        for xs in mgxs[mat].keys():
            print(f'\t\t{xs}')

    # print some values
    tabular_data = np.zeros((56,5),dtype=object)
    tabular_data[:,0] = np.arange(56)
    tabular_data[:,1] = mgxs['fuel']['fission']
    tabular_data[:,2] = mgxs['fuel']['total']
    tabular_data[:,3] = mgxs['fuel']['nu']
    tabular_data[:,4] = mgxs['fuel']['chi']
    
    print(f'\nFuel pellet fission and total cross-section values:\n Group 0 - Highest energy, Group 55 - Lowest energy')
    print(tabulate.tabulate(tabular_data=tabular_data, headers=['Group','Fission XS', 'Total XS', 'nu','chi'], floatfmt='1.6e', tablefmt='simple_grid'))

    # plot the fission and total cross-section values for the fuel pellet 
    fig = plt.figure()
    plt.plot(mgxs['fuel']['fission'], label='fission')
    plt.plot(mgxs['fuel']['total'], label='total')
    plt.xlabel('Energy group index [-]')
    plt.ylabel(r'$\Sigma$ [1/cm]')
    plt.yscale('log')
    plt.title(f'Fuel pellet macroscopic xs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sample_fuelpellet_macroscopic_xs.png')
    plt.close()

    # plot the total cross-sections for bh2o, zirc4 and b4c
    fig = plt.figure()
    for mat in ['bh2o', 'zirc4', 'b4c']:
        plt.plot(mgxs[mat]['total'],label=mat)
    plt.xlabel('Energy group index [-]')
    plt.ylabel(r'$\Sigma$ [1/cm]')
    plt.yscale('log')
    plt.title(f'Material total macroscopic xs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sample_materials_macroscopic_xs.png')
    plt.close()
    

    # spy plot scattering matrix as a spy plot to verify that most off-diagonal elements are 0. p0scatter[i,j] represents the nu-scattering value from energy group i to energy group j
    scatter_spy = plt.spy(mgxs['fuel']['p0scatter'], marker='o', markersize=3)
    plt.ylabel('Incident neutron energy index [-]')
    plt.xlabel('Outgoing neutron energy index [-]')
    plt.savefig('sample_mgxs_fuel_p0scatter_matrix.png')
    plt.close()

    # Few group library generation
    print(f"Generating few-group library, defining energy groups 0:39 as the 'fast' group and groups 40:55 as the 'thermal' group. This corresponds to a energy boundary at {MAREN_standalone.scale_56group_library()[40]} eV")
    fgxs, fuel_flux, bh2o_flux = maren.predict_fewgroup_xslibraries(fuelpellet_nuclide_concentrations=fuelpellet_comp_input, instantaneous_state_parameters=isp, bh2o_nuclide_concentrations=bh2o_composition, fewgroup_bin_edge_index=[40])

    # print some values
    tabular_data = np.zeros((2,5),dtype=object)
    tabular_data[:,0] = ['Fast','Thermal']
    tabular_data[:,1] = fgxs['fuel']['fission']
    tabular_data[:,2] = fgxs['fuel']['total']
    tabular_data[:,3] = fgxs['fuel']['nu']
    tabular_data[:,4] = fgxs['fuel']['chi']

    print(f'\nFuel pellet fission and total few-group cross-section values:\n Group 0 - Fast, Group 1 - Thermal')
    print(tabulate.tabulate(tabular_data=tabular_data, headers=['Group','Fission XS', 'Total XS','nu','chi'], floatfmt='1.6e', tablefmt='simple_grid'))

    energy_bin_width = np.abs(np.diff(MAREN_standalone.scale_56group_library()))

    fig = plt.figure()
    plt.plot(fuel_flux/energy_bin_width, label='fuel')
    plt.plot(bh2o_flux/energy_bin_width, label='bh2o')
    plt.yscale('log')
    plt.title(f'Flux spectra divided by energy bin widths')
    plt.xlabel(f'Group index [-]')
    plt.ylabel(r'$\phi_{g}$ [1/cm2-s-eV]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'sample_flux_prediction.png')
    plt.close()

if __name__ == "__main__":

    
    example_generate_xs_library()

    pass