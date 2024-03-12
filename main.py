from MAREN_standalone import MAREN_standalone

import matplotlib.pyplot as plt
import numpy as np

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

    bh2o_composition = [5.01341657e-02, 7.80911115e-06, 2.48466803e-06, 1.00514976e-05, 2.50113335e-02, 9.50190422e-06]

    isp = [900, 550, 0.75, 300]

    # Load maren model
    maren = MAREN_standalone.load_from_dir('maren_sa_01042024')

    # predict the mgxs with the original 56 energy group edges (see maren.scale_56group_library() for the values of the energy edges)
    mgxs = maren.predict_multigroup_xslibraries(fuelpellet_nuclide_concentrations=fuelpellet_comp_input, instantaneous_state_parameters=isp, bh2o_nuclide_concentrations=bh2o_composition)

    pass

def get_bh2o_openmc(rho, b):
    
    import openmc
    bh2oMat = openmc.model.borated_water(boron_ppm = b, density=rho)
    atom_dens = bh2oMat.get_nuclide_atom_densities()

    OpenMCnames = ['H1','H2','B10','B11','O16','O17']

    nuc = np.zeros(6)

    for c,v in enumerate(OpenMCnames):
        nuc[c] = atom_dens[v]

    import tabulate

    table_vals = np.zeros((6,2),dtype=object)
    table_vals[:,0] = OpenMCnames
    table_vals[:,1] = nuc
    headers=['Nuclide','Atomic concentration (atom/b-cm)']

    print(tabulate.tabulate(table_vals, headers=headers, tablefmt='simple_grid', floatfmt='1.6e',numalign="left"))

    return nuc


def get_31_fp():
    import tabulate
    
    nuclide_names = ['U234','U235','U238','O16']
    atm_conc = [ 5.7987e-06 , 7.2175e-04, 2.2253e-02, 4.5853e-02]

    table_vals = np.zeros((4,2),dtype=object)
    table_vals[:,0] = nuclide_names
    table_vals[:,1] = atm_conc

    headers=['Nuclide','Atomic concentration (atom/b-cm)']
    print(tabulate.tabulate(table_vals, headers=headers, tablefmt='simple_grid', floatfmt='1.6e',numalign="left"))

if __name__ == "__main__":
    # get_bh2o_openmc(0.75, 300)
    # get_31_fp()
    
    example_generate_xs_library()

    pass