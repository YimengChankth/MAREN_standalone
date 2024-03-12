# MAREN_standalone
This project implements a MAchine learning based REpresentation model for Nuclear cross-section libraries (MAREN). The cross-section libraries are based on fuel pellet nuclide concentrations and instantaneous state parameters. The model here is a stand-alone version. Users can run the stand-alone version using pre-trained MAREN to generate 56 group XS libraries and arbitrary few-group XS libraries. This project requires the following packages:

tensorflow >= 2.15.0
sklearn >= 1.3.0

See main.py for example to construct multi-group and few-group cross-section libraries. Neural networks are saved as tf.keras.Models under the folder "maren_sa_01042024".

 Expected usage of MAREN is to couple with discrete energy neutron transport solvers e.g. OpenMOC (a deterministic discrete ordinates method-of-characteristics code). The next update of this project will include methods to parse the cross-section libraries into OpenMOC libraries. 


Publication on MAREN details and methodology can be found at:

    Yi Meng Chan and Jan Dufek. A deep-learning representation of multi-group cross sections in lattice calculations. Annals of Nuclear Energy, 195:110123, 2024. ISSN 0306-4549. doi: https://doi.org/10.1016/j.anucene.2023.110123. URL: https://www.sciencedirect.com/science/article/pii/S0306454923004425

Geometry model specifications can be found in the BEAVRS benchmark problem specifications at:

    N. Horelik, B. Herman, B. Forget, and K. Smith. Benchmark for evaluation and validation of reactor simulations (beavrs), v1.0.1. 
    In Proc. Int. Conf. Mathematics and Computational Methods Applied to Nuc. Sci. and Eng., Sun Valley, Idaho, 2013
