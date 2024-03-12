# MAREN_standalone
This project implements a MAchine learning based REpresentation model for Nuclear cross-section libraries (MAREN). The cross-section libraries are based on fuel pellet nuclide concentrations and instantaneous state parameters. The model here is a stand-alone version. Users can run the stand-alone version using pre-trained MAREN to generate 56 group XS libraries and arbitrary few-group XS libraries. This project requires the following packages:

tensorflow >= 2.15.0
sklearn >= 1.3.0

See main.py for example to construct multi-group and few-group cross-section libraries. Neural networks are saved as tf.keras.Models under the folder "maren_sa_01042024".

 Expected usage of MAREN is to couple with discrete energy neutron transport solvers e.g. OpenMOC (a deterministic discrete ordinates method-of-characteristics code). The next update of this project will include methods to parse the cross-section libraries into OpenMOC libraries. 
