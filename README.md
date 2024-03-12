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

Running main.py should give the following output (including various .png files)

MGXS contains the following materials:
        fuel
                fission
                total
                p0scatter
                chi
                nu
        bh2o
                total
                p0scatter
        he
                total
                p0scatter
        zirc4
                total
                p0scatter
        air
                total
                p0scatter
        ss
                total
                p0scatter
        b4c
                total
                p0scatter

Fuel pellet fission and total cross-section values:
 Group 0 - Highest energy, Group 55 - Lowest energy
┌─────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│   Group │   Fission XS │     Total XS │           nu │          chi │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       0 │ 2.230111e-02 │ 2.006928e-01 │ 3.506006e+00 │ 1.931594e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       1 │ 1.350763e-02 │ 2.387999e-01 │ 3.112115e+00 │ 6.913837e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       2 │ 1.290847e-02 │ 2.894812e-01 │ 2.823157e+00 │ 1.248999e-01 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       3 │ 1.287151e-02 │ 2.263626e-01 │ 2.666023e+00 │ 2.275117e-01 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       4 │ 1.070099e-02 │ 2.530875e-01 │ 2.611045e+00 │ 9.779336e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       5 │ 4.102657e-03 │ 2.967777e-01 │ 2.584872e+00 │ 9.367916e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       6 │ 1.323619e-03 │ 3.532941e-01 │ 2.545734e+00 │ 1.146421e-01 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       7 │ 9.227056e-04 │ 2.969442e-01 │ 2.514279e+00 │ 3.885817e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       8 │ 8.462889e-04 │ 3.107880e-01 │ 2.497737e+00 │ 5.260285e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│       9 │ 8.311330e-04 │ 3.722151e-01 │ 2.486115e+00 │ 4.477417e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      10 │ 8.789087e-04 │ 4.998392e-01 │ 2.478176e+00 │ 4.518735e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      11 │ 8.948284e-04 │ 3.838957e-01 │ 2.474412e+00 │ 1.770331e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      12 │ 9.402423e-04 │ 3.947249e-01 │ 2.470843e+00 │ 1.886423e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      13 │ 1.123227e-03 │ 4.386428e-01 │ 2.442732e+00 │ 3.048754e-02 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      14 │ 1.469367e-03 │ 4.839216e-01 │ 2.425429e+00 │ 3.418308e-03 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      15 │ 1.753183e-03 │ 5.105118e-01 │ 2.429193e+00 │ 2.624857e-04 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      16 │ 2.364234e-03 │ 5.184114e-01 │ 2.433061e+00 │ 8.461569e-04 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      17 │ 3.678658e-03 │ 5.840225e-01 │ 2.433588e+00 │ 5.899007e-06 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      18 │ 8.549189e-03 │ 5.606738e-01 │ 2.434069e+00 │ 8.260940e-06 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      19 │ 4.154998e-03 │ 3.585757e+00 │ 2.434177e+00 │ 1.547872e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      20 │ 1.552166e-02 │ 4.778239e-01 │ 2.433736e+00 │ 2.859518e-07 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      21 │ 2.293990e-02 │ 5.362188e+00 │ 2.432715e+00 │ 6.110124e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      22 │ 1.124029e-02 │ 5.529987e-01 │ 2.433821e+00 │ 4.480763e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      23 │ 1.112441e-02 │ 2.678526e+00 │ 2.433300e+00 │ 1.547902e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      24 │ 1.847759e-02 │ 4.784016e-01 │ 2.434256e+00 │ 1.372751e-07 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      25 │ 2.674616e-03 │ 2.394996e+00 │ 2.437889e+00 │ 1.018369e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      26 │ 2.936091e-02 │ 5.925990e-01 │ 2.435286e+00 │ 1.135277e-07 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      27 │ 1.371983e-02 │ 1.247852e+01 │ 2.434930e+00 │ 4.603050e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      28 │ 3.319779e-02 │ 4.809277e-01 │ 2.435929e+00 │ 5.804742e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      29 │ 1.004758e-02 │ 1.753965e+00 │ 2.435973e+00 │ 2.240431e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      30 │ 4.482552e-02 │ 2.286403e+01 │ 2.436948e+00 │ 2.851456e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      31 │ 3.711894e-02 │ 4.889233e-01 │ 2.436888e+00 │ 5.499242e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      32 │ 1.752635e-02 │ 2.286033e+00 │ 2.437495e+00 │ 5.091906e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      33 │ 8.209055e-03 │ 2.343460e+01 │ 2.436168e+00 │ 1.527571e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      34 │ 5.866820e-02 │ 1.678632e+00 │ 2.434313e+00 │ 1.018382e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      35 │ 1.254422e-02 │ 4.726063e-01 │ 2.438702e+00 │ 5.091910e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      36 │ 1.465521e-02 │ 4.175493e-01 │ 2.437496e+00 │ 1.576452e-08 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      37 │ 7.290451e-02 │ 5.009079e-01 │ 2.436541e+00 │ 2.104355e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      38 │ 5.884953e-02 │ 4.931122e-01 │ 2.435996e+00 │ 2.851470e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      39 │ 4.228427e-02 │ 4.523274e-01 │ 2.436410e+00 │ 1.568308e-09 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      40 │ 5.608896e-02 │ 4.713458e-01 │ 2.436159e+00 │ 7.128673e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      41 │ 8.131257e-02 │ 5.050247e-01 │ 2.435239e+00 │ 3.055146e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      42 │ 1.025410e-01 │ 5.340525e-01 │ 2.435252e+00 │ 2.958175e-11 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      43 │ 1.171431e-01 │ 5.548178e-01 │ 2.436424e+00 │ 2.958175e-11 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      44 │ 1.359842e-01 │ 5.843394e-01 │ 2.437660e+00 │ 3.055146e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      45 │ 1.290749e-01 │ 5.770448e-01 │ 2.435272e+00 │ 2.104357e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      46 │ 1.321662e-01 │ 5.800066e-01 │ 2.436093e+00 │ 2.104357e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      47 │ 1.596862e-01 │ 6.148905e-01 │ 2.436788e+00 │ 2.104358e-10 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      48 │ 1.958709e-01 │ 6.616840e-01 │ 2.436891e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      49 │ 2.314024e-01 │ 7.084033e-01 │ 2.436905e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      50 │ 2.689043e-01 │ 7.581387e-01 │ 2.436866e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      51 │ 3.040169e-01 │ 8.055039e-01 │ 2.436837e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      52 │ 3.673257e-01 │ 8.926937e-01 │ 2.436859e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      53 │ 5.155100e-01 │ 1.102887e+00 │ 2.436862e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      54 │ 8.342058e-01 │ 1.567744e+00 │ 2.436841e+00 │ 0.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│      55 │ 1.404728e+00 │ 2.439020e+00 │ 2.436811e+00 │ 0.000000e+00 │
└─────────┴──────────────┴──────────────┴──────────────┴──────────────┘
Generating few-group library, defining energy groups 0:39 as the 'fast' group and groups 40:55 as the 'thermal' group. This corresponds to a energy boundary at 0.625 eV

Fuel pellet fission and total few-group cross-section values:
 Group 0 - Fast, Group 1 - Thermal
┌─────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Group   │   Fission XS │     Total XS │           nu │          chi │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Fast    │ 8.687896e-03 │ 4.127837e-01 │ 2.542593e+00 │ 1.000000e+00 │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Thermal │ 2.261642e-01 │ 7.046806e-01 │ 2.436589e+00 │ 2.014367e-09 │
└─────────┴──────────────┴──────────────┴──────────────┴──────────────┘