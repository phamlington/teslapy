from math import *

workdir='/Users/colin/workspace/scratch/spectralLES'  # not a problem arg

pid = 'test_N64_K30_spectral'                       # problem ID
odir = '%s/HIT_LES/%s/' % (workdir, pid)            # output folder
idir = odir                                         # input folder for restarts
L = 2.*pi                                           # domain size
N = 64                                              # linear grid size
cfl = 0.5                                           # CFL no.
tlimit = 10.*pi                                     # Time Limit ~ 20tau
tau = 0.5*pi                                        # ~ integral turnover time
dt_rst = 5.0*tau                                    # restart output rate
dt_bin = 0.5*tau                                    # snapshot output rate
dt_hst = 0.2*tau                                    # histogram output rate
dt_spec= 0.1*tau                                    # spectrum output rate
nu = 0.0011                                         # kinematic viscosity
eps_inj = 1.2                               # energy injection rate (cell avg.)
Urms = 2.0                                  # initial rms velocity
k_exp = -1.0                                # initial spectrum power law
k_peak = 16                                 # initial spectrum decay scaling

"""
For simulations at 256^3:
dx/eta = 1 -> nu = 0.00706 (DNS for 2nd-3rd order statistics)
dx/eta = 2 -> nu = 0.0028  (DNS for at best 2nd-order statistics)
dx/eta = 4 -> nu = 0.0011  (well-resolved LES)
eps_inj = 1.2
k_peak = 0-64 (depending on how much small-scale energy you'd like in IC)


Everything else stays the same.

Equivalence at 64^3: dx/eta =  4, 8, and 16
"""
