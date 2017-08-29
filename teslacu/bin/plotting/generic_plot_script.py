"""
"""

import os
import time
from generic_plotting_functions import *

# ddate = '2017_07_29/'

case = 'test_N64_K15_smooth'
N = 64
nk = (N/2)+1
ttags = ['%3.3d' % i for i in range(18, 200, 20)]

data_dir = ('/Users/colin/workspace/scratch/spectralLES/'
            'HIT_LES/%s' % case)
fig_dir = ('/Users/colin/Google Drive/workspace/media/'
           'spectralLES/HIT_LES/%s' % (time.strftime("%Y_%m_%d")))

try:
    # os.makedirs('%s/pdfs/multirun' % fig_dir)
    os.makedirs('%s/pdfs/tseries' % fig_dir)
    # os.makedirs('%s/spectra/multirun' % fig_dir)
    os.makedirs('%s/spectra/tseries' % fig_dir)
except OSError:
    if not os.path.isdir(fig_dir):
        raise

# Rgas = 8.314472296e7
# R = Rgas/26.8
# Tcoef = 3.41e-6
# T0 = 293.0
# P0 = 1.01325e6
# y = 1.3
# rho = P0/R/T0  # density corresponding to 1 atm / R*293 K
# nu0 = Tcoef/rho*T0**0.7

# K = 128
# N = 256
# nk = (N/2)+1
# config = 'CTU3_exact'
# rtag = 1
# ttags = range(1, 21)
# Md, Mc = 0.4, '0,4'
# case = 'M{0}_K{1}_N{2}_{3}'.format(Mc, K, N, config)

# K = 128
# Nx = [64, 128, 256, 512]
# configs = ['CTU3_exact', ]*4  # 'CTU2_exact', 'CTU2_hllc'
# rtags = [range(1, 33)]*4
# ttags = [[16], ]*4

# aMa = np.asarray([0.2, 0.4, 1.0])
# aMlab = ['0,2', '0,4', '1,0']

# aMa = np.asarray([0.1, 0.2, 0.4, 0.6, 1.0])
# aMlab = ['0,1', '0,2', '0,4', '0,6', '1,0']

# --------------

spect_tags = ['u', 'omga']
# ['u', 'v', 'Smm', 'T', 'P', 'rho', 'omga', 'gradT']
spect_titles = ['Velocity PSD',
                'Vorticity PSD']
# spect_titles = ['Velocity PSD',
#                 'Kinetic Energy Spectrum',
#                 'Dilatation PSD',
#                 'Temperature PSD',
#                 'Pressure PSD',
#                 'Density PSD',
#                 'Vorticity PSD',
#                 'Temperature Gradient PSD']

# hist_tags = ['T_tilde', 'Smm_tilde', 'enst_tilde', 'dTdT_tilde',
#              'repss', 'repsd', 'Pdil', 'repsT']
# hist_titles = ['Temperature',
#                'Dilatation',
#                'Enstrophy',
#                'Temperature Gradient',
#                'Solenoidal KE Dissipation',
#                'Dilatational KE Dissipation',
#                'Pressure-Dilatation Work',
#                'Temperature Scalar Dissipation']
# hist_xlabs = ['$T$',
#               '$\Theta$',
#               '$\Omega$',
#               '$\\frac{\partial T}{\partial x_i}'
#               '\\frac{\partial T}{\partial x_i}$',
#               '$\\rho\\varepsilon^\mathrm{s}$',
#               '$\\rho\\varepsilon^\mathrm{d}$',
#               '$P\Theta$',
#               '$\kappa\\frac{\partial T}{\partial x_i}'
#               '\\frac{\partial T}{\partial x_i}$']

# -----------------

# for Md, Mc in zip(aMa, aMlab):

#     cases = ['M{0}_K{1}_N{2}_{3}'.format(Mc, K, N, config)
#              for N, config in zip(Nx, configs)]

#     for c in range(len(cases)):

#         for i in range(len(hist_tags)):

#             plot_multirun_histogram(
#                     data_dir, fig_dir, cases[c], ttags[c][0], rtags[c],
#                     hist_tags[i], xlab=hist_xlabs[i], title=hist_titles[i])

spec_dir = '%s/analysis' % data_dir
for i in range(len(spect_tags)):
    plot_tseries_spectra(spec_dir, fig_dir, case, ttags, spect_tags[i], nk,
                         spect_titles[i])

plt.close('all')
