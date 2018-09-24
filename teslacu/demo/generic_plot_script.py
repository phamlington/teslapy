"""
"""
import numpy as np
from math import sqrt, log, log10, pi
import os
import time
import matplotlib.pyplot as plt
from matplotlib import colors, ticker


def image_triptic(data, norm, cmap, titles, fname):
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
    fig.set_size_inches(5.0, 2.0)
    # plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.8)
    cax = plt.axes([1.0, 0.1, 0.02, 0.8])

    plt.sca(ax1)
    plt.imshow(data[0], cmap=cmap, aspect=1, norm=norm)
    plt.title(titles[0], fontsize='larger')

    plt.sca(ax2)
    plt.imshow(data[1], cmap=cmap, aspect=1, norm=norm)
    plt.title(titles[1], fontsize='larger')

    plt.sca(ax3)
    plt.imshow(data[2], cmap=cmap, aspect=1, norm=norm)
    plt.title(titles[2], fontsize='larger')

    plt.colorbar(cax=cax)
    plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
    plt.savefig(fname)

    return


def histogram_triptic(centers, truth, prior, post, titles, labels, fname):
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
    fig.set_size_inches(6.0, 2.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    plt.sca(ax1)
    plt.semilogy(centers[0], truth[0])
    plt.semilogy(centers[1], prior[0])
    plt.semilogy(centers[2], post[0])
    plt.ylabel('pdf', fontsize='x-large')
    plt.xlabel(titles[0], fontsize='x-large')
    plt.grid(True)

    plt.sca(ax2)
    plt.semilogy(centers[0], truth[1])
    plt.semilogy(centers[1], prior[1])
    plt.semilogy(centers[2], post[1])
    plt.xlabel(titles[1], fontsize='x-large')
    plt.grid(True)

    plt.sca(ax3)
    plt.semilogy(centers[0], truth[2], label=labels[0])
    plt.semilogy(centers[1], prior[2], label=labels[1])
    plt.semilogy(centers[2], post[2], label=labels[2])
    plt.xlabel(titles[2], fontsize='x-large')
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), fontsize='large',
               borderpad=0.0, handlelength=1.5, borderaxespad=0.75)
    plt.grid(True)

    plt.suptitle(titles[3], y=1.01, fontsize='large')
    plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
    plt.savefig(fname)

    return


def Smagorinsky_SGS(U_hat, tau, C1=-6.39e-2, N=64):
    """
    Empty Docstring!
    """
    k0 = np.fft.fftfreq(N)*N
    k1 = np.fft.rfftfreq(N)*N
    K = np.array(np.meshgrid(k0, k0, k1, indexing='ij'), dtype=int)
    kmax = int(sqrt(2)*N/3)
    D = 2*pi/kmax
    les_filter = (np.sqrt(np.sum(K**2, axis=0)) < kmax).astype(np.int8)
    U_hat *= les_filter

    S = tau

    # --------------------------------------------------------------
    # Compute S_ij and |S|^2
    S_sqr = np.zeros((N, N, N))
    m = 0
    for i in range(3):
        for j in range(i, 3):
            Aij = 0.5j*K[j]*U_hat[i]

            if i == j:
                S[m] = np.fft.irfftn(2*Aij)
                S_sqr += S[m]**2

            else:
                Aji = 0.5j*K[i]*U_hat[j]
                S[m] = np.fft.irfftn(Aij + Aji)
                S_sqr += 2*S[m]**2

            m+=1

    tau[:] = C1*D**2*np.sqrt(2.0*S_sqr)*S

    # m == 0 -> ij == 00
    # m == 3 -> ij == 11
    # m == 5 -> ij == 22
    tau_kk = tau[0] + tau[3] + tau[5]
    # tau_rms = np.sqrt(np.sum(tau**2, axis=0))
    assert np.all(tau_kk < 1e-14)

    # tau_hat = np.fft.rfftn(tau[m])
    # KE_flux -= 1j*K[j]*tau_hat

    return


def Pope75_4term_SGS(U_hat, tau, C=None, N=64):
    """
    Empty Docstring!
    """
    k0 = np.fft.fftfreq(N)*N
    k1 = np.fft.rfftfreq(N)*N
    K = np.array(np.meshgrid(k0, k0, k1, indexing='ij'), dtype=int)
    kmax = int(sqrt(2)*N/3)
    D = 2*pi/kmax
    les_filter = (np.sqrt(np.sum(K**2, axis=0)) < kmax).astype(np.int8)
    U_hat *= les_filter

    S = np.empty((3, 3, N, N, N))
    R = np.empty((3, 3, N, N, N))

    # --------------------------------------------------------------
    # Compute S_ij and |S|^2
    S_sqr = np.zeros((N, N, N))
    R_sqr = np.zeros((N, N, N))

    for i in range(3):
        for j in range(3):
            Aij = 0.5j*K[j]*U_hat[i]

            if i == j:
                S[i, j] = np.fft.irfftn(2*Aij)
                R[i, j] = 0.0

            else:
                Aji = 0.5j*K[i]*U_hat[j]

                S[i, j] = np.fft.irfftn(Aij + Aji)
                R[i, j] = np.fft.irfftn(Aij - Aji)

            S_sqr += S[i, j]**2
            R_sqr += R[i, j]**2

    # --------------------------------------------------------------
    # Compute tau_ij = Delta**2 C_m G_ij^m and update RHS
    C = np.array(C)*D**2
    m = 0
    for i in range(3):
        for j in range(i, 3):

            # G_ij^1 = |S| S_ij
            tau[m] = C[0]*np.sqrt(2.0*S_sqr)*S[i, j]

            # G_ij^2 = -(S_ik R_jk + R_ik S_jk)
            tau[m] -= C[1]*np.sum(S[i]*R[j] + S[j]*R[i], axis=0)

            # G_ij^3 = S_ik S_jk - 1/3 delta_ij S_kl S_kl
            tau[m] += C[2]*np.sum(S[i]*S[j], axis=0)
            if i == j:
                tau[m] -= C[2]*(1/3)*S_sqr

            # G_ij^4 = - R_ik R_jk - 1/3 delta_ij R_kl R_kl
            tau[m] -= C[3]*np.sum(R[i]*R[j], axis=0)
            if i == j:
                tau[m] -= C[3]*(1/3)*R_sqr

            m+=1

    tau_kk = tau[0] + tau[3] + tau[5]
    # tau_rms = np.sqrt(np.sum(tau**2, axis=0))
    assert np.all(tau_kk < 1e-14)

    # tau_hat = np.fft.rfftn(tau[m])
    # KE_flux -= 1j*K[j]*tau_hat

    return


###############################################################################

# case = 'jhu_N64_abc_3term_test5'
eps = 0.103
nu = 0.000185
n = 5/3

N = 64
nk = (N//2)+1
ki = np.arange(nk)

# tf = 232
# ttags = ['%3.3d' % i for i in range(0, tf, 4)]
# nt = len(ttags)

demo_dir = '/Users/colin/workspace/teslapy/spectralLES/demo/analysis'
data_dir = '/Users/colin/workspace/teslapy/spectralLES/model_dev/analysis'
fig_dir = ('/Users/colin/workspace/teslapy/spectralLES/model_dev/figures/%s'
           % (time.strftime("%Y_%m_%d")))

try:
    os.makedirs(fig_dir)
except OSError:
    if not os.path.isdir(fig_dir):
        raise

# fname = '%s/%s-%s_KE.spectra' % (data_dir, case, ttags[0])
# with open(fname) as fh:
#     fh.readline()
#     metadata = fh.readline().rstrip('\n')

plt.loglog(ki[[1, -1]], [1.6, 1.6], 'k:', label='$C_k = 1.6$')
ymax = 1.6

# fname = '%s/%s-%s_KE.spectra' % (data_dir, case, ttags[0])
# spect = np.loadtxt(fname, skiprows=1)
# spect*= (ki**(5./3))*(eps**(-2/3))/N**6
# plt.loglog(ki[1:], spect[1:], 'b-.', lw=1, label='initial condition')
# ymax = max(ymax, spect.max())

# for i in range(1, nt):
#     fname = '%s/%s-%s_KE.spectra' % (data_dir, case, ttags[i])
#     spect = np.loadtxt(fname, skiprows=1)
#     spect*= (ki**(5./3))*(eps**(-2/3))/N**6
#     color=('%0.2f' % (0.3+0.01*i))
#     plt.loglog(ki[1:], spect[1:], c=color, lw=0.5)
#     ymax = max(ymax, spect.max())

fname = '%s/jhu_N64_nomodel_test-189_KE.spectra' % demo_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'm--', lw=1.0, label='no model')

fname = '%s/jhu_N64_staticSmag_test-225_KE.spectra' % demo_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'm-', lw=1.0,
           label='Smagorinsky $(C_s \!=\! 0.098)$')

fname = '%s/jhu_N64_abc_staticSmag_test-208_KE.spectra' % data_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'b--', lw=1.0,
           label='1-Term ABC $(C_s \!=\! 0.179)$')

fname = '%s/jhu_N64_abc_3term_test5-232_KE.spectra' % data_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'b-.', lw=1.5, label='3-Term ABC')

fname = '%s/jhu_N64_abc_4term_test5-193_KE.spectra' % data_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'b-', lw=2.0, label='4-Term ABC')

fname = '%s/jhu_N64_ales244_test-054_KE.spectra' % data_dir
spect = np.loadtxt(fname, skiprows=1)
spect*= (ki**n)*(eps**(-2/3))/N**6
ymax = max(ymax, spect.max())
plt.loglog(ki[1:], spect[1:], 'g-', lw=2.0, label='244-Term ALES')

# fname = '%s/%s-%s_KE.spectra' % (data_dir, case, '%3.3d' % tf)
# spect = np.loadtxt(fname, skiprows=1)
# spect*= (ki**(5./3))*(eps**(-2/3))/N**6
# ymax = max(ymax, spect.max())
# plt.loglog(ki[1:], spect[1:], 'c-', lw=2, label='final spectrum')

plt.ylim([1e-2, 100])
plt.grid(True)
plt.xlabel('$k$')
plt.ylabel('$k^{-1/3}\\varepsilon^{-2/3}E_\\Omega(k)$')
plt.title('ALES/ABC Forward Run Spectra')
plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0))

plt.gcf().set_size_inches(4.75, 3.0)
plt.savefig('%s/ALES_comparison-KE_spectra.png' % fig_dir)

plt.clf()
'''
###############################################################################

demo_dir = '/Users/colin/workspace/teslapy/spectralLES/demo/data'
data_dir = '/Users/colin/workspace/teslapy/spectralLES/model_dev/data'

vtitles = ['$u_1$', '$u_2$', '$u_3$']
ttitles = ['$\\tau_{11}$', '$\\tau_{12}$', '$\\tau_{13}$']

vel_norm = colors.Normalize(vmin=-2.0, vmax=2.0)
vlim = 0.05
norm = colors.Normalize(vmin=-vlim, vmax=vlim)

U = np.empty((3, N, N, N))
U_hat = np.empty((3, N, N, nk), dtype=complex)
tau = np.empty((6, N, N, N))

C3 = [-3.75e-02, 6.2487e-02, 6.9867e-03, 0.0]
C4 = [-3.15e-02, -5.25e-02, 2.7e-02, 2.7e-02]

# SPARSE SYMMETRIC TENSOR INDEXING:
# m == 0 -> ij == 00
# m == 1 -> ij == 01
# m == 2 -> ij == 02
# m == 3 -> ij == 11
# m == 4 -> ij == 12
# m == 5 -> ij == 22

# -----------------------------------------------------------------------------
ttag = '%3.3d' % 16
for i in range(3):
    case = 'jhu_N64_staticSmag_test'
    fname = '%s/%s-Velocity%d_%s.rst' % (demo_dir, case, i+1, ttag)
    U[i] = np.fromfile(fname).reshape(N, N, N)

iname = '%s/velocity_slices-staticSmag.png' % fig_dir
image_triptic(U[:, 0], vel_norm, 'RdBu', vtitles, iname)

U_hat[0] = np.fft.rfftn(U[0])
U_hat[1] = np.fft.rfftn(U[1])
U_hat[2] = np.fft.rfftn(U[2])

Ck = 1.6
Cs = sqrt((pi**-2)*((3*Ck)**-1.5))  # == 0.098...
Smagorinsky_SGS(U_hat, tau, C1=-2*Cs**2, N=N)

iname = '%s/tau_slices-staticSmag.png' % fig_dir
image_triptic(tau[:3, 0], norm, 'RdBu', ttitles, iname)

# -----------------------------------------------------------------------------
ttag = '%3.3d' % 18
for i in range(3):
    case = 'jhu_N64_abc_staticSmag_test'
    fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
    U[i] = np.fromfile(fname).reshape(N, N, N)

iname = '%s/velocity_slices-1term.png' % fig_dir
image_triptic(U[:, 0], vel_norm, 'RdBu', vtitles, iname)

U_hat[0] = np.fft.rfftn(U[0])
U_hat[1] = np.fft.rfftn(U[1])
U_hat[2] = np.fft.rfftn(U[2])
Smagorinsky_SGS(U_hat, tau, C1=-6.39e-2, N=N)

iname = '%s/tau_slices-1term.png' % fig_dir
image_triptic(tau[:3, 0], norm, 'RdBu', ttitles, iname)


# -----------------------------------------------------------------------------
ttag = '%3.3d' % 18
for i in range(3):
    case = 'jhu_N64_abc_3term_test5'
    fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
    U[i] = np.fromfile(fname).reshape(N, N, N)

iname = '%s/velocity_slices-3term.png' % fig_dir
image_triptic(U[:, 0], vel_norm, 'RdBu', vtitles, iname)

U_hat[0] = np.fft.rfftn(U[0])
U_hat[1] = np.fft.rfftn(U[1])
U_hat[2] = np.fft.rfftn(U[2])
Pope75_4term_SGS(U_hat, tau, C=C3, N=N)

iname = '%s/tau_slices-3term.png' % fig_dir
image_triptic(tau[:3, 0], norm, 'RdBu', ttitles, iname)

# -----------------------------------------------------------------------------
ttag = '%3.3d' % 18
for i in range(3):
    case = 'jhu_N64_abc_4term_test5'
    fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
    U[i] = np.fromfile(fname).reshape(N, N, N)

iname = '%s/velocity_slices-4term.png' % fig_dir
image_triptic(U[:, 0], vel_norm, 'RdBu', vtitles, iname)

U_hat[0] = np.fft.rfftn(U[0])
U_hat[1] = np.fft.rfftn(U[1])
U_hat[2] = np.fft.rfftn(U[2])
Pope75_4term_SGS(U_hat, tau, C=C4, N=N)

iname = '%s/tau_slices-4term.png' % fig_dir
image_triptic(tau[:3, 0], norm, 'RdBu', ttitles, iname)

###############################################################################
truth_dir = '/Users/colin/Google Drive/workspace/data/spectralLES'
nbins = 64
post1 = np.zeros((3, nbins))
post3 = np.zeros((3, nbins))
post4 = np.zeros((3, nbins))

labels = ['truth',
          'a priori',
          'a posteriori']
titles = ['$\\tau_{11}$', '$\\tau_{12}$', '$\\tau_{13}$', '']

width = 0.6/nbins
centers = [[], [], []]
centers[2] = np.linspace(-0.3+0.5*width, 0.3-0.5*width, nbins)

for j in range(11, 19):
    ttag = '%3.3d' % j

    for i in range(3):
        case = 'jhu_N64_abc_staticSmag_test'
        fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
        U[i] = np.fromfile(fname).reshape(N, N, N)

    U_hat[0] = np.fft.rfftn(U[0])
    U_hat[1] = np.fft.rfftn(U[1])
    U_hat[2] = np.fft.rfftn(U[2])
    Smagorinsky_SGS(U_hat, tau, C1=-6.39e-2, N=N)

    for m in range(3):
        post1[m] += np.histogram(tau[m], bins=nbins, range=(-0.3, 0.3))[0]

    # ------------------------------------------------------------------
    for i in range(3):
        case = 'jhu_N64_abc_3term_test5'
        fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
        U[i] = np.fromfile(fname).reshape(N, N, N)

    U_hat[0] = np.fft.rfftn(U[0])
    U_hat[1] = np.fft.rfftn(U[1])
    U_hat[2] = np.fft.rfftn(U[2])
    Pope75_4term_SGS(U_hat, tau, C=C3, N=N)

    for m in range(3):
        post3[m] += np.histogram(tau[m], bins=nbins, range=(-0.3, 0.3))[0]

    # ------------------------------------------------------------------
    for i in range(3):
        case = 'jhu_N64_abc_4term_test5'
        fname = '%s/%s-Velocity%d_%s.rst' % (data_dir, case, i+1, ttag)
        U[i] = np.fromfile(fname).reshape(N, N, N)

    U_hat[0] = np.fft.rfftn(U[0])
    U_hat[1] = np.fft.rfftn(U[1])
    U_hat[2] = np.fft.rfftn(U[2])
    Pope75_4term_SGS(U_hat, tau, C=C4, N=N)

    for m in range(3):
        post4[m] += np.histogram(tau[m], bins=nbins, range=(-0.3, 0.3))[0]

post1 = post1/post1.sum()
post3 = post3/post3.sum()
post4 = post4/post4.sum()

# -----------------------------------------------------------------------------
centers[1] = np.loadtxt('%s/1term/sum_stat_bins' % truth_dir)
centers[0] = centers[1]
truth = np.loadtxt('%s/1term/sum_stat_true' % truth_dir)
prior = np.loadtxt('%s/1term/sum_stat_min_dist_TEST' % truth_dir)

truth = np.exp(truth)
prior = np.exp(prior)
truth *= 1/truth.sum()
prior *= 1/prior.sum()

fname = '%s/tau_pdfs-1term.png' % fig_dir
titles[3] = '1-Term Model SGS Stress Comparison'
histogram_triptic(centers, truth, prior, post1, titles, labels, fname)

# -----------------------------------------------------------------------------
centers[1] = np.loadtxt('%s/3term/sum_stat_bins' % truth_dir)
centers[0] = centers[1]
truth = np.loadtxt('%s/3term/sum_stat_true' % truth_dir)
prior = np.loadtxt('%s/3term/sum_stat_max_joint_TEST' % truth_dir)

truth = np.exp(truth)
prior = np.exp(prior)
truth *= 1/truth.sum()
prior *= 1/prior.sum()

fname = '%s/tau_pdfs-3term.png' % fig_dir
titles[3] = '3-Term Model SGS Stress Comparison'
histogram_triptic(centers, truth, prior, post3, titles, labels, fname)

# -----------------------------------------------------------------------------
centers[1] = np.loadtxt('%s/4term/sum_stat_bins' % truth_dir)
centers[0] = centers[1]
truth = np.loadtxt('%s/4term/sum_stat_true' % truth_dir)
prior = np.loadtxt('%s/4term/sum_stat_max_joint_TEST' % truth_dir)

truth = np.exp(truth)
prior = np.exp(prior)
truth *= 1/truth.sum()
prior *= 1/prior.sum()

fname = '%s/tau_pdfs-4term.png' % fig_dir
titles[3] = '4-Term Model SGS Stress Comparison'
histogram_triptic(centers, truth, prior, post4, titles, labels, fname)

# -----------------------------------------------------------------------------
centers[0] = centers[1] = centers[2]
fname = '%s/model_comparison-tau_pdfs.png' % fig_dir
labels = ['1-term', '3-term', '4-term']
titles[3] = 'Comparison of Model SGS Stresses'
histogram_triptic(centers, post1, post3, post4, titles, labels, fname)
'''
# -----------------------------------------------------------------------------
plt.close('all')
